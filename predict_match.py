"""
Football Prediction Lab — v0.1 прогноз исхода матча.

Берёт рейтинги (Elo, Form, Momentum) из football.db и считает вероятности
П1 / Х / П2 для конкретной пары команд.

ВАЖНО (честно, без прикрас):
Веса, с которыми Form и Momentum подмешиваются к Elo, а также формула
вероятности ничьей — это ЭВРИСТИКА v0.1, подобранная "на глаз", а не
откалиброванная на исторических данных. Настоящая калибровка (как
описано в плане проекта) требует тысяч матчей и backtesting — тут пока
не тот объём данных. Это первая грубая версия для проверки идеи,
не окончательная точность.

Использование:
    py predict_match.py
"""

import sqlite3
import statistics
import math
from datetime import datetime

from fifa_ratings import get_starting_elo

DB_PATH = "football.db"
BASE_ELO = 1500
K_FACTOR = 32


def mov_multiplier(goal_diff, elo_diff_pre_match):
    """Множитель на основе разницы мячей (margin of victory) — см. compute_ratings.py."""
    goal_diff = abs(goal_diff)
    if goal_diff == 0:
        return 1.0
    return math.log(goal_diff + 1) * (2.2 / ((abs(elo_diff_pre_match) * 0.001) + 2.2))

# Если оставить пустым списком [] — скрипт сам найдёт ближайшие
# запланированные матчи (ищет ближайшую дату среди ещё не сыгранных).
# Можно по-прежнему задать вручную конкретные пары вместо автопоиска:
MATCHES_TO_PREDICT = []


def find_upcoming_matches(conn):
    """Берём ещё не сыгранные матчи и возвращаем все, что попадают в
    окно ~20 часов от самого раннего — так матчи одной "игровой ночи",
    но по разные стороны полуночи UTC, не теряются."""
    cur = conn.execute("""
        SELECT m.utc_date, t1.name, t2.name
        FROM matches m
        JOIN teams t1 ON m.home_team_id = t1.id
        JOIN teams t2 ON m.away_team_id = t2.id
        WHERE m.status IN ('TIMED', 'SCHEDULED')
          AND t1.name IS NOT NULL AND t2.name IS NOT NULL
        ORDER BY m.utc_date ASC
    """)
    rows = cur.fetchall()
    if not rows:
        return [], None

    fmt = "%Y-%m-%dT%H:%M:%SZ"
    earliest = datetime.strptime(rows[0][0], fmt)
    window_hours = 20

    same_window = []
    for date, h, a in rows:
        dt = datetime.strptime(date, fmt)
        if (dt - earliest).total_seconds() <= window_hours * 3600:
            same_window.append((h, a))

    return same_window, rows[0][0][:16]


def load_matches(conn):
    cur = conn.execute("""
        SELECT m.utc_date, t1.name, t2.name, m.regular_home, m.regular_away
        FROM matches m
        JOIN teams t1 ON m.home_team_id = t1.id
        JOIN teams t2 ON m.away_team_id = t2.id
        WHERE m.status = 'FINISHED'
          AND m.regular_home IS NOT NULL
        ORDER BY m.utc_date ASC
    """)
    return cur.fetchall()


def compute_elo(matches):
    elo = {}

    def get(team):
        return elo.setdefault(team, get_starting_elo(team))

    for _, home, away, hg, ag in matches:
        r_home, r_away = get(home), get(away)
        exp_home = 1 / (1 + 10 ** ((r_away - r_home) / 400))
        score_home = 1.0 if hg > ag else (0.0 if hg < ag else 0.5)
        mult = mov_multiplier(hg - ag, r_home - r_away)
        k_effective = K_FACTOR * mult
        elo[home] = r_home + k_effective * (score_home - exp_home)
        elo[away] = r_away + k_effective * ((1 - score_home) - (1 - exp_home))
    return elo


def team_match_sequence(matches):
    seq = {}
    for _, home, away, hg, ag in matches:
        seq.setdefault(home, []).append((hg, ag))
        seq.setdefault(away, []).append((ag, hg))
    return seq


def compute_form(seq):
    form = {}
    for team, games in seq.items():
        total, weight_sum = 0.0, 0.0
        for i, (gf, ga) in enumerate(games):
            weight = i + 1
            points = 4 if gf > ga else (1 if gf == ga else -3)
            total += points * weight
            weight_sum += weight
        form[team] = total / weight_sum if weight_sum else 0.0
    return form


def compute_momentum(seq):
    momentum = {}
    for team, games in seq.items():
        goals = [gf for gf, _ in games]
        n = len(goals)
        if n < 2:
            momentum[team] = 0.0
            continue
        mid = n // 2
        early = goals[:mid] or goals[:1]
        late = goals[mid:] or goals[-1:]
        momentum[team] = statistics.mean(late) - statistics.mean(early)
    return momentum


def draw_probability(effective_diff):
    """Эвристика: чем ближе команды, тем выше шанс ничьей.
    При equal силах ~28%, падает по мере роста разницы. Не откалибровано."""
    base = 0.28
    penalty = min(0.20, (abs(effective_diff) / 400) ** 1.5 * 0.18)
    return max(0.06, base - penalty)


def predict(team_a, team_b, elo, form, momentum):
    elo_a, elo_b = elo.get(team_a, BASE_ELO), elo.get(team_b, BASE_ELO)
    form_a, form_b = form.get(team_a, 0.0), form.get(team_b, 0.0)
    mom_a, mom_b = momentum.get(team_a, 0.0), momentum.get(team_b, 0.0)

    # Грубая поправка: разницу Form/Momentum переводим в "эло-очки" с небольшим весом
    effective_diff = (elo_a - elo_b) + (form_a - form_b) * 15 + (mom_a - mom_b) * 15

    expected_a = 1 / (1 + 10 ** (-effective_diff / 400))  # ~ P(win) + 0.5*P(draw)
    p_draw = draw_probability(effective_diff)
    p_a = max(0.0, expected_a - p_draw / 2)
    p_b = max(0.0, (1 - expected_a) - p_draw / 2)

    # нормализация, чтобы сумма была ровно 1
    total = p_a + p_draw + p_b
    return p_a / total, p_draw / total, p_b / total, effective_diff


def main():
    conn = sqlite3.connect(DB_PATH)
    matches = load_matches(conn)
    elo = compute_elo(matches)
    seq = team_match_sequence(matches)
    form = compute_form(seq)
    momentum = compute_momentum(seq)

    matches_to_predict = MATCHES_TO_PREDICT
    if not matches_to_predict:
        matches_to_predict, nearest_date = find_upcoming_matches(conn)
        if not matches_to_predict:
            print("Не найдено предстоящих матчей в базе. Обнови данные через fetch_matches.py.")
            conn.close()
            return
        print(f"Автоматически найдены ближайшие матчи (дата: {nearest_date}):")

    conn.close()

    for team_a, team_b in matches_to_predict:
        p_a, p_draw, p_b, diff = predict(team_a, team_b, elo, form, momentum)
        advance_a = p_a + p_draw / 2  # ничья в плей-офф -> серия пенальти ~50/50
        advance_b = p_b + p_draw / 2

        print(f"\n{team_a} — {team_b}")
        print(f"  Elo: {elo.get(team_a, BASE_ELO):.0f} vs {elo.get(team_b, BASE_ELO):.0f}  "
              f"(эффективная разница с поправкой на форму/momentum: {diff:+.0f})")
        print(f"  П1 (основное время): {p_a*100:.1f}%")
        print(f"  Х  (основное время): {p_draw*100:.1f}%")
        print(f"  П2 (основное время): {p_b*100:.1f}%")
        print(f"  Вероятность выхода в полуфинал: {team_a} {advance_a*100:.1f}% "
              f"— {team_b} {advance_b*100:.1f}%")


if __name__ == "__main__":
    main()
