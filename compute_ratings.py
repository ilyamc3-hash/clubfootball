"""
Football Prediction Lab — v0.1 рейтинги команд.

Считает 4 показателя по матчам, уже сохранённым в football.db:
  - Elo         : общий рейтинг силы (с учётом силы соперника)
  - Form        : взвешенная форма внутри турнира (свежие игры весят больше)
  - Stability   : стабильность результатов (низкий разброс = стабильна)
  - Momentum    : тренд — команда набирает или теряет ход по турниру

Важно: для ЧМ это НЕ скользящее окно "последние N месяцев", как в АПЛ/Ла Лиге,
а весь турнир целиком — потому что у сборной всего 4-7 матчей за 3 недели,
и никакой более длинной истории формы тут просто нет.

Использование:
    py compute_ratings.py
"""

import sqlite3
import statistics
import math

from fifa_ratings import get_starting_elo

DB_PATH = "football.db"
BASE_ELO = 1500
K_FACTOR = 32


def mov_multiplier(goal_diff, elo_diff_pre_match):
    """Множитель на основе разницы мячей (margin of victory).
    Логарифм гасит эффект крупных разгромов (5:0 не в 5 раз важнее 1:0).
    Знаменатель гасит бонус, если фаворит и так был намного сильнее —
    разгром аутсайдера ожидаем и не должен сильно поднимать рейтинг."""
    goal_diff = abs(goal_diff)
    if goal_diff == 0:
        return 1.0
    return math.log(goal_diff + 1) * (2.2 / ((abs(elo_diff_pre_match) * 0.001) + 2.2))


def load_matches(conn):
    """Берём только сыгранные матчи, по результату ОСНОВНОГО времени —
    исход по пенальти не отражает реальную силу команды, это лотерея."""
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

        if hg > ag:
            score_home = 1.0
        elif hg < ag:
            score_home = 0.0
        else:
            score_home = 0.5  # ничья по факту основного времени, даже если был буллитный сериал

        mult = mov_multiplier(hg - ag, r_home - r_away)
        k_effective = K_FACTOR * mult

        elo[home] = r_home + k_effective * (score_home - exp_home)
        elo[away] = r_away + k_effective * ((1 - score_home) - (1 - exp_home))

    return elo


def team_match_sequence(matches):
    """Для каждой команды — список (гол за, гол против) в хронологическом порядке."""
    seq = {}
    for _, home, away, hg, ag in matches:
        seq.setdefault(home, []).append((hg, ag))
        seq.setdefault(away, []).append((ag, hg))
    return seq


def compute_form(seq):
    """Взвешенная форма: свежие игры весят больше. Победа +4, ничья +1, поражение -3."""
    form = {}
    for team, games in seq.items():
        total, weight_sum = 0.0, 0.0
        n = len(games)
        for i, (gf, ga) in enumerate(games):
            weight = i + 1  # чем позже игра, тем больше вес (1, 2, 3...)
            if gf > ga:
                points = 4
            elif gf == ga:
                points = 1
            else:
                points = -3
            total += points * weight
            weight_sum += weight
        form[team] = round(total / weight_sum, 2) if weight_sum else 0.0
    return form


def compute_stability(seq):
    """Разброс разницы мячей: чем меньше дисперсия, тем стабильнее команда."""
    stability = {}
    for team, games in seq.items():
        diffs = [gf - ga for gf, ga in games]
        if len(diffs) >= 2:
            variance = statistics.pvariance(diffs)
        else:
            variance = 0.0
        # переводим в шкалу 0-100, где 100 = максимально стабильна
        stability[team] = round(max(0, 100 - variance * 8), 1)
    return stability


def compute_momentum(seq):
    """Тренд забитых голов внутри турнира: сравниваем вторую половину игр с первой."""
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
        momentum[team] = round(statistics.mean(late) - statistics.mean(early), 2)
    return momentum


def main():
    conn = sqlite3.connect(DB_PATH)
    matches = load_matches(conn)
    print(f"Загружено сыгранных матчей: {len(matches)}\n")

    elo = compute_elo(matches)
    seq = team_match_sequence(matches)
    form = compute_form(seq)
    stability = compute_stability(seq)
    momentum = compute_momentum(seq)

    teams = sorted(elo.keys(), key=lambda t: elo[t], reverse=True)

    print(f"{'Команда':<20}{'Elo':>8}{'Form':>8}{'Stability':>12}{'Momentum':>10}")
    print("-" * 60)
    for t in teams:
        print(f"{t:<20}{elo[t]:>8.0f}{form.get(t, 0):>8.2f}{stability.get(t, 0):>12.1f}{momentum.get(t, 0):>+10.2f}")

    conn.close()


if __name__ == "__main__":
    main()
