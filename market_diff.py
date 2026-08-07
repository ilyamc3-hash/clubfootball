"""
Football Prediction Lab — сравнение с рынком (market_diff).

У нас нет бесплатного доступа к live-коэффициентам (football-data.org
просит платный Odds-Package), поэтому коэффициенты вводятся ВРУЧНУЮ —
ты находишь их сам (сайт букмекера, поиск) и вписываешь в MANUAL_ODDS.

ВАЖНО, честно: большое расхождение с рынком — это НЕ повод думать
"нашли недооценённое событие". У букмекера есть данные, которых нет у
нас (составы, инсайды, миллионы ставок). Если модель сильно расходится
с рынком — в 9 случаях из 10 это значит, что ОШИБАЕТСЯ МОДЕЛЬ, а не рынок.
Этот скрипт — инструмент диагностики себя, а не поиска "лёгких денег".

Использование:
    py market_diff.py
"""

import sqlite3
import statistics
import math

from fifa_ratings import get_starting_elo

DB_PATH = "football.db"
K_FACTOR = 32
FORM_MOMENTUM_WEIGHT = 15

# ---- Впиши сюда коэффициенты вручную (основное время: победа / ничья / победа) ----
MANUAL_ODDS = {
    ("Norway", "England"): {"home": 4.05, "draw": 3.70, "away": 1.93},
    ("Argentina", "Switzerland"): {"home": 1.71, "draw": None, "away": None},  # пример неполных данных
}

DIVERGENCE_WARNING_THRESHOLD = 10.0  # процентных пунктов — выше этого выводим предупреждение


def mov_multiplier(goal_diff, elo_diff_pre_match):
    goal_diff = abs(goal_diff)
    if goal_diff == 0:
        return 1.0
    return math.log(goal_diff + 1) * (2.2 / ((abs(elo_diff_pre_match) * 0.001) + 2.2))


def draw_probability(effective_diff):
    base = 0.28
    penalty = min(0.20, (abs(effective_diff) / 400) ** 1.5 * 0.18)
    return max(0.06, base - penalty)


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


def form_for(seq_so_far):
    total, weight_sum = 0.0, 0.0
    for i, (gf, ga) in enumerate(seq_so_far):
        weight = i + 1
        points = 4 if gf > ga else (1 if gf == ga else -3)
        total += points * weight
        weight_sum += weight
    return total / weight_sum if weight_sum else 0.0


def momentum_for(seq_so_far):
    goals = [gf for gf, _ in seq_so_far]
    n = len(goals)
    if n < 2:
        return 0.0
    mid = n // 2
    early = goals[:mid] or goals[:1]
    late = goals[mid:] or goals[-1:]
    return statistics.mean(late) - statistics.mean(early)


def predict_probs(effective_diff):
    expected_a = 1 / (1 + 10 ** (-effective_diff / 400))
    p_draw = draw_probability(effective_diff)
    p_a = max(0.0, expected_a - p_draw / 2)
    p_b = max(0.0, (1 - expected_a) - p_draw / 2)
    total = p_a + p_draw + p_b
    return p_a / total, p_draw / total, p_b / total


def build_model(matches):
    elo = {}
    seq = {}

    def get_elo(t):
        return elo.setdefault(t, get_starting_elo(t))

    for _, home, away, hg, ag in matches:
        r_home, r_away = get_elo(home), get_elo(away)
        exp_home = 1 / (1 + 10 ** ((r_away - r_home) / 400))
        score_home = 1.0 if hg > ag else (0.0 if hg < ag else 0.5)
        mult = mov_multiplier(hg - ag, r_home - r_away)
        k_eff = K_FACTOR * mult
        elo[home] = r_home + k_eff * (score_home - exp_home)
        elo[away] = r_away + k_eff * ((1 - score_home) - (1 - exp_home))
        seq.setdefault(home, []).append((hg, ag))
        seq.setdefault(away, []).append((ag, hg))

    return elo, seq


def model_predict(team_a, team_b, elo, seq):
    elo_a, elo_b = elo.get(team_a, get_starting_elo(team_a)), elo.get(team_b, get_starting_elo(team_b))
    seq_a, seq_b = seq.get(team_a, []), seq.get(team_b, [])
    form_a, form_b = form_for(seq_a), form_for(seq_b)
    mom_a, mom_b = momentum_for(seq_a), momentum_for(seq_b)
    effective_diff = (elo_a - elo_b) + (form_a - form_b) * FORM_MOMENTUM_WEIGHT + (mom_a - mom_b) * FORM_MOMENTUM_WEIGHT
    return predict_probs(effective_diff)


def implied_probs_from_odds(odds):
    """Коэффициенты -> вероятности, с нормализацией (убираем маржу букмекера)."""
    raw = {}
    for key, value in odds.items():
        raw[key] = (1 / value) if value else None

    known = {k: v for k, v in raw.items() if v is not None}
    total = sum(known.values())
    normalized = {k: (v / total if v is not None else None) for k, v in raw.items()}
    overround_pct = (total - 1) * 100
    return normalized, overround_pct


def main():
    conn = sqlite3.connect(DB_PATH)
    matches = load_matches(conn)
    conn.close()

    elo, seq = build_model(matches)

    for (team_a, team_b), odds in MANUAL_ODDS.items():
        p_model_a, p_model_draw, p_model_b = model_predict(team_a, team_b, elo, seq)
        p_market, overround = implied_probs_from_odds(odds)

        print(f"{team_a} — {team_b}")
        print(f"  {'Исход':<8}{'Наша модель':>14}{'Рынок (норм.)':>16}{'Разница':>12}")

        labels = [("home", "П1", p_model_a), ("draw", "Х", p_model_draw), ("away", "П2", p_model_b)]
        max_gap = 0.0

        for key, label, model_p in labels:
            market_p = p_market.get(key)
            if market_p is None:
                print(f"  {label:<8}{model_p*100:>13.1f}%{'нет данных':>16}")
                continue
            gap = (model_p - market_p) * 100
            max_gap = max(max_gap, abs(gap))
            print(f"  {label:<8}{model_p*100:>13.1f}%{market_p*100:>15.1f}%{gap:>+11.1f}п.п.")

        print(f"  (маржа букмекера в этих котировках: ~{overround:.1f}%)")

        if max_gap >= DIVERGENCE_WARNING_THRESHOLD:
            print(f"  ⚠ Расхождение {max_gap:.1f} п.п. — скорее всего, ОШИБКА НАШЕЙ МОДЕЛИ,")
            print(f"    а не признак того, что рынок недооценил событие. Стоит перепроверить,")
            print(f"    почему модель здесь так сильно расходится с рынком, а не считать это находкой.")
        print()


if __name__ == "__main__":
    main()
