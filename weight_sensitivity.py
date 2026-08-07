"""
Football Prediction Lab — подбор веса Form/Momentum через backtest.

Идея: вместо того чтобы менять вес "на глаз" под конкретный матч
(Норвегия-Англия), прогоняем весь backtest с разными весами и смотрим,
какой из них ДЕЙСТВИТЕЛЬНО улучшает метрики на 74 исторических матчах.

Использование:
    py weight_sensitivity.py
"""

import sqlite3
import statistics
import math

from fifa_ratings import get_starting_elo

DB_PATH = "football.db"
BASE_ELO = 1500
K_FACTOR = 32
MIN_HISTORY = 1

WEIGHTS_TO_TEST = [0, 10, 15, 20, 30, 40, 50, 65, 80, 100]


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


def evaluate(matches, weight):
    elo = {}
    seq = {}

    def get_elo(t):
        return elo.setdefault(t, get_starting_elo(t))

    results = []
    correct, total_evaluated = 0, 0

    for _, home, away, hg, ag in matches:
        seq_home = seq.get(home, [])
        seq_away = seq.get(away, [])

        if len(seq_home) >= MIN_HISTORY and len(seq_away) >= MIN_HISTORY:
            elo_a, elo_b = get_elo(home), get_elo(away)
            form_a, form_b = form_for(seq_home), form_for(seq_away)
            mom_a, mom_b = momentum_for(seq_home), momentum_for(seq_away)

            effective_diff = (elo_a - elo_b) + (form_a - form_b) * weight + (mom_a - mom_b) * weight

            p_home, p_draw, p_away = predict_probs(effective_diff)
            actual = 0 if hg > ag else (1 if hg == ag else 2)
            probs = [p_home, p_draw, p_away]
            results.append((probs, actual))

            if probs.index(max(probs)) == actual:
                correct += 1
            total_evaluated += 1

        r_home, r_away = get_elo(home), get_elo(away)
        exp_home = 1 / (1 + 10 ** ((r_away - r_home) / 400))
        score_home = 1.0 if hg > ag else (0.0 if hg < ag else 0.5)
        mult = mov_multiplier(hg - ag, r_home - r_away)
        k_eff = K_FACTOR * mult
        elo[home] = r_home + k_eff * (score_home - exp_home)
        elo[away] = r_away + k_eff * ((1 - score_home) - (1 - exp_home))

        seq.setdefault(home, []).append((hg, ag))
        seq.setdefault(away, []).append((ag, hg))

    eps = 1e-15
    brier_sum, logloss_sum = 0.0, 0.0
    for probs, actual in results:
        for i, p in enumerate(probs):
            target = 1.0 if i == actual else 0.0
            brier_sum += (p - target) ** 2
        logloss_sum += -math.log(max(probs[actual], eps))

    n = len(results)
    return {
        "accuracy": correct / total_evaluated,
        "brier": brier_sum / n / 3,
        "logloss": logloss_sum / n,
    }


def main():
    conn = sqlite3.connect(DB_PATH)
    matches = load_matches(conn)
    conn.close()

    print(f"{'Вес':<8}{'Accuracy':>12}{'Brier':>12}{'Log loss':>12}")
    print("-" * 44)

    best_brier = None
    best_weight = None

    for w in WEIGHTS_TO_TEST:
        m = evaluate(matches, w)
        if best_brier is None or m["brier"] < best_brier:
            best_brier = m["brier"]
            best_weight = w
        print(f"{w:<8}{m['accuracy']*100:>11.1f}%{m['brier']:>12.4f}{m['logloss']:>12.4f}")

    print(f"\nЛучший Brier score при весе = {best_weight} (Brier {best_brier:.4f})")
    print("Если оптимум лежит на краю диапазона (0 или 100) — стоит расширить диапазон теста.")


if __name__ == "__main__":
    main()
