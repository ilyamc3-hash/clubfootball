"""
Football Prediction Lab — ablation test.

Сравниваем две версии модели на одинаковых матчах (walk-forward,
без подглядывания в будущее):

  BASELINE : только Elo (с margin of victory), без Form/Momentum
  FULL     : Elo + Form + Momentum (как в predict_match.py)

Если FULL не лучше BASELINE — значит Form/Momentum пока не несут
полезного сигнала в текущем виде, и не стоит их "докручивать" дальше,
пока не появится больше данных.

Использование:
    py ablation_test.py
"""

import sqlite3
import statistics
import math

from fifa_ratings import get_starting_elo

DB_PATH = "football.db"
BASE_ELO = 1500
K_FACTOR = 32
MIN_HISTORY = 1


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


def evaluate(matches, use_form_momentum):
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
            effective_diff = elo_a - elo_b

            if use_form_momentum:
                form_a, form_b = form_for(seq_home), form_for(seq_away)
                mom_a, mom_b = momentum_for(seq_home), momentum_for(seq_away)
                effective_diff += (form_a - form_b) * 15 + (mom_a - mom_b) * 15

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
        "n": n,
        "accuracy": correct / total_evaluated,
        "brier": brier_sum / n / 3,
        "logloss": logloss_sum / n,
    }


def main():
    conn = sqlite3.connect(DB_PATH)
    matches = load_matches(conn)
    conn.close()

    baseline = evaluate(matches, use_form_momentum=False)
    full = evaluate(matches, use_form_momentum=True)

    print(f"{'Метрика':<15}{'BASELINE (Elo)':>18}{'FULL (Elo+Form+Mom)':>22}{'Разница':>12}")
    print("-" * 67)
    print(f"{'Матчей':<15}{baseline['n']:>18}{full['n']:>22}")
    print(f"{'Accuracy':<15}{baseline['accuracy']*100:>17.1f}%{full['accuracy']*100:>21.1f}%"
          f"{(full['accuracy']-baseline['accuracy'])*100:>+11.1f}%")
    print(f"{'Brier score':<15}{baseline['brier']:>18.4f}{full['brier']:>22.4f}"
          f"{full['brier']-baseline['brier']:>+12.4f}")
    print(f"{'Log loss':<15}{baseline['logloss']:>18.4f}{full['logloss']:>22.4f}"
          f"{full['logloss']-baseline['logloss']:>+12.4f}")

    print("\n(Для Brier и Log loss отрицательная разница = FULL лучше, положительная = FULL хуже)")


if __name__ == "__main__":
    main()
