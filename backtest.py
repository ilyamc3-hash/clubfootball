"""
Football Prediction Lab — backtesting v0.1 (с margin of victory).

Идея: проходим по всем сыгранным матчам В ХРОНОЛОГИЧЕСКОМ ПОРЯДКЕ.
Перед каждым матчем считаем рейтинги ТОЛЬКО по предыдущим играм
(иначе модель "подглядывает" в будущее — это была бы читерская проверка).
Затем сравниваем прогноз с реальным результатом.

Метрики:
  - Accuracy   : как часто угадан исход (просто для интуиции)
  - Brier score: точность калибровки вероятностей, 0 = идеально, ~0.67 = случайно
  - Log loss   : похожая идея, сильнее штрафует за уверенные, но неверные прогнозы

Использование:
    py backtest.py
"""

import sqlite3
import statistics
import math

from fifa_ratings import get_starting_elo

DB_PATH = "football.db"
BASE_ELO = 1500
K_FACTOR = 32
MIN_HISTORY = 1  # минимум сыгранных матчей у обеих команд, чтобы включать в оценку


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


def predict_pre_match(elo, team_a, team_b, seq_a, seq_b):
    elo_a, elo_b = elo.get(team_a, BASE_ELO), elo.get(team_b, BASE_ELO)
    form_a, form_b = form_for(seq_a), form_for(seq_b)
    mom_a, mom_b = momentum_for(seq_a), momentum_for(seq_b)

    effective_diff = (elo_a - elo_b) + (form_a - form_b) * 15 + (mom_a - mom_b) * 15
    expected_a = 1 / (1 + 10 ** (-effective_diff / 400))
    p_draw = draw_probability(effective_diff)
    p_a = max(0.0, expected_a - p_draw / 2)
    p_b = max(0.0, (1 - expected_a) - p_draw / 2)
    total = p_a + p_draw + p_b
    return p_a / total, p_draw / total, p_b / total


def main():
    conn = sqlite3.connect(DB_PATH)
    matches = load_matches(conn)
    conn.close()

    elo = {}
    seq = {}  # team -> list[(gf, ga)] по мере прохождения истории

    def get_elo(t):
        return elo.setdefault(t, get_starting_elo(t))

    results = []  # (predicted_probs, actual_outcome_index) для метрик
    correct, total_evaluated = 0, 0

    for _, home, away, hg, ag in matches:
        seq_home = seq.get(home, [])
        seq_away = seq.get(away, [])

        # Оцениваем прогноз, только если у обеих команд есть история матчей
        if len(seq_home) >= MIN_HISTORY and len(seq_away) >= MIN_HISTORY:
            p_home, p_draw, p_away = predict_pre_match(elo, home, away, seq_home, seq_away)

            if hg > ag:
                actual = 0  # победа хозяев
            elif hg == ag:
                actual = 1  # ничья
            else:
                actual = 2  # победа гостей

            predicted_probs = [p_home, p_draw, p_away]
            results.append((predicted_probs, actual))

            predicted_outcome = predicted_probs.index(max(predicted_probs))
            if predicted_outcome == actual:
                correct += 1
            total_evaluated += 1

        # Обновляем Elo и историю ПОСЛЕ оценки — на будущее
        r_home, r_away = get_elo(home), get_elo(away)
        exp_home = 1 / (1 + 10 ** ((r_away - r_home) / 400))
        score_home = 1.0 if hg > ag else (0.0 if hg < ag else 0.5)
        mult = mov_multiplier(hg - ag, r_home - r_away)
        k_eff = K_FACTOR * mult
        elo[home] = r_home + k_eff * (score_home - exp_home)
        elo[away] = r_away + k_eff * ((1 - score_home) - (1 - exp_home))

        seq.setdefault(home, []).append((hg, ag))
        seq.setdefault(away, []).append((ag, hg))

    # ---- Метрики ----
    brier_sum = 0.0
    logloss_sum = 0.0
    eps = 1e-15  # чтобы не брать log(0)

    for probs, actual in results:
        for i, p in enumerate(probs):
            target = 1.0 if i == actual else 0.0
            brier_sum += (p - target) ** 2
        p_actual = max(probs[actual], eps)
        logloss_sum += -math.log(p_actual)

    n = len(results)
    brier_score = brier_sum / n / 3  # делим на число исходов, чтобы было в диапазоне ~0-1
    log_loss = logloss_sum / n
    accuracy = correct / total_evaluated

    print(f"Оценено матчей: {n} (из {len(matches)} сыгранных — остальные не имели истории для прогноза)")
    print(f"Accuracy (угадан исход):  {accuracy*100:.1f}%")
    print(f"Brier score:              {brier_score:.4f}  (0 = идеально, ~0.22 = равномерное гадание на 3 исхода)")
    print(f"Log loss:                 {log_loss:.4f}  (ниже = лучше, ~1.10 = равномерное гадание)")


if __name__ == "__main__":
    main()
