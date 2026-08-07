"""
Football Prediction Lab — версионирование моделей.

Формализует то, что мы уже делали неформально в разговоре: прогоняет
несколько именованных конфигураций модели через один и тот же
walk-forward backtest и сохраняет результаты в таблицу model_versions —
чтобы всегда можно было посмотреть историю и не потерять, что и когда
улучшало точность.

Использование:
    py model_versions.py
"""

import sqlite3
import statistics
import math
import json
from datetime import datetime

from fifa_ratings import get_starting_elo

DB_PATH = "football.db"
MIN_HISTORY = 1

# ---- Определения версий модели ----
# use_fifa_start: если False — все команды стартуют с 1500 (как в самом начале)
# use_mov: если False — Elo без поправки на margin of victory
# form_momentum_weight: вес Form/Momentum относительно Elo (0 = отключены)
VERSIONS = [
    {
        "name": "v0.1_baseline",
        "description": "Голый Elo, все команды с 1500, без margin of victory, без Form/Momentum",
        "use_fifa_start": False,
        "use_mov": False,
        "form_momentum_weight": 0,
    },
    {
        "name": "v0.1_form_momentum",
        "description": "+ Form и Momentum (вес 15), без margin of victory, без старта ФИФА",
        "use_fifa_start": False,
        "use_mov": False,
        "form_momentum_weight": 15,
    },
    {
        "name": "v0.1_mov",
        "description": "+ Margin of victory, без старта ФИФА",
        "use_fifa_start": False,
        "use_mov": True,
        "form_momentum_weight": 15,
    },
    {
        "name": "v0.2_fifa_start",
        "description": "+ Стартовый рейтинг ФИФА вместо единого 1500 для всех",
        "use_fifa_start": True,
        "use_mov": True,
        "form_momentum_weight": 15,
    },
    {
        "name": "v0.2_fifa_no_form",
        "description": "То же самое, но без Form/Momentum (ablation-проверка)",
        "use_fifa_start": True,
        "use_mov": True,
        "form_momentum_weight": 0,
    },
]


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
        WHERE m.status = 'FINISHED' AND m.regular_home IS NOT NULL
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


def evaluate_version(matches, config):
    base_elo = 1500
    K_FACTOR = 32
    elo, seq = {}, {}

    def get_elo(t):
        if config["use_fifa_start"]:
            return elo.setdefault(t, get_starting_elo(t))
        return elo.setdefault(t, base_elo)

    results = []
    correct, total_evaluated = 0, 0

    for _, home, away, hg, ag in matches:
        seq_home = seq.get(home, [])
        seq_away = seq.get(away, [])

        if len(seq_home) >= MIN_HISTORY and len(seq_away) >= MIN_HISTORY:
            elo_a, elo_b = get_elo(home), get_elo(away)
            effective_diff = elo_a - elo_b

            if config["form_momentum_weight"] > 0:
                form_a, form_b = form_for(seq_home), form_for(seq_away)
                mom_a, mom_b = momentum_for(seq_home), momentum_for(seq_away)
                w = config["form_momentum_weight"]
                effective_diff += (form_a - form_b) * w + (mom_a - mom_b) * w

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

        if config["use_mov"]:
            mult = mov_multiplier(hg - ag, r_home - r_away)
        else:
            mult = 1.0
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


def create_table(conn):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS model_versions (
        name TEXT PRIMARY KEY,
        description TEXT,
        params_json TEXT,
        n_matches INTEGER,
        accuracy REAL,
        brier_score REAL,
        log_loss REAL,
        evaluated_at TEXT
    );
    """)
    conn.commit()


def save_version(conn, config, metrics):
    conn.execute("""
        INSERT INTO model_versions (name, description, params_json, n_matches, accuracy, brier_score, log_loss, evaluated_at)
        VALUES (:name, :description, :params_json, :n, :accuracy, :brier, :logloss, :evaluated_at)
        ON CONFLICT(name) DO UPDATE SET
            description=excluded.description,
            params_json=excluded.params_json,
            n_matches=excluded.n_matches,
            accuracy=excluded.accuracy,
            brier_score=excluded.brier_score,
            log_loss=excluded.log_loss,
            evaluated_at=excluded.evaluated_at
    """, {
        "name": config["name"],
        "description": config["description"],
        "params_json": json.dumps({k: v for k, v in config.items() if k not in ("name", "description")}),
        "n": metrics["n"],
        "accuracy": metrics["accuracy"],
        "brier": metrics["brier"],
        "logloss": metrics["logloss"],
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
    })


def main():
    conn = sqlite3.connect(DB_PATH)
    create_table(conn)
    matches = load_matches(conn)

    print(f"Прогоняю {len(VERSIONS)} версий модели на {len(matches)} сыгранных матчах...\n")

    for config in VERSIONS:
        metrics = evaluate_version(matches, config)
        save_version(conn, config, metrics)
    conn.commit()

    cur = conn.execute("""
        SELECT name, description, n_matches, accuracy, brier_score, log_loss
        FROM model_versions
        ORDER BY brier_score ASC
    """)
    rows = cur.fetchall()
    conn.close()

    print(f"{'Версия':<22}{'n':>5}{'Accuracy':>11}{'Brier':>10}{'LogLoss':>10}")
    print("-" * 60)
    for name, desc, n, acc, brier, logloss in rows:
        print(f"{name:<22}{n:>5}{acc*100:>10.1f}%{brier:>10.4f}{logloss:>10.4f}")

    print("\nЛучшая по Brier score (отсортировано выше — первая строка):")
    best = rows[0]
    print(f"  {best[0]} — {best[1]}")


if __name__ == "__main__":
    main()
