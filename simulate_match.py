"""
Football Prediction Lab — симуляция матча (Монте-Карло).

Идея: вместо того чтобы просто посчитать вероятность математически
(как в poisson_model.py), "разыгрываем" матч случайным образом N раз
подряд — каждый раз генерируем случайное число голов для каждой команды
(с учётом её ожидаемого количества голов, λ), считаем результат, и в
конце смотрим, как часто выигрывала каждая команда.

При большом N (1000+) результат должен сходиться к тем же цифрам, что
даёт точный математический расчёт в poisson_model.py — это хороший
способ проверить, что там нет ошибки в формулах (два независимых
метода должны давать одинаковый ответ).

Использование:
    py simulate_match.py
"""

import sqlite3
import random
import math
from datetime import datetime

DB_PATH = "football.db"
N_SIMULATIONS = 10000  # сколько раз "сыграть" матч

MATCHES_TO_SIMULATE = []  # пусто — автопоиск ближайших матчей, как в других скриптах


def load_matches(conn):
    cur = conn.execute("""
        SELECT t1.name, t2.name, m.regular_home, m.regular_away
        FROM matches m
        JOIN teams t1 ON m.home_team_id = t1.id
        JOIN teams t2 ON m.away_team_id = t2.id
        WHERE m.status = 'FINISHED' AND m.regular_home IS NOT NULL
    """)
    return cur.fetchall()


def find_upcoming_matches(conn):
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
    window = []
    for date, h, a in rows:
        dt = datetime.strptime(date, fmt)
        if (dt - earliest).total_seconds() <= 20 * 3600:
            window.append((h, a))
    return window, rows[0][0][:16]


def build_attack_defense(matches):
    gf, ga, games = {}, {}, {}
    total_goals, total_games = 0, 0
    for home, away, hg, ag in matches:
        for team, s, c in [(home, hg, ag), (away, ag, hg)]:
            gf[team] = gf.get(team, 0) + s
            ga[team] = ga.get(team, 0) + c
            games[team] = games.get(team, 0) + 1
        total_goals += hg + ag
        total_games += 2
    league_avg = total_goals / total_games if total_games else 1.4
    attack = {t: (gf[t] / games[t]) / league_avg for t in games}
    defense = {t: (ga[t] / games[t]) / league_avg for t in games}
    return attack, defense, league_avg


def sample_poisson(lam):
    """Генерирует случайное число событий (голов) по распределению Пуассона
    с ожидаемым значением lam. Реализация без numpy (алгоритм Кнута)."""
    l = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= random.random()
        if p <= l:
            return k - 1


def simulate(lambda_a, lambda_b, n):
    wins_a, draws, wins_b = 0, 0, 0
    score_counter = {}

    for _ in range(n):
        goals_a = sample_poisson(lambda_a)
        goals_b = sample_poisson(lambda_b)

        if goals_a > goals_b:
            wins_a += 1
        elif goals_a == goals_b:
            draws += 1
        else:
            wins_b += 1

        key = (goals_a, goals_b)
        score_counter[key] = score_counter.get(key, 0) + 1

    return wins_a, draws, wins_b, score_counter


def main():
    conn = sqlite3.connect(DB_PATH)
    finished = load_matches(conn)

    matches_to_predict = MATCHES_TO_SIMULATE
    if not matches_to_predict:
        matches_to_predict, when = find_upcoming_matches(conn)
        if not matches_to_predict:
            print("Не найдено предстоящих матчей. Обнови базу через fetch_matches.py.")
            conn.close()
            return
        print(f"Автоматически найдены ближайшие матчи ({when} UTC):\n")

    conn.close()

    attack, defense, league_avg = build_attack_defense(finished)

    for team_a, team_b in matches_to_predict:
        att_a, def_a = attack.get(team_a, 1.0), defense.get(team_a, 1.0)
        att_b, def_b = attack.get(team_b, 1.0), defense.get(team_b, 1.0)
        lambda_a = att_a * def_b * league_avg
        lambda_b = att_b * def_a * league_avg

        wins_a, draws, wins_b, score_counter = simulate(lambda_a, lambda_b, N_SIMULATIONS)

        print(f"{team_a} — {team_b}  (симулировано {N_SIMULATIONS} раз, λ = {lambda_a:.2f} — {lambda_b:.2f})")
        print(f"  {team_a} победила: {wins_a} раз  ({wins_a/N_SIMULATIONS*100:.1f}%)")
        print(f"  Ничья:            {draws} раз  ({draws/N_SIMULATIONS*100:.1f}%)")
        print(f"  {team_b} победила: {wins_b} раз  ({wins_b/N_SIMULATIONS*100:.1f}%)")

        top5 = sorted(score_counter.items(), key=lambda kv: kv[1], reverse=True)[:5]
        print("  Самые частые счета в симуляции:")
        for (i, j), count in top5:
            print(f"    {i}:{j} — {count} раз ({count/N_SIMULATIONS*100:.1f}%)")
        print()


if __name__ == "__main__":
    main()
