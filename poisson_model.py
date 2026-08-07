"""
Football Prediction Lab — Пуассон-модель голов (v0.1).

Идея (блок 4 из изначального плана проекта):
Вместо того чтобы сравнивать общую "силу" команд (как в Elo), считаем
для каждой команды ожидаемое число голов в конкретном матче — и уже
из распределения Пуассона получаем вероятность ЛЮБОГО счёта, а из него —
П1/Х/П2, тотал больше/меньше 2.5 и "обе забьют" МАТЕМАТИЧЕСКИ,
а не через эвристику "ничья ~28%", как в Elo-модели.

Attack/Defense считаются с поправкой на соперника: 2 гола Франции
"весят" не так, как 2 гола Панамы — если противник в среднем хорошо
обороняется, забить ему сложнее, и наоборот.

ВАЖНО: это ПЕРВАЯ версия Пуассон-блока, независимая от Elo/Form.
Дальнейшее развитие — комбинировать оба подхода (Elo для общей силы,
Пуассон для конкретного счёта), но это отдельный шаг.

Использование:
    py poisson_model.py
"""

import sqlite3
import math
from datetime import datetime

DB_PATH = "football.db"

# Если оставить пустым списком [] — скрипт сам найдёт ближайшие
# запланированные матчи. Можно задать вручную конкретные пары вместо этого.
MATCHES_TO_PREDICT = []

MAX_GOALS = 8  # считаем счета от 0:0 до MAX_GOALS:MAX_GOALS включительно


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
    window_hours = 20

    same_window = []
    for date, h, a in rows:
        dt = datetime.strptime(date, fmt)
        if (dt - earliest).total_seconds() <= window_hours * 3600:
            same_window.append((h, a))

    return same_window, rows[0][0][:16]


def load_matches(conn):
    cur = conn.execute("""
        SELECT t1.name, t2.name, m.regular_home, m.regular_away
        FROM matches m
        JOIN teams t1 ON m.home_team_id = t1.id
        JOIN teams t2 ON m.away_team_id = t2.id
        WHERE m.status = 'FINISHED'
          AND m.regular_home IS NOT NULL
    """)
    return cur.fetchall()


def build_attack_defense(matches):
    """Считаем среднее число забитых/пропущенных голов на команду,
    затем нормализуем относительно среднего гола по турниру."""
    goals_for = {}
    goals_against = {}
    games_played = {}

    total_goals, total_games = 0, 0

    for home, away, hg, ag in matches:
        for team, gf, ga in [(home, hg, ag), (away, ag, hg)]:
            goals_for[team] = goals_for.get(team, 0) + gf
            goals_against[team] = goals_against.get(team, 0) + ga
            games_played[team] = games_played.get(team, 0) + 1

        total_goals += hg + ag
        total_games += 2  # два "командо-матча" на игру

    league_avg_goals = total_goals / total_games

    attack = {}
    defense = {}
    for team in games_played:
        n = games_played[team]
        avg_scored = goals_for[team] / n
        avg_conceded = goals_against[team] / n
        attack[team] = avg_scored / league_avg_goals
        defense[team] = avg_conceded / league_avg_goals

    return attack, defense, league_avg_goals


def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / math.factorial(k)


def score_matrix(lambda_a, lambda_b):
    """Возвращает словарь {(голы_a, голы_b): вероятность} для всех счетов до MAX_GOALS."""
    matrix = {}
    for i in range(MAX_GOALS + 1):
        for j in range(MAX_GOALS + 1):
            matrix[(i, j)] = poisson_pmf(i, lambda_a) * poisson_pmf(j, lambda_b)
    return matrix


def summarize(matrix):
    p_home, p_draw, p_away = 0.0, 0.0, 0.0
    p_over25, p_under25, p_btts_yes = 0.0, 0.0, 0.0

    for (i, j), p in matrix.items():
        if i > j:
            p_home += p
        elif i == j:
            p_draw += p
        else:
            p_away += p

        if i + j >= 3:
            p_over25 += p
        else:
            p_under25 += p

        if i >= 1 and j >= 1:
            p_btts_yes += p

    return {
        "p_home": p_home, "p_draw": p_draw, "p_away": p_away,
        "p_over25": p_over25, "p_under25": p_under25,
        "p_btts_yes": p_btts_yes, "p_btts_no": 1 - p_btts_yes,
    }


def top_scores(matrix, n=5):
    return sorted(matrix.items(), key=lambda kv: kv[1], reverse=True)[:n]


def main():
    conn = sqlite3.connect(DB_PATH)
    matches = load_matches(conn)

    matches_to_predict = MATCHES_TO_PREDICT
    if not matches_to_predict:
        matches_to_predict, nearest_date = find_upcoming_matches(conn)
        if not matches_to_predict:
            print("Не найдено предстоящих матчей в базе. Обнови данные через fetch_matches.py.")
            conn.close()
            return
        print(f"Автоматически найдены ближайшие матчи (дата: {nearest_date}):\n")

    conn.close()

    attack, defense, league_avg = build_attack_defense(matches)

    print(f"Средний гол за команду по турниру: {league_avg:.2f}\n")

    for team_a, team_b in matches_to_predict:
        att_a, def_a = attack.get(team_a, 1.0), defense.get(team_a, 1.0)
        att_b, def_b = attack.get(team_b, 1.0), defense.get(team_b, 1.0)

        lambda_a = att_a * def_b * league_avg
        lambda_b = att_b * def_a * league_avg

        matrix = score_matrix(lambda_a, lambda_b)
        s = summarize(matrix)

        print(f"{team_a} — {team_b}")
        print(f"  Ожидаемые голы: {team_a} {lambda_a:.2f} — {lambda_b:.2f} {team_b}")
        print(f"  П1: {s['p_home']*100:.1f}%  Х: {s['p_draw']*100:.1f}%  П2: {s['p_away']*100:.1f}%")
        print(f"  Тотал больше 2.5: {s['p_over25']*100:.1f}%   Тотал меньше 2.5: {s['p_under25']*100:.1f}%")
        print(f"  Обе забьют — да: {s['p_btts_yes']*100:.1f}%   нет: {s['p_btts_no']*100:.1f}%")

        print("  Самые вероятные счета:")
        for (i, j), p in top_scores(matrix):
            print(f"    {i}:{j}  —  {p*100:.1f}%")
        print()


if __name__ == "__main__":
    main()
