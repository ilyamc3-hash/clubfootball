"""
Football Prediction Lab -> Club Football: рефит Dixon-Coles для /totals.

ОТДЕЛЬНЫЙ скрипт, НЕ часть бота - запускается по расписанию (cron, раз в
REFIT_INTERVAL_DAYS=21 дней, то же значение, что в честном бэктесте
laliga_dixon_coles.py). Дорогой MLE-фит (numpy/scipy) не должен жить в
процессе бота - бот только читает готовый dc_params.json через
dixon_coles_model.DixonColesModel.

Источники данных, объединяются в один список и сортируются хронологически
ПЕРЕД фитом (Dixon-Coles фитится через MLE сразу на всём датасете, не
инкрементально как Elo - в отличие от LaLigaModel, здесь нет пошагового
обновления по сезонам):
  - laliga_matches_combined.csv - 12 законченных сезонов (уже нормализован
    per-season при сборке, читается обычным DictReader)
  - football.db, competition_code='PD', status='FINISHED' - сыгранные
    матчи ТЕКУЩЕГО сезона (тот же SQL-паттерн и тот же fallback для команд
    без маппинга, что LaLigaModel._apply_db_matches в laliga_model.py)

xi=0.0015 - лучший гиперпараметр по train из честного бэктеста, здесь уже
не перебирается заново.

ПРИМЕЧАНИЕ по rho: в литературе (Dixon & Coles, 1997) rho обычно выходит
небольшим ОТРИЦАТЕЛЬНЫМ (~-0.05) - именно поэтому и нужна tau-поправка:
без неё Пуассон-произведение недооценивает низкоскоринговые ничьи вроде
0:0/1:1 и переоценивает 1:0/0:1. На первом прогоне (2026-08-15, полный
датасет 12 сезонов + 0 матчей текущего сезона) rho сошёлся к +0.0041 -
близко к нулю, но с другим знаком, чем "типичный" пример из литературы.
Это НЕ баг (bounds [-0.3, 0.3], MLE сошёлся нормально) - слабый rho просто
означает, что tau-поправка на этом окне данных почти не меняет grid, а её
знак может плавать от рефита к рефиту в зависимости от того, что попало в
recency-взвешенное окно. Стоит иметь в виду при интерпретации BTTS и
топ-счетов: если rho когда-нибудь выйдет заметно положительным (не около
нуля), это стоит перепроверить отдельно - такое уже не похоже на "шум
близко к нулю".

Использование:
    py refit_dixon_coles.py
"""

import csv
import json
import math
import os
import sqlite3
import sys
from datetime import datetime

from laliga_dixon_coles import fit_dixon_coles
from laliga_model import load_team_mapping

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_FILE = os.path.join(BASE_DIR, "laliga_matches_combined.csv")
MAPPING_FILE = os.path.join(BASE_DIR, "team_name_mapping.csv")
DB_PATH = os.path.join(BASE_DIR, "football.db")
OUTPUT_FILE = os.path.join(BASE_DIR, "dc_params.json")

XI = 0.0015


def load_csv_history(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if not r.get("HomeTeam") or not r.get("FTHG"):
                continue
            try:
                date = datetime.strptime(r["Date"], "%d/%m/%Y")
            except ValueError:
                try:
                    date = datetime.strptime(r["Date"], "%d/%m/%y")
                except ValueError:
                    continue
            try:
                hg, ag = int(r["FTHG"]), int(r["FTAG"])
            except (ValueError, TypeError):
                continue
            rows.append({
                "date": date, "home": r["HomeTeam"].strip(),
                "away": r["AwayTeam"].strip(), "hg": hg, "ag": ag,
            })
    return rows


def load_db_matches(db_path, org_to_couk):
    """Сыгранные матчи текущего сезона Ла Лиги из football.db, имена
    переведены в couk-пространство (или заведены под org-именем для
    новичков без маппинга) - идентично LaLigaModel._apply_db_matches."""
    try:
        conn = sqlite3.connect(db_path)
        rows = conn.execute("""
            SELECT m.utc_date, t1.name, t2.name, m.regular_home, m.regular_away
            FROM matches m
            JOIN teams t1 ON m.home_team_id = t1.id
            JOIN teams t2 ON m.away_team_id = t2.id
            WHERE m.competition_code = 'PD' AND m.status = 'FINISHED'
              AND m.regular_home IS NOT NULL
            ORDER BY m.utc_date ASC
        """).fetchall()
        conn.close()
    except sqlite3.Error as e:
        print(f"!! Не удалось прочитать football.db: {e} - использую только CSV-историю.")
        return []

    fmt = "%Y-%m-%dT%H:%M:%SZ"
    matches = []
    for utc_date, home_org, away_org, hg, ag in rows:
        date = datetime.strptime(utc_date, fmt)
        home_couk = org_to_couk.get(home_org, home_org)
        away_couk = org_to_couk.get(away_org, away_org)
        matches.append({"date": date, "home": home_couk, "away": away_couk, "hg": hg, "ag": ag})
    return matches


def main():
    print(f"[{datetime.now().isoformat(timespec='seconds')}] Рефит Dixon-Coles (xi={XI})...")

    csv_matches = load_csv_history(HISTORY_FILE)
    print(f"  CSV-история ({HISTORY_FILE}): {len(csv_matches)} матчей")

    org_to_couk = load_team_mapping(MAPPING_FILE)
    db_matches = load_db_matches(DB_PATH, org_to_couk)
    print(f"  football.db (PD, FINISHED, текущий сезон): {len(db_matches)} матчей")

    all_matches = sorted(csv_matches + db_matches, key=lambda m: m["date"])
    if not all_matches:
        print("!! Нет матчей для фита. Прерываю, dc_params.json не трогаю.")
        sys.exit(1)

    date_min, date_max = all_matches[0]["date"], all_matches[-1]["date"]
    print(f"  Итого: {len(all_matches)} матчей, {date_min.date()} .. {date_max.date()}")

    all_teams = sorted({m["home"] for m in all_matches} | {m["away"] for m in all_matches})
    team_index = {t: i for i, t in enumerate(all_teams)}
    n_teams = len(all_teams)
    print(f"  Команд: {n_teams} (референс: {all_teams[0]})")

    as_of_date = datetime.now()
    params = fit_dixon_coles(all_matches, as_of_date, XI, team_index, n_teams)

    mu, home_adv, rho = params["mu"], params["home_adv"], params["rho"]
    attack, defense = params["attack"], params["defense"]

    values_to_check = [mu, home_adv, rho] + list(attack) + list(defense)
    if any(math.isnan(v) or math.isinf(v) for v in values_to_check):
        print("!! Фит дал NaN/inf в параметрах - НЕ перезаписываю dc_params.json.")
        print(f"   mu={mu} home_adv={home_adv} rho={rho}")
        sys.exit(1)

    league_avg_goals = sum(m["hg"] + m["ag"] for m in all_matches) / (2 * len(all_matches))

    attack_by_team = {team: float(attack[i]) for team, i in team_index.items()}
    defense_by_team = {team: float(defense[i]) for team, i in team_index.items()}

    output = {
        "fit_date": as_of_date.isoformat(timespec="seconds"),
        "xi": XI,
        "n_matches": len(all_matches),
        "n_teams": n_teams,
        "date_range": [date_min.date().isoformat(), date_max.date().isoformat()],
        "mu": float(mu),
        "home_adv": float(home_adv),
        "rho": float(rho),
        "league_avg_goals": league_avg_goals,
        "attack": attack_by_team,
        "defense": defense_by_team,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, sort_keys=True)

    print(f"\nmu={mu:.4f} home_adv={home_adv:.4f} rho={rho:.4f}")
    if rho > 0.02:
        print("  !! rho заметно положительный (не около нуля) - см. примечание "
              "в docstring этого файла, стоит перепроверить.")
    print(f"Средний тотал (по сырым данным): {league_avg_goals:.2f}")

    ranked = sorted(attack_by_team.items(), key=lambda kv: kv[1], reverse=True)
    print("\nТоп-5 атака:")
    for team, val in ranked[:5]:
        print(f"  {team:30s} {val:+.3f}")
    print("Анти-топ-5 атака:")
    for team, val in ranked[-5:]:
        print(f"  {team:30s} {val:+.3f}")

    print(f"\nСохранено в {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
