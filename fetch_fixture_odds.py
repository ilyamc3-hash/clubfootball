"""
Football Prediction Lab — захват предматчевых кэфов Ла Лиги в кэш.

Источник: https://www.football-data.co.uk/fixtures.csv — ТОТ ЖЕ провайдер,
что и 12-сезонная история (laliga_matches_combined.csv), поэтому колонки
AvgH/AvgD/AvgA и Avg>2.5/Avg<2.5 методологически совпадают с теми, по
которым считался эталонный OOS-разрыв с рынком (+0.0042 в
laliga_oos_validation.py). Файл обновляется на стороне провайдера
~2 раза в неделю и покрывает только ближайшие туры.

Запуск: по cron раз в день (по образцу refit_dixon_coles.py):
    0 4 * * *  cd /opt/football-bot && ... python fetch_fixture_odds.py

Локальный тест без сети:
    py fetch_fixture_odds.py --file fixtures_local.csv

Пишет в market_odds_cache (football.db). Имена клубов мапятся на имена
football-data.org; НЕсматченные имена печатаются ЯВНО (не молчим) — это
критично для новичков лиги, которых нет в исторических данных.
"""

import argparse
import csv
import io
import sqlite3
import sys
from datetime import datetime, timezone

from accuracy_store import (
    ensure_accuracy_tables, map_fixture_team,
    odds_to_probs_3way, odds_to_probs_2way,
)

FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
DB_PATH = "football.db"
TARGET_DIV = "SP1"  # Ла Лига у football-data.co.uk


def download_fixtures():
    import requests  # уже стоит на сервере (используется fetch_matches.py)
    resp = requests.get(FIXTURES_URL, timeout=30)
    resp.raise_for_status()
    # у провайдера файл с BOM
    return resp.content.decode("utf-8-sig", errors="replace")


def read_local(path):
    with open(path, encoding="utf-8-sig") as f:
        return f.read()


def parse_date(d):
    """fixtures.csv: DD/MM/YYYY -> ISO YYYY-MM-DD"""
    try:
        return datetime.strptime(d.strip(), "%d/%m/%Y").strftime("%Y-%m-%d")
    except ValueError:
        try:
            return datetime.strptime(d.strip(), "%d/%m/%y").strftime("%Y-%m-%d")
        except ValueError:
            return None


def ffloat(v):
    try:
        return float(v) if v not in (None, "") else None
    except ValueError:
        return None


def db_liga_team_names(conn):
    rows = conn.execute("""
        SELECT DISTINCT t.name FROM teams t
        JOIN matches m ON t.id IN (m.home_team_id, m.away_team_id)
        WHERE m.competition_code = 'PD' AND t.name IS NOT NULL
    """).fetchall()
    return {r[0] for r in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="локальный fixtures.csv вместо скачивания (для теста)")
    ap.add_argument("--db", default=DB_PATH)
    args = ap.parse_args()

    print(f"[{datetime.now():%H:%M:%S}] Источник: "
          + (args.file if args.file else FIXTURES_URL))
    text = read_local(args.file) if args.file else download_fixtures()

    conn = sqlite3.connect(args.db)
    ensure_accuracy_tables(conn)
    team_names = db_liga_team_names(conn)
    if not team_names:
        print("В football.db нет команд PD — сначала запусти fetch_matches.py")
        sys.exit(1)

    captured_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    reader = csv.DictReader(io.StringIO(text))
    n_rows, n_mapped, unmatched = 0, 0, set()

    for r in reader:
        if (r.get("Div") or "").strip() != TARGET_DIV:
            continue
        match_date = parse_date(r.get("Date", ""))
        home_raw = (r.get("HomeTeam") or "").strip()
        away_raw = (r.get("AwayTeam") or "").strip()
        if not (match_date and home_raw and away_raw):
            continue
        n_rows += 1

        home = map_fixture_team(home_raw, team_names)
        away = map_fixture_team(away_raw, team_names)
        for raw, mapped in ((home_raw, home), (away_raw, away)):
            if mapped is None:
                unmatched.add(raw)

        avg_h, avg_d, avg_a = ffloat(r.get("AvgH")), ffloat(r.get("AvgD")), ffloat(r.get("AvgA"))
        # ВАЖНО: в заголовке провайдера колонки называются буквально "Avg>2.5"/"Avg<2.5"
        avg_over, avg_under = ffloat(r.get("Avg>2.5")), ffloat(r.get("Avg<2.5"))

        probs3 = odds_to_probs_3way(avg_h, avg_d, avg_a)
        p_over = odds_to_probs_2way(avg_over, avg_under)

        conn.execute("""
            INSERT INTO market_odds_cache
                (div, match_date, home_raw, away_raw, home_team, away_team,
                 avg_h, avg_d, avg_a, p_home, p_draw, p_away,
                 avg_over25, avg_under25, p_over25, captured_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(div, match_date, home_raw, away_raw) DO UPDATE SET
                home_team=excluded.home_team, away_team=excluded.away_team,
                avg_h=excluded.avg_h, avg_d=excluded.avg_d, avg_a=excluded.avg_a,
                p_home=excluded.p_home, p_draw=excluded.p_draw, p_away=excluded.p_away,
                avg_over25=excluded.avg_over25, avg_under25=excluded.avg_under25,
                p_over25=excluded.p_over25, captured_at=excluded.captured_at
        """, (TARGET_DIV, match_date, home_raw, away_raw, home, away,
              avg_h, avg_d, avg_a,
              probs3[0] if probs3 else None,
              probs3[1] if probs3 else None,
              probs3[2] if probs3 else None,
              avg_over, avg_under, p_over, captured_at))
        if home and away:
            n_mapped += 1

    conn.commit()

    print(f"Строк {TARGET_DIV} в fixtures.csv: {n_rows}, полностью смаплено: {n_mapped}")
    if unmatched:
        print("⚠ НЕ СМАПЛЕНЫ (кэфы по этим матчам не прикрепятся к прогнозам!):")
        for name in sorted(unmatched):
            print(f"    {name!r} — добавь алиас в FIXTURES_ALIASES (accuracy_store.py)")
    cached = conn.execute(
        "SELECT COUNT(*) FROM market_odds_cache WHERE div = ?", (TARGET_DIV,)
    ).fetchone()[0]
    print(f"Всего в кэше {TARGET_DIV}: {cached} строк")
    conn.close()


if __name__ == "__main__":
    main()
