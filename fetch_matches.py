"""
Football Prediction Lab -- сбор данных с football-data.org в SQLite.

Теперь поддерживает несколько турниров/лиг сразу (COMPETITIONS) --
изначально только ЧМ, теперь добавлена Ла Лига. Каждый матч хранит
season/matchday (нужно для регрессии между сезонами и walk-forward
по турам в клубном футболе; для ЧМ эти поля просто NULL).

Использование:
    python fetch_matches.py

Требуется: pip install requests
"""

import os
import sqlite3
import requests
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

# ---- Настройки ----
API_TOKEN = os.environ["FOOTBALL_DATA_TOKEN"]
COMPETITIONS = ["WC", "PD"]  # WC = World Cup, PD = La Liga (Primera Division)
DB_PATH = "football.db"

BASE_URL = "https://api.football-data.org/v4"
HEADERS = {"X-Auth-Token": API_TOKEN}


def create_tables(conn):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS teams (
        id INTEGER PRIMARY KEY,
        name TEXT,
        short_name TEXT,
        tla TEXT,
        crest_url TEXT
    );

    CREATE TABLE IF NOT EXISTS matches (
        id INTEGER PRIMARY KEY,
        competition_code TEXT,
        stage TEXT,
        utc_date TEXT,
        status TEXT,
        home_team_id INTEGER,
        away_team_id INTEGER,
        duration TEXT,
        regular_home INTEGER,
        regular_away INTEGER,
        full_home INTEGER,
        full_away INTEGER,
        penalties_home INTEGER,
        penalties_away INTEGER,
        winner TEXT,
        season TEXT,
        matchday INTEGER,
        FOREIGN KEY (home_team_id) REFERENCES teams(id),
        FOREIGN KEY (away_team_id) REFERENCES teams(id)
    );
    """)
    conn.commit()


def upsert_team(conn, team):
    if team.get("id") is None:
        return
    conn.execute("""
        INSERT INTO teams (id, name, short_name, tla, crest_url)
        VALUES (:id, :name, :shortName, :tla, :crest)
        ON CONFLICT(id) DO UPDATE SET
            name=excluded.name,
            short_name=excluded.short_name,
            tla=excluded.tla,
            crest_url=excluded.crest_url
    """, {
        "id": team.get("id"),
        "name": team.get("name"),
        "shortName": team.get("shortName"),
        "tla": team.get("tla"),
        "crest": team.get("crest"),
    })


def upsert_match(conn, m):
    score = m.get("score", {})
    full = score.get("fullTime", {})
    regular = score.get("regularTime", full)
    penalties = score.get("penalties", {})

    season_obj = m.get("season", {}) or {}
    season_start = season_obj.get("startDate")
    season_str = season_start[:4] if season_start else None

    conn.execute("""
        INSERT INTO matches (
            id, competition_code, stage, utc_date, status,
            home_team_id, away_team_id, duration,
            regular_home, regular_away, full_home, full_away,
            penalties_home, penalties_away, winner, season, matchday
        ) VALUES (
            :id, :competition_code, :stage, :utc_date, :status,
            :home_team_id, :away_team_id, :duration,
            :regular_home, :regular_away, :full_home, :full_away,
            :penalties_home, :penalties_away, :winner, :season, :matchday
        )
        ON CONFLICT(id) DO UPDATE SET
            stage=excluded.stage,
            utc_date=excluded.utc_date,
            status=excluded.status,
            home_team_id=excluded.home_team_id,
            away_team_id=excluded.away_team_id,
            full_home=excluded.full_home,
            full_away=excluded.full_away,
            regular_home=excluded.regular_home,
            regular_away=excluded.regular_away,
            penalties_home=excluded.penalties_home,
            penalties_away=excluded.penalties_away,
            winner=excluded.winner,
            season=excluded.season,
            matchday=excluded.matchday
    """, {
        "id": m["id"],
        "competition_code": m.get("competition", {}).get("code"),
        "stage": m.get("stage"),
        "utc_date": m.get("utcDate"),
        "status": m.get("status"),
        "home_team_id": m.get("homeTeam", {}).get("id"),
        "away_team_id": m.get("awayTeam", {}).get("id"),
        "duration": score.get("duration"),
        "regular_home": regular.get("home"),
        "regular_away": regular.get("away"),
        "full_home": full.get("home"),
        "full_away": full.get("away"),
        "penalties_home": penalties.get("home"),
        "penalties_away": penalties.get("away"),
        "winner": score.get("winner"),
        "season": season_str,
        "matchday": m.get("matchday"),
    })


def fetch_matches(competition_code, status=None):
    url = f"{BASE_URL}/competitions/{competition_code}/matches"
    params = {"status": status} if status else {}
    resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json().get("matches", [])


def main():
    conn = sqlite3.connect(DB_PATH)
    create_tables(conn)

    for competition in COMPETITIONS:
        print(f"[{datetime.now():%H:%M:%S}] Запрашиваю матчи ({competition})...")
        try:
            matches = fetch_matches(competition)
        except Exception as e:
            print(f"  Ошибка при запросе {competition}: {type(e).__name__}: {e}")
            continue
        print(f"  Получено матчей: {len(matches)}")

        for m in matches:
            upsert_team(conn, m.get("homeTeam", {}))
            upsert_team(conn, m.get("awayTeam", {}))
            upsert_match(conn, m)

        conn.commit()

    # Небольшая проверка -- последние завершённые матчи по каждому турниру
    for competition in COMPETITIONS:
        cur = conn.execute("""
            SELECT m.utc_date, t1.name, m.full_home, m.full_away, t2.name, m.stage
            FROM matches m
            JOIN teams t1 ON m.home_team_id = t1.id
            JOIN teams t2 ON m.away_team_id = t2.id
            WHERE m.status = 'FINISHED' AND m.competition_code = ?
            ORDER BY m.utc_date DESC
            LIMIT 5
        """, (competition,))
        rows = cur.fetchall()
        if rows:
            print(f"\nПоследние завершённые матчи ({competition}):")
            for row in rows:
                date, home, hs, away_s, away, stage = row
                print(f"  {date[:10]} [{stage}] {home} {hs}:{away_s} {away}")

    conn.close()
    print(f"\nГотово. База сохранена в {DB_PATH}")


if __name__ == "__main__":
    main()
