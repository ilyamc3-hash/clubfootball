import sqlite3

conn = sqlite3.connect('football.db')
rows = conn.execute("""
    SELECT m.id, m.status, m.utc_date, t1.name, t2.name
    FROM matches m
    LEFT JOIN teams t1 ON m.home_team_id = t1.id
    LEFT JOIN teams t2 ON m.away_team_id = t2.id
    WHERE m.status != 'FINISHED'
    ORDER BY m.utc_date ASC
    LIMIT 20
""").fetchall()

if not rows:
    print("В базе вообще нет незавершённых матчей.")
else:
    print(f"Найдено незавершённых матчей: {len(rows)}\n")
    for match_id, status, date, home, away in rows:
        print(f"id={match_id}  status={status!r}  {date}  {home!r} — {away!r}")
