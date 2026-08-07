import sqlite3

conn = sqlite3.connect('football.db')
rows = conn.execute("""
    SELECT id, home_team_id, away_team_id, status, utc_date
    FROM matches
    WHERE id IN (537388, 537389, 537390)
""").fetchall()

for r in rows:
    print(r)
