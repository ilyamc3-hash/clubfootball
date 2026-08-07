"""
Проверка: в каком статусе сейчас матчи Ла Лиги в базе -
подтверждаем гипотезу про межсезонье.
"""
import sqlite3

conn = sqlite3.connect("football.db")
rows = conn.execute("""
    SELECT status, COUNT(*), MIN(utc_date), MAX(utc_date)
    FROM matches
    WHERE competition_code = 'PD'
    GROUP BY status
""").fetchall()

print("Статусы матчей Ла Лиги в базе:")
for status, count, min_date, max_date in rows:
    print(f"  {status}: {count} матчей (с {min_date[:10]} по {max_date[:10]})")
