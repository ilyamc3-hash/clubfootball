import sqlite3

conn = sqlite3.connect('football.db')
rows = conn.execute("""
    SELECT p.match_id, p.home_team, p.away_team, p.p_home, p.p_draw, p.p_away,
           p.actual_result, p.was_correct, m.utc_date
    FROM predictions p
    LEFT JOIN matches m ON p.match_id = m.id
    WHERE p.actual_result IS NOT NULL
    ORDER BY m.utc_date ASC
""").fetchall()

print(f"Сверенных прогнозов: {len(rows)}\n")

for match_id, home, away, p_home, p_draw, p_away, actual, correct, date in rows:
    if actual == "HOME_WIN":
        actual_str = f"победа {home}"
    elif actual == "AWAY_WIN":
        actual_str = f"победа {away}"
    else:
        actual_str = "ничья"
    mark = "✅" if correct else "❌"
    print(f"{mark} [{date}] {home} — {away}")
    print(f"   Прогноз: П1 {p_home*100:.1f}%  Х {p_draw*100:.1f}%  П2 {p_away*100:.1f}%")
    print(f"   Реальный исход: {actual_str}\n")
