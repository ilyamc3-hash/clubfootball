import sqlite3

conn = sqlite3.connect('football.db')

# Смотрим, сколько дублей есть по каждому match_id
dupes = conn.execute("""
    SELECT match_id, COUNT(*) as cnt
    FROM predictions
    GROUP BY match_id
    HAVING cnt > 1
""").fetchall()

if not dupes:
    print("Дублей не найдено — база чистая.")
else:
    print(f"Найдено match_id с дублями: {len(dupes)}")
    for match_id, cnt in dupes:
        print(f"  match_id={match_id}: {cnt} записей")

    # Оставляем только самую раннюю запись (минимальный id) по каждому match_id
    conn.execute("""
        DELETE FROM predictions
        WHERE id NOT IN (
            SELECT MIN(id)
            FROM predictions
            GROUP BY match_id
        )
    """)
    conn.commit()

    remaining = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    print(f"\nГотово. Дубли удалены. Осталось записей в predictions: {remaining}")

conn.close()
