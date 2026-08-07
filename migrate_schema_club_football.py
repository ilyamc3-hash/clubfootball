"""
Football Prediction Lab -> Club Football: миграция схемы football.db.

Добавляет nullable-колонки season и matchday в таблицу matches -
нужны для клубного футбола (регрессия между сезонами, walk-forward
по турам), для ЧМ они просто останутся NULL и ничего не сломают.

Безопасно запускать повторно - проверяет, что колонки ещё не добавлены,
прежде чем добавлять (иначе SQLite выдаст ошибку "duplicate column").

Использование:
    py migrate_schema_club_football.py
"""
import sqlite3

DB_PATH = "football.db"


def column_exists(conn, table, column):
    cur = conn.execute(f"PRAGMA table_info({table})")
    return any(row[1] == column for row in cur.fetchall())


def main():
    conn = sqlite3.connect(DB_PATH)

    added = []
    skipped = []

    if not column_exists(conn, "matches", "season"):
        conn.execute("ALTER TABLE matches ADD COLUMN season TEXT")
        added.append("season")
    else:
        skipped.append("season")

    if not column_exists(conn, "matches", "matchday"):
        conn.execute("ALTER TABLE matches ADD COLUMN matchday INTEGER")
        added.append("matchday")
    else:
        skipped.append("matchday")

    conn.commit()

    print("Миграция завершена.")
    if added:
        print(f"Добавлены колонки: {', '.join(added)}")
    if skipped:
        print(f"Уже существовали (пропущены): {', '.join(skipped)}")

    # Проверка - выводим текущую структуру таблицы matches
    print("\nТекущая структура таблицы matches:")
    cur = conn.execute("PRAGMA table_info(matches)")
    for row in cur.fetchall():
        print(f"  {row[1]:<20}{row[2]}")

    # Проверка, что существующие данные ЧМ не пострадали
    total = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    finished = conn.execute("SELECT COUNT(*) FROM matches WHERE status='FINISHED'").fetchone()[0]
    print(f"\nПроверка целостности: всего матчей {total}, из них FINISHED {finished}")
    print("(эти числа должны совпадать с тем, что было до миграции)")

    conn.close()


if __name__ == "__main__":
    main()
