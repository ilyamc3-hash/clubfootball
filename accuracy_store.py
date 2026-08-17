"""
Football Prediction Lab — хранилище прогнозов Ла Лиги для /accuracy_liga.

Отдельный модуль (stdlib only), чтобы:
  - bot.py оставался тонким (по образцу dixon_coles_model.py);
  - логику можно было тестировать локально без aiogram/Telegram.

Что здесь:
  - миграция таблиц (predictions += competition_code/market_*/is_fallback,
    новые predictions_totals и market_odds_cache);
  - маппинг имён клубов football-data.co.uk (fixtures.csv) -> имена
    football-data.org (как в football.db), с логированием несматченных;
  - сохранение прогнозов 1X2 (Elo) и тоталов (Dixon-Coles) с прикреплением
    рыночных вероятностей ИЗ КЭША на момент сохранения (никаких поздних
    "дозаписей" рынка — это сломало бы честность сравнения);
  - сверка с реальными результатами (reconcile);
  - агрегация статистики для /accuracy_liga.

Методология сравнения с рынком — 1:1 как в laliga_oos_validation.py:
  - колонки AvgH/AvgD/AvgA (средние предматчевые кэфы, не closing);
  - пропорциональная нормализация маржи: p_i = (1/o_i) / sum(1/o_j);
  - Brier 1X2 = sum((p_k - t_k)^2) / 3 (то же деление на 3);
  - Brier рынка считается ТОЛЬКО по строкам, где рынок был захвачен
    (парное сравнение, как в бэктесте).
Для тоталов: Avg>2.5 / Avg<2.5, двухисходная нормализация. Brier
бинарный: (p - t)^2. ВНИМАНИЕ: перед сравнением живого gap с бэктестовым
+0.0044 сверить конвенцию с laliga_totals.py (там может быть /2).
BTTS-рынка в fixtures.csv нет — по BTTS только Brier модели.
"""

import sqlite3
import unicodedata
from datetime import datetime, timedelta

LIGA_COMPETITION = "PD"

# --- Маппинг имён fixtures.csv (football-data.co.uk) -> football.db ---
# Явные алиасы для имён, которые не разрешаются простым вхождением подстроки.
# Ключи — normalize_name() от имени в fixtures.csv.
FIXTURES_ALIASES = {
    "ath bilbao": "Athletic Club",
    "ath madrid": "Club Atlético de Madrid",
    "espanol": "RCD Espanyol de Barcelona",
    "dep a coruna": "RC Deportivo La Coruña",   # fixtures.csv: "Dep. A Coruna"
    "la coruna": "RC Deportivo La Coruña",      # так Депортиво звался в исторических SP1
    "santander": "Real Racing Club de Santander",  # новичок 2026/27, до истории 12 сезонов
}


def normalize_name(s):
    """нижний регистр, без диакритики и пунктуации: 'Deportivo Alavés' -> 'deportivo alaves'"""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = "".join(c if c.isalnum() or c.isspace() else " " for c in s.lower())
    return " ".join(s.split())


def map_fixture_team(raw_name, db_team_names):
    """Имя из fixtures.csv -> имя из football.db (или None, если не нашли/неоднозначно).

    Порядок: явный алиас -> уникальное вхождение подстроки -> уникальное
    совпадение по значимым токенам (>3 букв). Неоднозначность = None,
    чтобы не прикрепить кэфы не к тому матчу молча.
    """
    norm_raw = normalize_name(raw_name)
    if norm_raw in FIXTURES_ALIASES:
        alias_target = FIXTURES_ALIASES[norm_raw]
        return alias_target if alias_target in db_team_names else None

    norm_db = {name: normalize_name(name) for name in db_team_names}

    substr_hits = [name for name, n in norm_db.items() if norm_raw in n]
    if len(substr_hits) == 1:
        return substr_hits[0]
    if len(substr_hits) > 1:
        return None

    tokens = [t for t in norm_raw.split() if len(t) > 3]
    if tokens:
        token_hits = [name for name, n in norm_db.items()
                      if all(t in n.split() for t in tokens)]
        if len(token_hits) == 1:
            return token_hits[0]
    return None


# --- Таблицы и миграция ---

def _existing_columns(conn, table):
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def ensure_accuracy_tables(conn):
    """Идемпотентная миграция. Старые строки predictions помечаются 'WC'."""
    # predictions существует со времён ЧМ — добавляем недостающие колонки
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id INTEGER,
            predicted_at TEXT,
            home_team TEXT,
            away_team TEXT,
            p_home REAL, p_draw REAL, p_away REAL,
            model_version TEXT,
            actual_result TEXT,
            was_correct INTEGER
        )
    """)
    cols = _existing_columns(conn, "predictions")
    add = {
        "competition_code": "TEXT",
        "is_fallback": "INTEGER",
        "market_p_home": "REAL",
        "market_p_draw": "REAL",
        "market_p_away": "REAL",
        "odds_captured_at": "TEXT",
    }
    for col, typ in add.items():
        if col not in cols:
            conn.execute(f"ALTER TABLE predictions ADD COLUMN {col} {typ}")
    # всё, что было записано до миграции — прогнозы ЧМ
    conn.execute(
        "UPDATE predictions SET competition_code = 'WC' WHERE competition_code IS NULL"
    )

    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions_totals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id INTEGER,
            competition_code TEXT,
            predicted_at TEXT,
            home_team TEXT,
            away_team TEXT,
            lambda_home REAL, lambda_away REAL,
            p_over25 REAL, p_btts REAL,
            model_version TEXT,
            is_fallback INTEGER,
            market_p_over25 REAL,
            odds_captured_at TEXT,
            actual_total INTEGER,
            actual_over25 INTEGER,
            actual_btts INTEGER
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS market_odds_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            div TEXT,
            match_date TEXT,          -- ISO YYYY-MM-DD (дата из fixtures.csv, UK)
            home_raw TEXT, away_raw TEXT,   -- имена как в fixtures.csv
            home_team TEXT, away_team TEXT, -- смапленные на football.db (NULL = не нашли)
            avg_h REAL, avg_d REAL, avg_a REAL,
            p_home REAL, p_draw REAL, p_away REAL,
            avg_over25 REAL, avg_under25 REAL,
            p_over25 REAL,
            captured_at TEXT,
            UNIQUE(div, match_date, home_raw, away_raw)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_market_cache_teams
        ON market_odds_cache(home_team, away_team, match_date)
    """)
    conn.commit()


# --- Нормализация кэфов в вероятности (как в laliga_oos_validation.odds_to_probs) ---

def odds_to_probs_3way(oh, od, oa):
    if not (oh and od and oa):
        return None
    raw = (1 / oh, 1 / od, 1 / oa)
    total = sum(raw)
    return tuple(r / total for r in raw)


def odds_to_probs_2way(o_over, o_under):
    if not (o_over and o_under):
        return None
    raw_over, raw_under = 1 / o_over, 1 / o_under
    total = raw_over + raw_under
    return raw_over / total


# --- Поиск рыночных кэфов для матча ---

def find_market_odds(conn, home_team, away_team, utc_date_str):
    """Ищет захваченные кэфы по смапленным именам, дата с допуском ±1 день
    (fixtures.csv хранит дату в UK-времени, football.db — в UTC)."""
    try:
        match_date = datetime.strptime(utc_date_str[:10], "%Y-%m-%d")
    except (ValueError, TypeError):
        return None
    dates = [(match_date + timedelta(days=d)).strftime("%Y-%m-%d") for d in (-1, 0, 1)]
    row = conn.execute("""
        SELECT p_home, p_draw, p_away, p_over25, captured_at
        FROM market_odds_cache
        WHERE home_team = ? AND away_team = ? AND match_date IN (?, ?, ?)
        ORDER BY captured_at DESC LIMIT 1
    """, (home_team, away_team, *dates)).fetchone()
    if not row:
        return None
    return {
        "p_home": row[0], "p_draw": row[1], "p_away": row[2],
        "p_over25": row[3], "captured_at": row[4],
    }


def _match_utc_date(conn, match_id):
    row = conn.execute("SELECT utc_date FROM matches WHERE id = ?", (match_id,)).fetchone()
    return row[0] if row else None


# --- Сохранение прогнозов (рынок прикрепляется в момент сохранения) ---

def save_prediction_1x2(conn, match_id, home, away, p_home, p_draw, p_away,
                        model_version, is_fallback):
    """Один прогноз на матч (первый сохранённый; повторные вызовы — no-op),
    как в старой save_prediction. Рынок берётся из кэша СЕЙЧАС и больше
    не обновляется."""
    if conn.execute("SELECT 1 FROM predictions WHERE match_id = ?", (match_id,)).fetchone():
        return False
    market = find_market_odds(conn, home, away, _match_utc_date(conn, match_id) or "")
    conn.execute("""
        INSERT INTO predictions
            (match_id, predicted_at, home_team, away_team,
             p_home, p_draw, p_away, model_version,
             competition_code, is_fallback,
             market_p_home, market_p_draw, market_p_away, odds_captured_at)
        VALUES (?, datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (match_id, home, away, p_home, p_draw, p_away, model_version,
          LIGA_COMPETITION, 1 if is_fallback else 0,
          market["p_home"] if market else None,
          market["p_draw"] if market else None,
          market["p_away"] if market else None,
          market["captured_at"] if market else None))
    conn.commit()
    return True


def save_prediction_totals(conn, match_id, home, away, lambda_home, lambda_away,
                           p_over25, p_btts, model_version, is_fallback):
    if conn.execute("SELECT 1 FROM predictions_totals WHERE match_id = ?",
                    (match_id,)).fetchone():
        return False
    market = find_market_odds(conn, home, away, _match_utc_date(conn, match_id) or "")
    conn.execute("""
        INSERT INTO predictions_totals
            (match_id, competition_code, predicted_at, home_team, away_team,
             lambda_home, lambda_away, p_over25, p_btts, model_version,
             is_fallback, market_p_over25, odds_captured_at)
        VALUES (?, ?, datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (match_id, LIGA_COMPETITION, home, away,
          lambda_home, lambda_away, p_over25, p_btts, model_version,
          1 if is_fallback else 0,
          market["p_over25"] if market else None,
          market["captured_at"] if market else None))
    conn.commit()
    return True


# --- Сверка с результатами ---

def reconcile_totals(conn):
    """Дозаполняет actual_* в predictions_totals по завершённым матчам.
    Возвращает число сверенных строк. (1X2 сверяет существующая
    reconcile_predictions в bot.py — она работает и для строк PD.)"""
    rows = conn.execute("""
        SELECT p.id, m.regular_home, m.regular_away
        FROM predictions_totals p
        JOIN matches m ON p.match_id = m.id
        WHERE p.actual_total IS NULL
          AND m.status = 'FINISHED'
          AND m.regular_home IS NOT NULL
    """).fetchall()
    for pred_id, hg, ag in rows:
        total = hg + ag
        conn.execute("""
            UPDATE predictions_totals
            SET actual_total = ?, actual_over25 = ?, actual_btts = ?
            WHERE id = ?
        """, (total, 1 if total >= 3 else 0,
              1 if (hg >= 1 and ag >= 1) else 0, pred_id))
    conn.commit()
    return len(rows)


# --- Статистика для /accuracy_liga ---

def accuracy_liga_stats(conn):
    """Собирает всё, что нужно для ответа /accuracy_liga, в один dict."""
    out = {"n_1x2": 0, "n_totals": 0}

    rows = conn.execute("""
        SELECT p_home, p_draw, p_away, actual_result, was_correct,
               is_fallback, market_p_home, market_p_draw, market_p_away
        FROM predictions
        WHERE competition_code = ? AND actual_result IS NOT NULL
    """, (LIGA_COMPETITION,)).fetchall()

    if rows:
        n = len(rows)
        brier_sum = 0.0
        market_brier_sum, paired_model_brier_sum, market_n = 0.0, 0.0, 0
        correct = 0
        fb_n, fb_brier_sum = 0, 0.0
        for ph, pd_, pa, actual, was_correct, is_fb, mh, md, ma in rows:
            probs = {"HOME_WIN": ph, "DRAW": pd_, "AWAY_WIN": pa}
            b = sum((probs[k] - (1.0 if k == actual else 0.0)) ** 2 for k in probs)
            brier_sum += b
            correct += was_correct or 0
            if is_fb:
                fb_n += 1
                fb_brier_sum += b
            if mh is not None and md is not None and ma is not None:
                mp = {"HOME_WIN": mh, "DRAW": md, "AWAY_WIN": ma}
                market_brier_sum += sum(
                    (mp[k] - (1.0 if k == actual else 0.0)) ** 2 for k in mp)
                paired_model_brier_sum += b
                market_n += 1
        out.update({
            "n_1x2": n,
            "accuracy": correct / n,
            "brier_1x2": brier_sum / n / 3,
            "market_n": market_n,
        })
        if market_n:
            out["market_brier_1x2"] = market_brier_sum / market_n / 3
            out["paired_model_brier_1x2"] = paired_model_brier_sum / market_n / 3
            out["gap_1x2"] = out["paired_model_brier_1x2"] - out["market_brier_1x2"]
        if fb_n:
            out["fallback_n"] = fb_n
            out["fallback_brier_1x2"] = fb_brier_sum / fb_n / 3

    trows = conn.execute("""
        SELECT p_over25, p_btts, actual_over25, actual_btts,
               market_p_over25, is_fallback
        FROM predictions_totals
        WHERE competition_code = ? AND actual_total IS NOT NULL
    """, (LIGA_COMPETITION,)).fetchall()

    if trows:
        n = len(trows)
        over_brier = sum((po - ao) ** 2 for po, _, ao, _, _, _ in trows) / n
        btts_brier = sum((pb - ab) ** 2 for _, pb, _, ab, _, _ in trows) / n
        over_correct = sum(1 for po, _, ao, _, _, _ in trows
                           if (po >= 0.5) == bool(ao))
        paired = [(po, ao, mo) for po, _, ao, _, mo, _ in trows if mo is not None]
        out.update({
            "n_totals": n,
            "brier_over25": over_brier,
            "brier_btts": btts_brier,
            "accuracy_over25": over_correct / n,
            "market_n_totals": len(paired),
        })
        if paired:
            out["market_brier_over25"] = sum((mo - ao) ** 2 for _, ao, mo in paired) / len(paired)
            out["paired_model_brier_over25"] = sum((po - ao) ** 2 for po, ao, _ in paired) / len(paired)
            out["gap_over25"] = out["paired_model_brier_over25"] - out["market_brier_over25"]

    return out
