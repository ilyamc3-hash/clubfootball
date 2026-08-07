"""
Football Prediction Lab — Telegram-бот (aiogram 3).

ЧМ-2026 завершён — бот работает по Ла Лиге (LaLigaModel: Elo + HomeAdvantage,
без Form/Momentum — отключены как вредные по ablation-тесту). Пуассон-модель
(attack/defense по забитым голам) используется отдельно для тоталов/симуляции.
Данные читаются из той же football.db.

Команды:
    /liga      — автопоиск ближайших матчей Ла Лиги + прогноз П1/Х/П2
    /totals    — тоталы и "обе забьют" по ближайшим матчам (Пуассон-модель)
    /match Реал Мадрид Барселона   — прогноз на конкретную пару клубов вручную
    /ratings   — топ-15 клубов по текущему рейтингу

Установка:
    pip install aiogram --break-system-packages   (на Windows: pip install aiogram)

Перед запуском:
    1. Получи токен бота у @BotFather в Telegram
    2. Впиши его в BOT_TOKEN ниже (или в переменную окружения BOT_TOKEN)
    3. football.db и fifa_ratings.py должны лежать в той же папке

Запуск:
    py bot.py
"""

import asyncio
import logging
import os
import re
import sqlite3
import math
import random
import sys
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from aiogram import Bot, Dispatcher, BaseMiddleware
from aiogram.filters import Command
from aiogram.types import Message, BotCommand, BotCommandScopeDefault, BotCommandScopeChat
from laliga_model import LaLigaModel, HOME_ADVANTAGE, predict_probs

laliga_model = None  # строится лениво при первом обращении к клубным командам, чтобы не тормозить старт бота

# Русские алиасы клубов Ла Лиги -> имя, как оно хранится в football.db
# (приходит из football-data.org). Намеренно НЕ включает короткие
# неоднозначные токены вроде "реал" или "депортиво" — они разрешаются
# через частичное совпадение в resolve_club_name(), которое само
# обнаруживает и репортит неоднозначность, а не молча выбирает один клуб.
CLUB_ALIASES = {
    "атлетик бильбао": "Athletic Club", "атлетик": "Athletic Club",
    "осасуна": "CA Osasuna",
    "атлетико мадрид": "Club Atlético de Madrid", "атлетико": "Club Atlético de Madrid",
    "алавес": "Deportivo Alavés", "депортиво алавес": "Deportivo Alavés",
    "эльче": "Elche CF",
    "барселона": "FC Barcelona",
    "хетафе": "Getafe CF",
    "леванте": "Levante UD",
    "малага": "Málaga CF",
    "сельта": "RC Celta de Vigo", "сельта виго": "RC Celta de Vigo",
    "депортиво ла корунья": "RC Deportivo La Coruña",
    "эспаньол": "RCD Espanyol de Barcelona",
    "райо вальекано": "Rayo Vallecano de Madrid", "райо": "Rayo Vallecano de Madrid",
    "реал бетис": "Real Betis Balompié", "бетис": "Real Betis Balompié",
    "реал мадрид": "Real Madrid CF",
    "расинг сантандер": "Real Racing Club de Santander",
    "реал сосьедад": "Real Sociedad de Fútbol", "сосьедад": "Real Sociedad de Fútbol",
    "севилья": "Sevilla FC",
    "валенсия": "Valencia CF",
    "вильярреал": "Villarreal CF",
}


def get_known_pd_teams(conn):
    """Список названий клубов Ла Лиги как они хранятся в football.db (имена football-data.org)."""
    rows = conn.execute("""
        SELECT DISTINCT t.name FROM teams t
        JOIN matches m ON t.id = m.home_team_id OR t.id = m.away_team_id
        WHERE m.competition_code = 'PD'
    """).fetchall()
    return [r[0] for r in rows]


def _contains_as_word(haystack: str, needle: str) -> bool:
    """Проверяет вхождение needle в haystack по границам слов, а не как
    голый substring — иначе "real" находился бы и внутри "Villarreal"."""
    return re.search(r"(?<!\w)" + re.escape(needle) + r"(?!\w)", haystack) is not None


def resolve_club_name(raw_name, known_names):
    """Пытается однозначно определить клуб по вводу пользователя.

    Возвращает (имя, None) при однозначном совпадении,
    (None, [кандидаты]) при неоднозначности (нужно уточнение у пользователя),
    (None, []) если вообще ничего не найдено.

    Тиры проверки (от точных к размытым): точный алиас -> точное английское
    имя -> частичное совпадение по алиасам -> частичное совпадение по
    английским именам. Частичные тиры могут дать несколько разных клубов
    (например "реал" -> Реал Мадрид/Сосьедад/Бетис) — в этом случае
    результат считается неоднозначным, а не берётся первый попавшийся.
    """
    normalized = raw_name.strip().lower()
    if not normalized:
        return None, []

    if normalized in CLUB_ALIASES:
        return CLUB_ALIASES[normalized], None

    for name in known_names:
        if name.lower() == normalized:
            return name, None

    alias_matches = sorted({v for k, v in CLUB_ALIASES.items() if _contains_as_word(k, normalized)})
    if len(alias_matches) == 1:
        return alias_matches[0], None
    if len(alias_matches) > 1:
        return None, alias_matches

    name_matches = sorted({name for name in known_names if _contains_as_word(name.lower(), normalized)})
    if len(name_matches) == 1:
        return name_matches[0], None
    if len(name_matches) > 1:
        return None, name_matches

    return None, []


def parse_two_clubs(tokens, known_names):
    """Аналог parse_two_teams, но для клубов: перебирает точку разреза
    многословного ввода и требует, чтобы ОБЕ половины разрешились
    однозначно через resolve_club_name. Возвращает
    (team_a, team_b, ambiguous_a, ambiguous_b) — team_x is None, если
    не распознано или неоднозначно (тогда ambiguous_x — список кандидатов
    или пустой список, если вообще не распознано)."""
    n = len(tokens)
    for k in range(1, n):
        part1 = " ".join(tokens[:k])
        part2 = " ".join(tokens[k:])
        r1, _ = resolve_club_name(part1, known_names)
        r2, _ = resolve_club_name(part2, known_names)
        if r1 and r2:
            return r1, r2, None, None

    # ни один разрез не дал однозначного результата с обеих сторон —
    # запасной вариант: первое слово / остальное, с честным репортом
    # неоднозначности/нераспознавания по каждой половине отдельно
    part1 = tokens[0]
    part2 = " ".join(tokens[1:])
    r1, amb1 = resolve_club_name(part1, known_names)
    r2, amb2 = resolve_club_name(part2, known_names)
    return r1, r2, (amb1 if not r1 else None), (amb2 if not r2 else None)


async def ensure_laliga_model(message: Message):
    """Обеспечивает актуальную (пересобранную при необходимости) модель
    Ла Лиги. При ошибке отсутствующих файлов отвечает пользователю сама
    и возвращает None — вызывающий обработчик должен в этом случае просто
    завершиться (return)."""
    global laliga_model
    current_finished = LaLigaModel.count_finished_pd(DB_PATH)
    needs_rebuild = (
        laliga_model is None
        or laliga_model.db_finished_count != current_finished
    )
    if needs_rebuild:
        try:
            laliga_model = LaLigaModel(db_path=DB_PATH)
        except FileNotFoundError as e:
            await message.answer(
                f"Не найдены файлы модели Ла Лиги ({e.filename}). "
                f"Нужны laliga_matches_combined.csv и team_name_mapping.csv "
                f"в папке с bot.py."
            )
            return None
    return laliga_model


BOT_TOKEN = os.environ["BOT_TOKEN"]
DB_PATH = "football.db"
MAX_GOALS = 8

# Твой Telegram user_id (не username!) — узнать можно у @userinfobot.
# Только эти ID смогут пользоваться /admin_* командами.
ADMIN_IDS = {0}  # ЗАМЕНИ 0 на свой реальный Telegram user_id

logging.basicConfig(level=logging.INFO)
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()


# ---------- Общая логика (та же, что в compute_ratings.py / predict_match.py) ----------

def load_finished_matches(conn, competition_code="WC"):
    cur = conn.execute("""
        SELECT m.utc_date, t1.name, t2.name, m.regular_home, m.regular_away
        FROM matches m
        JOIN teams t1 ON m.home_team_id = t1.id
        JOIN teams t2 ON m.away_team_id = t2.id
        WHERE m.status = 'FINISHED' AND m.regular_home IS NOT NULL
          AND m.competition_code = ?
        ORDER BY m.utc_date ASC
    """, (competition_code,))
    return cur.fetchall()


def find_upcoming_matches(conn, competition_code="WC"):
    cur = conn.execute("""
        SELECT m.id, m.utc_date, t1.name, t2.name
        FROM matches m
        JOIN teams t1 ON m.home_team_id = t1.id
        JOIN teams t2 ON m.away_team_id = t2.id
        WHERE m.status IN ('TIMED', 'SCHEDULED')
          AND t1.name IS NOT NULL AND t2.name IS NOT NULL
          AND m.competition_code = ?
        ORDER BY m.utc_date ASC
    """, (competition_code,))
    rows = cur.fetchall()
    if not rows:
        return [], None
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    earliest = datetime.strptime(rows[0][1], fmt)
    window = []
    for match_id, date, h, a in rows:
        dt = datetime.strptime(date, fmt)
        if (dt - earliest).total_seconds() <= 20 * 3600:
            window.append((match_id, h, a))
    return window, rows[0][1][:16]


def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / math.factorial(k)


def build_attack_defense(matches):
    gf, ga, games = {}, {}, {}
    total_goals, total_games = 0, 0
    for _, home, away, hg, ag in matches:
        for team, s, c in [(home, hg, ag), (away, ag, hg)]:
            gf[team] = gf.get(team, 0) + s
            ga[team] = ga.get(team, 0) + c
            games[team] = games.get(team, 0) + 1
        total_goals += hg + ag
        total_games += 2
    league_avg = total_goals / total_games if total_games else 1.4
    attack = {t: (gf[t] / games[t]) / league_avg for t in games}
    defense = {t: (ga[t] / games[t]) / league_avg for t in games}
    return attack, defense, league_avg


def poisson_summary(team_a, team_b, attack, defense, league_avg):
    att_a, def_a = attack.get(team_a, 1.0), defense.get(team_a, 1.0)
    att_b, def_b = attack.get(team_b, 1.0), defense.get(team_b, 1.0)
    lam_a = att_a * def_b * league_avg
    lam_b = att_b * def_a * league_avg

    over25, btts_yes = 0.0, 0.0
    score_probs = {}
    for i in range(MAX_GOALS + 1):
        for j in range(MAX_GOALS + 1):
            p = poisson_pmf(i, lam_a) * poisson_pmf(j, lam_b)
            score_probs[(i, j)] = p
            if i + j >= 3:
                over25 += p
            if i >= 1 and j >= 1:
                btts_yes += p

    top_scores = sorted(score_probs.items(), key=lambda kv: kv[1], reverse=True)[:5]
    return lam_a, lam_b, over25, btts_yes, top_scores


def sample_poisson(lam):
    """Случайное число голов по распределению Пуассона (алгоритм Кнута, без numpy)."""
    l = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= random.random()
        if p <= l:
            return k - 1


def simulate_match(lambda_a, lambda_b, n):
    """Разыгрывает матч n раз случайными числами голов (Монте-Карло),
    вместо точного математического расчёта — должно сходиться к тем же
    цифрам, что и poisson_summary при большом n."""
    wins_a, draws, wins_b = 0, 0, 0
    score_counter = {}
    for _ in range(n):
        ga, gb = sample_poisson(lambda_a), sample_poisson(lambda_b)
        if ga > gb:
            wins_a += 1
        elif ga == gb:
            draws += 1
        else:
            wins_b += 1
        score_counter[(ga, gb)] = score_counter.get((ga, gb), 0) + 1
    return wins_a, draws, wins_b, score_counter


MODEL_VERSION = "v0.3-elo-form-momentum-fifa"


def ensure_predictions_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id INTEGER,
            predicted_at TEXT,
            home_team TEXT,
            away_team TEXT,
            p_home REAL,
            p_draw REAL,
            p_away REAL,
            model_version TEXT,
            actual_result TEXT,
            was_correct INTEGER
        )
    """)
    conn.commit()


def save_prediction(conn, match_id, team_a, team_b, p_a, p_draw, p_b):
    existing = conn.execute(
        "SELECT id FROM predictions WHERE match_id = ?", (match_id,)
    ).fetchone()
    if existing:
        return  # прогноз на этот матч уже сохранён — не дублируем
    conn.execute("""
        INSERT INTO predictions
            (match_id, predicted_at, home_team, away_team, p_home, p_draw, p_away, model_version)
        VALUES (?, datetime('now'), ?, ?, ?, ?, ?, ?)
    """, (match_id, team_a, team_b, p_a, p_draw, p_b, MODEL_VERSION))
    conn.commit()


def reconcile_predictions(conn):
    """Находит прогнозы по уже сыгранным матчам и дозаполняет реальный результат.
    Возвращает количество только что сверенных строк."""
    cur = conn.execute("""
        SELECT DISTINCT p.match_id
        FROM predictions p
        JOIN matches m ON p.match_id = m.id
        WHERE p.actual_result IS NULL
          AND m.status = 'FINISHED'
          AND m.regular_home IS NOT NULL
    """)
    match_ids = [row[0] for row in cur.fetchall()]

    resolved_count = 0
    for match_id in match_ids:
        row = conn.execute(
            "SELECT regular_home, regular_away FROM matches WHERE id = ?", (match_id,)
        ).fetchone()
        if not row:
            continue
        hg, ag = row
        if hg > ag:
            actual = "HOME_WIN"
        elif hg == ag:
            actual = "DRAW"
        else:
            actual = "AWAY_WIN"

        preds = conn.execute(
            "SELECT id, p_home, p_draw, p_away FROM predictions WHERE match_id = ? AND actual_result IS NULL",
            (match_id,)
        ).fetchall()

        for pred_id, p_home, p_draw, p_away in preds:
            probs = {"HOME_WIN": p_home, "DRAW": p_draw, "AWAY_WIN": p_away}
            predicted_outcome = max(probs, key=probs.get)
            was_correct = 1 if predicted_outcome == actual else 0
            conn.execute(
                "UPDATE predictions SET actual_result = ?, was_correct = ? WHERE id = ?",
                (actual, was_correct, pred_id)
            )
            resolved_count += 1

    conn.commit()
    return resolved_count


def ensure_users_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            telegram_id INTEGER PRIMARY KEY,
            username TEXT,
            first_seen TEXT,
            last_seen TEXT,
            subscription_until TEXT
        )
    """)
    conn.commit()


def track_user(conn, from_user):
    """Записывает/обновляет пользователя при каждом обращении к боту."""
    conn.execute("""
        INSERT INTO users (telegram_id, username, first_seen, last_seen)
        VALUES (?, ?, datetime('now'), datetime('now'))
        ON CONFLICT(telegram_id) DO UPDATE SET
            username = excluded.username,
            last_seen = datetime('now')
    """, (from_user.id, from_user.username or from_user.full_name))
    conn.commit()


def is_admin(user_id: int) -> bool:
    return user_id in ADMIN_IDS


class UserTrackingMiddleware(BaseMiddleware):
    """Тихо записывает каждого, кто пишет боту, в таблицу users —
    без этого мы не сможем сделать рассылку или посчитать, сколько людей пользуется ботом."""
    async def __call__(self, handler, event, data):
        if event.from_user:
            conn = sqlite3.connect(DB_PATH)
            ensure_users_table(conn)
            track_user(conn, event.from_user)
            conn.close()
        return await handler(event, data)


# ---------- Команды бота ----------

@dp.message(Command("today"))
async def cmd_today(message: Message):
    """ЧМ-2026 завершён (финал: Испания 1:0 Аргентина) — команда больше
    не считает прогнозы, только направляет на /liga."""
    await message.answer(
        "⚽ Чемпионат мира 2026 завершён (финал: Испания 1:0 Аргентина) — "
        "прогнозы по нему больше не считаются.\n\n"
        "Сейчас бот работает по Ла Лиге — используй /liga."
    )


@dp.message(Command("liga"))
async def cmd_liga(message: Message):
    """Прогнозы на ближайшие матчи Ла Лиги (клубный футбол).
    Использует отдельную Elo-модель (laliga_model.py): история 12 сезонов
    football-data.co.uk + дообновление сыгранными матчами текущего сезона
    из football.db. Модель пересобирается автоматически, когда в базе
    появляются новые завершённые матчи (после fetch_matches.py)."""
    model = await ensure_laliga_model(message)
    if model is None:
        return

    conn = sqlite3.connect(DB_PATH)
    upcoming, when = find_upcoming_matches(conn, competition_code="PD")
    conn.close()

    if not upcoming:
        await message.answer("Не найдено предстоящих матчей Ла Лиги. Нужно обновить базу (fetch_matches.py).")
        return

    lines = [f"⚽ Ближайшие матчи Ла Лиги ({when} UTC):\n"]
    for match_id, team_a, team_b in upcoming:
        result = laliga_model.predict(team_a, team_b)
        p_a, p_draw, p_b = result["p_home"], result["p_draw"], result["p_away"]

        lines.append(f"<b>{team_a} — {team_b}</b>")
        lines.append(f"Elo: {result['elo_home']:.0f} vs {result['elo_away']:.0f}")
        if not result["home_known"]:
            lines.append(f"⚠ {team_a} — нет истории в базе (средний рейтинг лиги)")
        if not result["away_known"]:
            lines.append(f"⚠ {team_b} — нет истории в базе (средний рейтинг лиги)")
        lines.append(f"П1 {p_a*100:.1f}%  Х {p_draw*100:.1f}%  П2 {p_b*100:.1f}%")

        outcomes = [("П1", p_a, team_a), ("Х", p_draw, None), ("П2", p_b, team_b)]
        best_label, best_p, best_team = max(outcomes, key=lambda x: x[1])
        gap = best_p - sorted([p_a, p_draw, p_b])[-2]
        if gap > 0.30:
            confidence = "высокая уверенность"
        elif gap > 0.15:
            confidence = "средняя уверенность"
        else:
            confidence = "низкая уверенность, матч равный"
        who = best_team if best_team else "ничья"
        lines.append(f"➡ Самое вероятное: {who} ({best_label}, {best_p*100:.1f}%) — {confidence}\n")

    await message.answer("\n".join(lines), parse_mode="HTML")


@dp.message(Command("totals"))
async def cmd_totals(message: Message):
    conn = sqlite3.connect(DB_PATH)
    finished = load_finished_matches(conn, competition_code="PD")
    upcoming, when = find_upcoming_matches(conn, competition_code="PD")
    conn.close()

    if not upcoming:
        await message.answer("Не найдено предстоящих матчей Ла Лиги.")
        return

    attack, defense, league_avg = build_attack_defense(finished)
    lines = [f"📊 Тоталы и обе забьют, Ла Лига ({when} UTC):\n"]
    if not finished:
        lines.append(
            "⚠ Сезон ещё не начался (нет сыгранных матчей в базе) — цифры ниже "
            "усреднены по лиге, а не по конкретным командам. Точность вырастет "
            "после первых туров.\n"
        )
    for match_id, team_a, team_b in upcoming:
        lam_a, lam_b, over25, btts, top_scores = poisson_summary(team_a, team_b, attack, defense, league_avg)
        lines.append(f"<b>{team_a} — {team_b}</b>")
        lines.append(f"Ож. голы: {lam_a:.2f} — {lam_b:.2f}")
        lines.append(f"Тотал больше 2.5: {over25*100:.1f}%   Обе забьют: {btts*100:.1f}%")
        lines.append("Самые вероятные счета:")
        for (i, j), p in top_scores:
            lines.append(f"  {i}:{j} — {p*100:.1f}%")
        lines.append("")

    await message.answer("\n".join(lines), parse_mode="HTML")


@dp.message(Command("simulate"))
async def cmd_simulate(message: Message):
    """Разыгрывает ближайшие матчи 100 000 раз случайным образом (Монте-Карло),
    вместо точного расчёта — для наглядности, что итог должен сходиться
    к тем же цифрам, что и /totals."""
    conn = sqlite3.connect(DB_PATH)
    finished = load_finished_matches(conn, competition_code="PD")
    upcoming, when = find_upcoming_matches(conn, competition_code="PD")
    conn.close()

    if not upcoming:
        await message.answer("Не найдено предстоящих матчей Ла Лиги.")
        return

    attack, defense, league_avg = build_attack_defense(finished)
    n = 1000000

    lines = [f"🎲 Симуляция {n} раз, Ла Лига ({when} UTC):\n"]
    if not finished:
        lines.append(
            "⚠ Сезон ещё не начался (нет сыгранных матчей в базе) — цифры ниже "
            "усреднены по лиге, а не по конкретным командам.\n"
        )
    for match_id, team_a, team_b in upcoming:
        att_a, def_a = attack.get(team_a, 1.0), defense.get(team_a, 1.0)
        att_b, def_b = attack.get(team_b, 1.0), defense.get(team_b, 1.0)
        lam_a = att_a * def_b * league_avg
        lam_b = att_b * def_a * league_avg

        wins_a, draws, wins_b, score_counter = simulate_match(lam_a, lam_b, n)
        top5 = sorted(score_counter.items(), key=lambda kv: kv[1], reverse=True)[:5]

        lines.append(f"<b>{team_a} — {team_b}</b>  (λ = {lam_a:.2f} — {lam_b:.2f})")
        lines.append(f"{team_a} победила: {wins_a} раз ({wins_a/n*100:.1f}%)")
        lines.append(f"Ничья: {draws} раз ({draws/n*100:.1f}%)")
        lines.append(f"{team_b} победила: {wins_b} раз ({wins_b/n*100:.1f}%)")
        lines.append("Частые счета:")
        for (i, j), count in top5:
            lines.append(f"  {i}:{j} — {count/n*100:.1f}%")
        lines.append("")

    await message.answer("\n".join(lines), parse_mode="HTML")


async def _resolve_match_pair(message: Message, tokens):
    """Общая логика распознавания пары клубов для /match, /explain, /diff.
    Возвращает (team_a, team_b) при успехе; при неоднозначности или
    нераспознанном названии сама отвечает пользователю и возвращает None."""
    conn = sqlite3.connect(DB_PATH)
    known_names = get_known_pd_teams(conn)
    conn.close()

    team_a, team_b, amb_a, amb_b = parse_two_clubs(tokens, known_names)

    if amb_a:
        await message.answer(f"⚠ Уточни клуб: {' / '.join(amb_a)}\nПовтори запрос с более точным названием.")
        return None
    if amb_b:
        await message.answer(f"⚠ Уточни клуб: {' / '.join(amb_b)}\nПовтори запрос с более точным названием.")
        return None
    if not team_a or not team_b:
        await message.answer(
            "⚠ Не смог распознать название клуба.\n"
            "Проверь написание (например: Реал Мадрид, Барселона, Атлетико Мадрид...).\n"
            "Список всех клубов — /ratings"
        )
        return None
    return team_a, team_b


@dp.message(Command("match"))
async def cmd_match(message: Message):
    args = message.text.replace("/match", "").strip()
    tokens = [p.strip() for p in args.split(" ") if p.strip()]
    if len(tokens) < 2:
        await message.answer("Использование: /match Клуб1 Клуб2\nНапример: /match Реал Мадрид Барселона")
        return

    pair = await _resolve_match_pair(message, tokens)
    if pair is None:
        return
    team_a, team_b = pair

    model = await ensure_laliga_model(message)
    if model is None:
        return

    result = model.predict(team_a, team_b)
    p_a, p_draw, p_b = result["p_home"], result["p_draw"], result["p_away"]

    lines = [
        f"<b>{team_a} — {team_b}</b>",
        f"Elo: {result['elo_home']:.0f} vs {result['elo_away']:.0f}",
    ]
    if not result["home_known"]:
        lines.append(f"⚠ {team_a} — нет истории в базе (средний рейтинг лиги)")
    if not result["away_known"]:
        lines.append(f"⚠ {team_b} — нет истории в базе (средний рейтинг лиги)")
    lines.append(f"П1 {p_a*100:.1f}%  Х {p_draw*100:.1f}%  П2 {p_b*100:.1f}%")

    await message.answer("\n".join(lines), parse_mode="HTML")


@dp.message(Command("explain"))
async def cmd_explain(message: Message):
    """Раскладывает прогноз Ла Лиги на факторы: Elo и преимущество своего
    поля. Использование: /explain Клуб1 Клуб2

    В модели Ла Лиги нет Form/Momentum (отключены как вредные по
    ablation-тесту — см. laliga_grid_search.py), поэтому раскладка
    ограничена двумя факторами, которые в модели реально есть."""
    args = message.text.replace("/explain", "").strip()
    tokens = [p.strip() for p in args.split(" ") if p.strip()]
    if len(tokens) < 2:
        await message.answer("Использование: /explain Клуб1 Клуб2\nНапример: /explain Реал Мадрид Барселона")
        return

    pair = await _resolve_match_pair(message, tokens)
    if pair is None:
        return
    team_a, team_b = pair

    model = await ensure_laliga_model(message)
    if model is None:
        return

    result = model.predict(team_a, team_b)
    elo_a, elo_b = result["elo_home"], result["elo_away"]

    diff_elo_only = elo_a - elo_b
    diff_with_home_adv = diff_elo_only + HOME_ADVANTAGE
    p_home_base = predict_probs(diff_elo_only)[0]
    p_home_final = predict_probs(diff_with_home_adv)[0]
    home_adv_contribution = (p_home_final - p_home_base) * 100

    lines = [
        f"<b>Почему {team_a} — {team_b}?</b>\n",
        f"База (только Elo {elo_a:.0f} vs {elo_b:.0f}): {p_home_base*100:.1f}%",
        f"  Своё поле (+{HOME_ADVANTAGE} к Elo хозяев): {home_adv_contribution:+.1f} п.п.",
        f"\nИтоговая вероятность победы {team_a}: <b>{p_home_final*100:.1f}%</b>",
        "\n<i>В модели Ла Лиги нет Form/Momentum — они отключены как вредные "
        "по результатам ablation-теста, поэтому раскладка ограничена Elo и "
        "преимуществом своего поля.</i>",
    ]

    await message.answer("\n".join(lines), parse_mode="HTML")


@dp.message(Command("ratings"))
async def cmd_ratings(message: Message):
    model = await ensure_laliga_model(message)
    if model is None:
        return

    top = sorted(model.elo.items(), key=lambda kv: kv[1], reverse=True)[:15]

    lines = ["🏆 Топ-15 клубов Ла Лиги по текущему рейтингу:\n"]
    for i, (team, rating) in enumerate(top, 1):
        lines.append(f"{i}. {team} — {rating:.0f}")

    await message.answer("\n".join(lines))


@dp.message(Command("accuracy"))
async def cmd_accuracy(message: Message):
    """Показывает накопленную статистику точности модели по всем
    сохранённым и уже сверенным прогнозам (таблица predictions)."""
    conn = sqlite3.connect(DB_PATH)
    ensure_predictions_table(conn)
    just_resolved = reconcile_predictions(conn)

    rows = conn.execute("""
        SELECT p.home_team, p.away_team, p.p_home, p.p_draw, p.p_away,
               p.actual_result, p.was_correct, m.stage, m.winner
        FROM predictions p
        JOIN matches m ON p.match_id = m.id
        WHERE p.actual_result IS NOT NULL
    """).fetchall()
    conn.close()

    if not rows:
        await message.answer(
            "Пока нет сверенных прогнозов. Прогнозы сохраняются автоматически при "
            "каждом /today, а сверяются после того, как матч закончится и база "
            "обновлена через fetch_matches.py."
        )
        return

    n = len(rows)
    correct = sum(r[6] for r in rows)
    accuracy = correct / n

    eps = 1e-9
    brier_sum = 0.0
    favorite_prob_sum = 0.0  # средняя вероятность, которую модель давала фавориту
    favorite_win_count = 0   # сколько раз фаворит реально победил

    team_mistakes = {}  # команда -> (ошибок, всего прогнозов с её участием)

    # Отдельная метрика для плей-офф: "угадал, кто прошёл дальше" -
    # сравнивает фаворита модели (без учёта ничьей) с полем winner
    # из API, которое учитывает доп. время и пенальти. Считается
    # независимо от Brier/accuracy по 90 минутам, чтобы не искажать
    # статистическую калибровку модели.
    KNOCKOUT_STAGES = {
        "ROUND_OF_16", "QUARTER_FINALS", "SEMI_FINALS",
        "THIRD_PLACE", "FINAL",
    }
    playoff_total = 0
    playoff_correct = 0

    for home, away, p_home, p_draw, p_away, actual, was_correct, stage, winner in rows:
        probs = {"HOME_WIN": p_home, "DRAW": p_draw, "AWAY_WIN": p_away}
        for key in probs:
            t = 1.0 if key == actual else 0.0
            brier_sum += (probs[key] - t) ** 2

        favorite_key = max(probs, key=probs.get)
        favorite_prob_sum += probs[favorite_key]
        if favorite_key == actual:
            favorite_win_count += 1

        for team in (home, away):
            errs, total = team_mistakes.get(team, (0, 0))
            team_mistakes[team] = (errs + (0 if was_correct else 1), total + 1)

        if stage in KNOCKOUT_STAGES and winner in ("HOME_TEAM", "AWAY_TEAM"):
            # фаворит модели без учёта ничьей - кто выше, П1 или П2
            model_favorite = "HOME_TEAM" if p_home >= p_away else "AWAY_TEAM"
            playoff_total += 1
            if model_favorite == winner:
                playoff_correct += 1

    brier = brier_sum / n / 3
    avg_favorite_confidence = favorite_prob_sum / n
    actual_favorite_rate = favorite_win_count / n
    bias = (avg_favorite_confidence - actual_favorite_rate) * 100

    lines = [f"📊 История прогнозов ({n} сверенных матчей"
             + (f", +{just_resolved} только что" if just_resolved else "") + ")\n"]
    lines.append(f"Точность (accuracy): {accuracy*100:.1f}%")
    lines.append(f"Brier score: {brier:.4f}")

    if playoff_total > 0:
        playoff_acc = playoff_correct / playoff_total * 100
        lines.append(
            f"Точность по плей-офф (с учётом доп. времени/пенальти): "
            f"{playoff_acc:.1f}% ({playoff_correct}/{playoff_total})"
        )

    lines.append("")

    if bias > 3:
        lines.append(f"⚠ Модель переоценивает фаворитов на ~{bias:.1f} п.п. "
                      f"(даёт им в среднем {avg_favorite_confidence*100:.1f}%, "
                      f"а реально побеждают в {actual_favorite_rate*100:.1f}% случаев)")
    elif bias < -3:
        lines.append(f"⚠ Модель недооценивает фаворитов на ~{abs(bias):.1f} п.п.")
    else:
        lines.append(f"Калибровка по фаворитам в норме (расхождение {bias:+.1f} п.п.)")

    # команды, где модель ошибается чаще всего (минимум 2 прогноза, чтобы не шумело)
    worst_teams = sorted(
        [(team, errs, total) for team, (errs, total) in team_mistakes.items() if total >= 2],
        key=lambda x: x[1] / x[2], reverse=True
    )[:5]
    if worst_teams:
        lines.append("\nЧаще всего модель ошибается с участием:")
        for team, errs, total in worst_teams:
            lines.append(f"  {team}: {errs} ошибок из {total} прогнозов")

    await message.answer("\n".join(lines))


@dp.message(Command("diff"))
async def cmd_diff(message: Message):
    """Использование: /diff Реал Мадрид Барселона 2.10 3.40 3.20
    (клуб1 клуб2 кэф_П1 кэф_Х кэф_П2 — коэффициенты на основное время)"""
    args = message.text.replace("/diff", "").strip().split()
    if len(args) < 5:
        await message.answer(
            "Использование: /diff Клуб1 Клуб2 кэф_П1 кэф_Х кэф_П2\n"
            "Например: /diff Реал Мадрид Барселона 2.10 3.40 3.20"
        )
        return

    def is_float(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    # Последние 3 токена — коэффициенты, всё, что до них — названия клубов
    # (могут быть многословными, например "Реал Сосьедад")
    if not all(is_float(x) for x in args[-3:]) or len(args) < 5:
        await message.answer(
            "Использование: /diff Клуб1 Клуб2 кэф_П1 кэф_Х кэф_П2\n"
            "Например: /diff Реал Мадрид Барселона 2.10 3.40 3.20"
        )
        return

    name_tokens = args[:-3]
    odds_home, odds_draw, odds_away = (float(x) for x in args[-3:])

    pair = await _resolve_match_pair(message, name_tokens)
    if pair is None:
        return
    team_a, team_b = pair

    model = await ensure_laliga_model(message)
    if model is None:
        return

    result = model.predict(team_a, team_b)
    p_model_a, p_model_draw, p_model_b = result["p_home"], result["p_draw"], result["p_away"]

    raw = {"home": 1 / odds_home, "draw": 1 / odds_draw, "away": 1 / odds_away}
    total = sum(raw.values())
    market = {k: v / total for k, v in raw.items()}
    overround = (total - 1) * 100

    def gap(model_p, market_p):
        return (model_p - market_p) * 100

    lines = [f"<b>{team_a} — {team_b}</b>\n"]
    lines.append(f"{'Исход':<6}{'Модель':>10}{'Рынок':>10}{'Разница':>12}")
    max_gap = 0.0
    for label, model_p, market_p in [
        ("П1", p_model_a, market["home"]),
        ("Х", p_model_draw, market["draw"]),
        ("П2", p_model_b, market["away"]),
    ]:
        g = gap(model_p, market_p)
        max_gap = max(max_gap, abs(g))
        lines.append(f"{label:<6}{model_p*100:>9.1f}%{market_p*100:>9.1f}%{g:>+11.1f}п.п.")

    lines.append(f"\nМаржа букмекера: ~{overround:.1f}%")

    if max_gap >= 10.0:
        lines.append(
            f"\n⚠ Расхождение {max_gap:.1f} п.п. — скорее всего, ошибка нашей модели, "
            "а не находка. У букмекера обычно больше информации (составы, инсайды)."
        )

    await message.answer("\n".join(lines), parse_mode="HTML")


@dp.message(Command("arbitrage"))
async def cmd_arbitrage(message: Message):
    """Использование: /arbitrage [Имя1] П1 Х П2 [Имя2] П1 Х П2 ... [stake=СУММА] [round=ШАГ]
    Название букмекера перед тройкой коэффициентов — необязательно.
    stake — сумма для расчёта (по умолчанию 10000).
    round — шаг округления ставки на каждый исход, чтобы не выглядело подозрительно
            для букмекера (по умолчанию 10, можно поставить 100 и т.д.).

    Примеры:
      /arbitrage Fonbet 4.05 3.70 1.93 WINLINE 4.20 3.60 1.90
      /arbitrage Fonbet 4.05 3.70 1.93 WINLINE 4.20 3.60 1.90 stake=5000 round=100
    """
    raw_args = message.text.replace("/arbitrage", "").strip().split()

    def is_float(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    # Вытаскиваем stake= и round=, если есть
    total_stake = 10000.0
    round_step = 10.0
    filtered_args = []
    for token in raw_args:
        low = token.lower()
        if low.startswith("stake="):
            value = low.split("=", 1)[1]
            if is_float(value):
                total_stake = float(value)
        elif low.startswith("round="):
            value = low.split("=", 1)[1]
            if is_float(value) and float(value) > 0:
                round_step = float(value)
        else:
            filtered_args.append(token)
    raw_args = filtered_args

    bookmakers = []
    i, counter = 0, 1
    parse_error = False

    while i < len(raw_args):
        if not is_float(raw_args[i]):
            name = raw_args[i]
            i += 1
        else:
            name = f"БК{counter}"

        if i + 3 > len(raw_args) or not all(is_float(x) for x in raw_args[i:i + 3]):
            parse_error = True
            break

        h, d, a = (float(x) for x in raw_args[i:i + 3])
        bookmakers.append((name, h, d, a))
        i += 3
        counter += 1

    if parse_error or not bookmakers:
        await message.answer(
            "Использование: /arbitrage [Имя1] П1 Х П2 [Имя2] П1 Х П2 ... [stake=СУММА] [round=ШАГ]\n"
            "Имя, сумма ставки и шаг округления — необязательны.\n\n"
            "Например:\n"
            "/arbitrage Fonbet 4.05 3.70 1.93 WINLINE 4.20 3.60 1.90 stake=5000 round=100"
        )
        return

    best = {"home": (0, None), "draw": (0, None), "away": (0, None)}
    for name, h, d, a in bookmakers:
        for key, odds in [("home", h), ("draw", d), ("away", a)]:
            if odds > best[key][0]:
                best[key] = (odds, name)

    implied_sum = sum(1 / odds for odds, _ in best.values())
    labels = {"home": "П1", "draw": "Х", "away": "П2"}

    lines = ["Лучшие коэффициенты по каждому исходу:"]
    for key, (odds, book) in best.items():
        lines.append(f"  {labels[key]}: {odds}  ({book})")

    lines.append(f"\nСумма обратных коэффициентов: {implied_sum*100:.2f}%")

    if implied_sum < 1.0:
        margin = (1 - implied_sum) * 100
        lines.append(f"✓ ВИЛКА НАЙДЕНА — теоретическая маржа: {margin:.2f}%")
        lines.append(f"(ставки округлены до {round_step:.0f}, чтобы не выглядеть подозрительно)\n")

        rounded_stakes = {}
        for key, (odds, book) in best.items():
            raw_stake = (1 / odds) / implied_sum * total_stake
            rounded = round(raw_stake / round_step) * round_step
            rounded_stakes[key] = rounded
            payout = rounded * odds
            lines.append(f"  {labels[key]} ({book}, {odds}): поставить {rounded:.0f}  → выплата {payout:.0f}")

        actual_spent = sum(rounded_stakes.values())
        min_payout = min(rounded_stakes[key] * best[key][0] for key in best)
        actual_profit = min_payout - actual_spent

        lines.append(f"\nФактически потрачено (после округления): {actual_spent:.0f}")
        lines.append(f"Гарантированная прибыль: {actual_profit:.0f}")

        if actual_profit <= 0:
            lines.append(
                "\n⚠ После округления вилка исчезла (прибыль ушла в 0 или минус) — "
                "попробуй меньший шаг округления (round=) или большую сумму ставки (stake=)."
            )
    else:
        overround = (implied_sum - 1) * 100
        lines.append(f"✗ Вилки нет — суммарная маржа рынка {overround:.2f}%")

    await message.answer("\n".join(lines))


@dp.message(Command("arbitrage2"))
async def cmd_arbitrage2(message: Message):
    """То же самое, что /arbitrage, но для рынков с ДВУМЯ исходами:
    тотал больше/меньше, обе забьют да/нет, фора и т.п.

    Использование: /arbitrage2 [Имя] исход1 исход2 [Имя] исход1 исход2 ... [stake=] [round=]

    Например:
      /arbitrage2 Fonbet 1.95 1.90 WINLINE 2.05 1.85 stake=5000
    """
    raw_args = message.text.replace("/arbitrage2", "").strip().split()

    def is_float(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    total_stake = 10000.0
    round_step = 10.0
    filtered_args = []
    for token in raw_args:
        low = token.lower()
        if low.startswith("stake="):
            value = low.split("=", 1)[1]
            if is_float(value):
                total_stake = float(value)
        elif low.startswith("round="):
            value = low.split("=", 1)[1]
            if is_float(value) and float(value) > 0:
                round_step = float(value)
        else:
            filtered_args.append(token)
    raw_args = filtered_args

    bookmakers = []
    i, counter = 0, 1
    parse_error = False

    while i < len(raw_args):
        if not is_float(raw_args[i]):
            name = raw_args[i]
            i += 1
        else:
            name = f"БК{counter}"

        if i + 2 > len(raw_args) or not all(is_float(x) for x in raw_args[i:i + 2]):
            parse_error = True
            break

        o1, o2 = (float(x) for x in raw_args[i:i + 2])
        bookmakers.append((name, o1, o2))
        i += 2
        counter += 1

    if parse_error or not bookmakers:
        await message.answer(
            "Использование: /arbitrage2 [Имя] исход1 исход2 [Имя] исход1 исход2 ... [stake=] [round=]\n"
            "Для рынков с ДВУМЯ исходами: тотал больше/меньше, обе забьют, фора.\n\n"
            "Например:\n"
            "/arbitrage2 Fonbet 1.95 1.90 WINLINE 2.05 1.85 stake=5000"
        )
        return

    best = {"outcome1": (0, None), "outcome2": (0, None)}
    for name, o1, o2 in bookmakers:
        for key, odds in [("outcome1", o1), ("outcome2", o2)]:
            if odds > best[key][0]:
                best[key] = (odds, name)

    implied_sum = sum(1 / odds for odds, _ in best.values())

    lines = ["Лучшие коэффициенты по каждому исходу:"]
    lines.append(f"  Исход 1: {best['outcome1'][0]}  ({best['outcome1'][1]})")
    lines.append(f"  Исход 2: {best['outcome2'][0]}  ({best['outcome2'][1]})")
    lines.append(f"\nСумма обратных коэффициентов: {implied_sum*100:.2f}%")

    if implied_sum < 1.0:
        margin = (1 - implied_sum) * 100
        lines.append(f"✓ ВИЛКА НАЙДЕНА — теоретическая маржа: {margin:.2f}%")
        lines.append(f"(ставки округлены до {round_step:.0f})\n")

        rounded_stakes = {}
        for key, (odds, book) in best.items():
            raw_stake = (1 / odds) / implied_sum * total_stake
            rounded = round(raw_stake / round_step) * round_step
            rounded_stakes[key] = rounded
            payout = rounded * odds
            label = "Исход 1" if key == "outcome1" else "Исход 2"
            lines.append(f"  {label} ({book}, {odds}): поставить {rounded:.0f}  → выплата {payout:.0f}")

        actual_spent = sum(rounded_stakes.values())
        min_payout = min(rounded_stakes[key] * best[key][0] for key in best)
        actual_profit = min_payout - actual_spent

        lines.append(f"\nФактически потрачено: {actual_spent:.0f}")
        lines.append(f"Гарантированная прибыль: {actual_profit:.0f}")

        if actual_profit <= 0:
            lines.append(
                "\n⚠ После округления вилка исчезла — попробуй меньший шаг (round=) "
                "или большую сумму (stake=)."
            )
    else:
        overround = (implied_sum - 1) * 100
        lines.append(f"✗ Вилки нет — суммарная маржа рынка {overround:.2f}%")

    await message.answer("\n".join(lines))


# ---------- Админ-команды (доступны только ID из ADMIN_IDS) ----------

@dp.message(Command("admin_stats"))
async def cmd_admin_stats(message: Message):
    if not is_admin(message.from_user.id):
        return  # молча игнорируем — не палим, что команда вообще существует

    conn = sqlite3.connect(DB_PATH)
    ensure_users_table(conn)

    total_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    active_7d = conn.execute(
        "SELECT COUNT(*) FROM users WHERE last_seen >= datetime('now', '-7 days')"
    ).fetchone()[0]
    active_subs = conn.execute(
        "SELECT COUNT(*) FROM users WHERE subscription_until >= datetime('now')"
    ).fetchone()[0]

    ensure_predictions_table(conn)
    total_predictions = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    resolved = conn.execute(
        "SELECT COUNT(*) FROM predictions WHERE actual_result IS NOT NULL"
    ).fetchone()[0]
    conn.close()

    lines = [
        "🔧 Статистика бота",
        f"Всего пользователей: {total_users}",
        f"Активны за 7 дней: {active_7d}",
        f"С активной подпиской: {active_subs}",
        f"Прогнозов сохранено: {total_predictions} (сверено: {resolved})",
    ]
    await message.answer("\n".join(lines))


@dp.message(Command("admin_broadcast"))
async def cmd_admin_broadcast(message: Message):
    if not is_admin(message.from_user.id):
        return

    text = message.text.replace("/admin_broadcast", "").strip()
    if not text:
        await message.answer("Использование: /admin_broadcast Текст сообщения для всех пользователей")
        return

    conn = sqlite3.connect(DB_PATH)
    ensure_users_table(conn)
    user_ids = [row[0] for row in conn.execute("SELECT telegram_id FROM users").fetchall()]
    conn.close()

    sent, failed = 0, 0
    for uid in user_ids:
        try:
            await bot.send_message(uid, text)
            sent += 1
        except Exception:
            failed += 1  # пользователь мог заблокировать бота — это нормально

    await message.answer(f"Рассылка отправлена: {sent} успешно, {failed} не удалось.")


@dp.message(Command("admin_grant"))
async def cmd_admin_grant(message: Message):
    """Вручную выдать подписку пользователю (пока нет автоматической оплаты).
    Использование: /admin_grant telegram_id количество_дней"""
    if not is_admin(message.from_user.id):
        return

    args = message.text.replace("/admin_grant", "").strip().split()
    if len(args) != 2 or not all(a.lstrip("-").isdigit() for a in args):
        await message.answer("Использование: /admin_grant telegram_id количество_дней\nНапример: /admin_grant 123456789 30")
        return

    target_id, days = int(args[0]), int(args[1])
    conn = sqlite3.connect(DB_PATH)
    ensure_users_table(conn)
    conn.execute("""
        UPDATE users SET subscription_until = datetime('now', ?)
        WHERE telegram_id = ?
    """, (f"+{days} days", target_id))
    conn.commit()
    conn.close()

    await message.answer(f"Пользователю {target_id} выдана подписка на {days} дней.")


@dp.message(Command("start"))
async def cmd_start(message: Message):
    await message.answer(
        "⚽ Football Prediction Lab — Ла Лига\n\n"
        "/liga — прогноз на ближайшие матчи\n"
        "/totals — тоталы и обе забьют (+ вероятные счета)\n"
        "/simulate — разыграть ближайшие матчи 1 000 000 раз (Монте-Карло)\n"
        "/match Клуб1 Клуб2 — прогноз на конкретную пару вручную\n"
        "/explain Клуб1 Клуб2 — разбивка прогноза на факторы (Elo/своё поле)\n"
        "/ratings — топ клубов по рейтингу\n"
        "/accuracy — история прогнозов ЧМ-2026 (архив, для Ла Лиги пока не ведётся)\n"
        "/diff Клуб1 Клуб2 П1 Х П2 — сравнить с коэффициентами букмекера\n"
        "/arbitrage [Имя] П1 Х П2 [Имя] П1 Х П2 ... [stake=] [round=] — поиск вилки (3 исхода)\n"
        "/arbitrage2 [Имя] исход1 исход2 ... [stake=] [round=] — вилка для тотала/обе забьют (2 исхода)"
    )


FETCH_INTERVAL_SECONDS = 3600  # раз в час


async def periodic_fetch_matches():
    """
    Фоновая задача: раз в час запускает fetch_matches.py, чтобы база
    обновлялась сама (свежие результаты, участники следующих раундов),
    пока бот работает — без необходимости запускать скрипт вручную.
    """
    while True:
        try:
            logging.info("Автообновление: запускаю fetch_matches.py...")
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "fetch_matches.py",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode == 0:
                logging.info("Автообновление: fetch_matches.py успешно отработал")
            else:
                logging.error(
                    "Автообновление: fetch_matches.py завершился с ошибкой:\n%s",
                    stderr.decode(errors="ignore"),
                )
        except Exception as e:
            logging.error("Автообновление: не удалось запустить fetch_matches.py: %s", e)

        await asyncio.sleep(FETCH_INTERVAL_SECONDS)


async def set_commands(bot: Bot):
    """
    Настраивает всплывающее меню команд в Telegram (появляется при вводе "/").
    Обычным пользователям показываются только пользовательские команды,
    админам (ADMIN_IDS) — ещё и админские.
    """
    default_commands = [
        BotCommand(command="liga", description="Прогноз на ближайшие матчи Ла Лиги"),
        BotCommand(command="totals", description="Тоталы и обе забьют"),
        BotCommand(command="simulate", description="Монте-Карло симуляция матча"),
        BotCommand(command="match", description="Прогноз на конкретную пару клубов вручную"),
        BotCommand(command="explain", description="Разбивка прогноза на факторы"),
        BotCommand(command="ratings", description="Топ клубов по рейтингу"),
        BotCommand(command="accuracy", description="Точность прошлых прогнозов (архив ЧМ)"),
        BotCommand(command="diff", description="Сравнить с коэффициентами букмекера"),
        BotCommand(command="arbitrage", description="Поиск вилки (3 исхода)"),
        BotCommand(command="arbitrage2", description="Поиск вилки (2 исхода)"),
        BotCommand(command="start", description="Список команд"),
    ]
    await bot.set_my_commands(default_commands, scope=BotCommandScopeDefault())

    admin_commands = default_commands + [
        BotCommand(command="admin_stats", description="[admin] Статистика бота"),
        BotCommand(command="admin_broadcast", description="[admin] Рассылка всем пользователям"),
        BotCommand(command="admin_grant", description="[admin] Выдать подписку пользователю"),
    ]
    for admin_id in ADMIN_IDS:
        if admin_id == 0:
            continue  # пропускаем заглушку, пока не вписан реальный ID
        try:
            await bot.set_my_commands(
                admin_commands, scope=BotCommandScopeChat(chat_id=admin_id)
            )
        except Exception as e:
            logging.warning("Не удалось задать админ-команды для %s: %s", admin_id, e)


async def main():
    dp.message.middleware(UserTrackingMiddleware())
    await set_commands(bot)
    asyncio.create_task(periodic_fetch_matches())
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
