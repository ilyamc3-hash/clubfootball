"""
Football Prediction Lab -> Club Football: модуль прогноза для бота.

Строит Elo-состояние на истории (12 сезонов football-data.co.uk,
откалиброванная конфигурация: HOME_ADVANTAGE=60, SEASON_REGRESSION=0.20,
MoV включён, form/momentum отключён - см. laliga_grid_search.py),
применяет регрессию к среднему на границе нового сезона, и затем
ДОПОЛНИТЕЛЬНО обновляет рейтинги сыгранными матчами текущего сезона
из football.db (competition_code='PD') - чтобы прогнозы на 30-й тур
учитывали результаты предыдущих 29 туров, а не оставались замороженными
на состоянии "до старта сезона".

Использование как модуля:
    from laliga_model import LaLigaModel
    model = LaLigaModel(db_path="football.db")
    result = model.predict("Real Madrid CF", "FC Barcelona")

    # проверка, не появились ли новые сыгранные матчи (для пересборки):
    if model.db_finished_count != LaLigaModel.count_finished_pd(db_path):
        model = LaLigaModel(db_path=db_path)
"""

import csv
import math
import os
import sqlite3
from datetime import datetime

HISTORY_FILE = "laliga_matches_combined.csv"
MAPPING_FILE = "team_name_mapping.csv"

BASE_ELO = 1500
K_FACTOR = 32
HOME_ADVANTAGE = 60
SEASON_REGRESSION = 0.20


def mov_multiplier(goal_diff, elo_diff_pre_match):
    goal_diff = abs(goal_diff)
    if goal_diff == 0:
        return 1.0
    return math.log(goal_diff + 1) * (2.2 / ((abs(elo_diff_pre_match) * 0.001) + 2.2))


def draw_probability(effective_diff):
    base = 0.28
    penalty = min(0.20, (abs(effective_diff) / 400) ** 1.5 * 0.18)
    return max(0.06, base - penalty)


def predict_probs(effective_diff):
    expected_home = 1 / (1 + 10 ** (-effective_diff / 400))
    p_draw = draw_probability(effective_diff)
    p_home = max(0.0, expected_home - p_draw / 2)
    p_away = max(0.0, (1 - expected_home) - p_draw / 2)
    total = p_home + p_draw + p_away
    return p_home / total, p_draw / total, p_away / total


class LaLigaModel:
    def __init__(self, base_dir=None, db_path=None):
        base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
        self.history_path = os.path.join(base_dir, HISTORY_FILE)
        self.mapping_path = os.path.join(base_dir, MAPPING_FILE)
        self.db_path = db_path
        self.db_finished_count = 0  # сколько сыгранных матчей из БД учтено в Elo

        self.org_to_couk = self._load_mapping()
        self.elo = self._build_current_elo()
        if db_path:
            self._apply_db_matches(db_path)

    # ---------- статический помощник для проверки свежести ----------

    @staticmethod
    def count_finished_pd(db_path):
        """Число завершённых матчей Ла Лиги в football.db - используется
        ботом, чтобы понять, пора ли пересобрать модель."""
        try:
            conn = sqlite3.connect(db_path)
            row = conn.execute("""
                SELECT COUNT(*) FROM matches
                WHERE competition_code = 'PD' AND status = 'FINISHED'
                  AND regular_home IS NOT NULL
            """).fetchone()
            conn.close()
            return row[0] if row else 0
        except sqlite3.Error:
            return 0

    # ---------- загрузка ----------

    def _load_mapping(self):
        """org (football-data.org) -> couk (football-data.co.uk) имя."""
        mapping = {}
        with open(self.mapping_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                org_name = r.get("football_data_org", "").strip()
                couk_name = r.get("football_data_co_uk", "").strip()
                if org_name and couk_name and "ИЛИ" not in org_name:
                    mapping[org_name] = couk_name
        return mapping

    def _load_history(self):
        rows = []
        with open(self.history_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if not r.get("HomeTeam") or not r.get("FTHG"):
                    continue
                try:
                    date = datetime.strptime(r["Date"], "%d/%m/%Y")
                except ValueError:
                    try:
                        date = datetime.strptime(r["Date"], "%d/%m/%y")
                    except ValueError:
                        continue
                try:
                    hg, ag = int(r["FTHG"]), int(r["FTAG"])
                except (ValueError, TypeError):
                    continue
                rows.append({
                    "date": date, "home": r["HomeTeam"].strip(),
                    "away": r["AwayTeam"].strip(), "hg": hg, "ag": ag,
                    "season": r.get("SeasonCode", ""),
                })
        rows.sort(key=lambda x: x["date"])
        return rows

    # ---------- построение Elo ----------

    def _update_elo_one_match(self, home_couk, away_couk, hg, ag):
        """Одно Elo-обновление (та же логика, что в бэктесте)."""
        elo_home = self.elo.setdefault(home_couk, BASE_ELO)
        elo_away = self.elo.setdefault(away_couk, BASE_ELO)
        effective_diff = (elo_home + HOME_ADVANTAGE) - elo_away
        score_home = 1.0 if hg > ag else (0.0 if hg < ag else 0.5)
        exp_home = 1 / (1 + 10 ** (-effective_diff / 400))
        mult = mov_multiplier(hg - ag, effective_diff)
        k_eff = K_FACTOR * mult
        self.elo[home_couk] = elo_home + k_eff * (score_home - exp_home)
        self.elo[away_couk] = elo_away + k_eff * ((1 - score_home) - (1 - exp_home))

    def _apply_season_regression(self):
        league_avg = sum(self.elo.values()) / len(self.elo) if self.elo else BASE_ELO
        for team in self.elo:
            self.elo[team] = (1 - SEASON_REGRESSION) * self.elo[team] + SEASON_REGRESSION * league_avg

    def _build_current_elo(self):
        matches = self._load_history()
        self.elo = {}
        current_season = None

        for m in matches:
            if current_season is not None and m["season"] != current_season:
                self._apply_season_regression()
            current_season = m["season"]
            self._update_elo_one_match(m["home"], m["away"], m["hg"], m["ag"])

        # Регрессия на границе нового сезона (история CSV закончилась в мае,
        # новый сезон стартует в августе - составы поменялись за межсезонье)
        self._apply_season_regression()
        return self.elo

    def _apply_db_matches(self, db_path):
        """Обновляет Elo сыгранными матчами текущего сезона из football.db.
        Имена команд там в формате football-data.org - переводим через маппинг.
        Команды без маппинга (например, новички из Сегунды) добавляются в Elo
        под своим org-именем со стартовым средним рейтингом лиги - дальше их
        рейтинг обновляется как обычно по их реальным результатам."""
        try:
            conn = sqlite3.connect(db_path)
            rows = conn.execute("""
                SELECT m.utc_date, t1.name, t2.name, m.regular_home, m.regular_away, m.season
                FROM matches m
                JOIN teams t1 ON m.home_team_id = t1.id
                JOIN teams t2 ON m.away_team_id = t2.id
                WHERE m.competition_code = 'PD' AND m.status = 'FINISHED'
                  AND m.regular_home IS NOT NULL
                ORDER BY m.utc_date ASC
            """).fetchall()
            conn.close()
        except sqlite3.Error:
            return

        league_avg = sum(self.elo.values()) / len(self.elo) if self.elo else BASE_ELO
        current_season = None

        for utc_date, home_org, away_org, hg, ag, season in rows:
            # регрессия на границе сезонов внутри данных БД (пригодится,
            # когда в базе накопится больше одного сезона Ла Лиги)
            if current_season is not None and season != current_season:
                self._apply_season_regression()
            current_season = season

            home_couk = self.org_to_couk.get(home_org)
            away_couk = self.org_to_couk.get(away_org)

            # Команда без маппинга (новичок) - ведём её под org-именем,
            # стартуя со среднего по лиге
            if home_couk is None:
                home_couk = home_org
                self.elo.setdefault(home_couk, league_avg)
            if away_couk is None:
                away_couk = away_org
                self.elo.setdefault(away_couk, league_avg)

            self._update_elo_one_match(home_couk, away_couk, hg, ag)
            self.db_finished_count += 1

    # ---------- прогноз ----------

    def get_elo_couk_name(self, org_name):
        """Переводит имя из football-data.org в ключ Elo-словаря."""
        couk_name = self.org_to_couk.get(org_name)
        if couk_name is not None and couk_name in self.elo:
            return couk_name, True
        # новички ведутся под org-именем после _apply_db_matches
        if org_name in self.elo:
            return org_name, True
        return None, False

    def predict(self, home_org_name, away_org_name):
        """Прогноз по именам команд из football-data.org."""
        league_avg = sum(self.elo.values()) / len(self.elo) if self.elo else BASE_ELO

        home_key, home_known = self.get_elo_couk_name(home_org_name)
        away_key, away_known = self.get_elo_couk_name(away_org_name)

        elo_home = self.elo[home_key] if home_known else league_avg
        elo_away = self.elo[away_key] if away_known else league_avg

        effective_diff = (elo_home + HOME_ADVANTAGE) - elo_away
        p_home, p_draw, p_away = predict_probs(effective_diff)

        return {
            "p_home": p_home, "p_draw": p_draw, "p_away": p_away,
            "elo_home": elo_home, "elo_away": elo_away,
            "home_known": home_known, "away_known": away_known,
        }
