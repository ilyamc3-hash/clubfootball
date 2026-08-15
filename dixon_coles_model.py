"""
Football Prediction Lab -> Club Football: ДЕШЁВАЯ сторона Dixon-Coles
для бота - только чтение уже посчитанных параметров и предсказание.

Дорогой MLE-фит (numpy/scipy) живёт отдельно в refit_dixon_coles.py
(запускается по расписанию, не в процессе бота) и в laliga_dixon_coles.py
(референсный бэктест). Этот модуль - только math/json/csv, чтобы не тащить
numpy/scipy в процесс бота ради операции, которая тут не нужна.

Формулы (tau_correction, сетка голов, MAX_GOALS=8) - точная копия
константы/логики из laliga_dixon_coles.py, честно провалидированной
бэктестом (Brier тоталов 0.2418 против рынка 0.2374, разрыв +0.0044 на
1520 матчах 2022/23-2025/26). Дублирование сознательное: ~15 строк чистой
математики, чтобы не импортировать scipy-зависимый модуль в бота.

Использование:
    from dixon_coles_model import DixonColesModel
    model = DixonColesModel()
    result = model.predict_totals("Real Madrid CF", "FC Barcelona")

Про rho (влияет на tau_correction, а значит на BTTS/точные счета из
predict_totals) - см. примечание в docstring refit_dixon_coles.py: на
первом прогоне (2026-08-15) rho сошёлся к небольшому ПОЛОЖИТЕЛЬНОМУ
значению (+0.0041), не к типичному отрицательному из литературы - это не
баг, но при интерпретации точных счетов/BTTS стоит держать в уме.
"""

import json
import math
import os

from laliga_model import load_team_mapping

MAPPING_FILE = "team_name_mapping.csv"
PARAMS_FILE = "dc_params.json"

MAX_GOALS = 8
LOG_LAMBDA_MIN = -5
LOG_LAMBDA_MAX = 3


def tau_correction(hg, ag, lam_home, lam_away, rho):
    """Поправка Dixon-Coles на низкоскоринговые результаты - идентична
    tau_correction() в laliga_dixon_coles.py."""
    if hg == 0 and ag == 0:
        return 1 - lam_home * lam_away * rho
    elif hg == 0 and ag == 1:
        return 1 + lam_home * rho
    elif hg == 1 and ag == 0:
        return 1 + lam_away * rho
    elif hg == 1 and ag == 1:
        return 1 - rho
    return 1.0


def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / math.factorial(k)


class DixonColesModel:
    def __init__(self, base_dir=None, params_path=None, mapping_path=None):
        base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
        self.params_path = params_path or os.path.join(base_dir, PARAMS_FILE)
        mapping_path = mapping_path or os.path.join(base_dir, MAPPING_FILE)

        with open(self.params_path, encoding="utf-8") as f:
            data = json.load(f)

        self.attack = data["attack"]
        self.defense = data["defense"]
        self.mu = data["mu"]
        self.home_adv = data["home_adv"]
        self.rho = data["rho"]
        self.league_avg_goals = data["league_avg_goals"]
        self.fit_date = data.get("fit_date")
        self.teams = set(self.attack.keys())

        self.org_to_couk = load_team_mapping(mapping_path)
        self.params_mtime = os.path.getmtime(self.params_path)

    # ---------- резолв имени команды ----------

    def _team_key(self, org_name):
        """org-имя (football-data.org) -> ключ в attack/defense. Тот же
        fallback-паттерн, что LaLigaModel.get_elo_couk_name: сперва couk-имя
        через маппинг, потом само org-имя (новички, заведённые под ним при
        рефите), иначе команда неизвестна модели."""
        couk_name = self.org_to_couk.get(org_name)
        if couk_name is not None and couk_name in self.teams:
            return couk_name
        if org_name in self.teams:
            return org_name
        return None

    # ---------- предсказание ----------

    def _lambdas(self, home_key, away_key):
        if home_key is None or away_key is None:
            return self.league_avg_goals, self.league_avg_goals
        log_lam_home = self.mu + self.home_adv + self.attack[home_key] - self.defense[away_key]
        log_lam_away = self.mu + self.attack[away_key] - self.defense[home_key]
        log_lam_home = max(LOG_LAMBDA_MIN, min(LOG_LAMBDA_MAX, log_lam_home))
        log_lam_away = max(LOG_LAMBDA_MIN, min(LOG_LAMBDA_MAX, log_lam_away))
        return math.exp(log_lam_home), math.exp(log_lam_away)

    def _grid(self, lam_home, lam_away):
        grid = {}
        total = 0.0
        for h in range(MAX_GOALS + 1):
            for a in range(MAX_GOALS + 1):
                p = poisson_pmf(h, lam_home) * poisson_pmf(a, lam_away)
                p *= tau_correction(h, a, lam_home, lam_away, self.rho)
                p = max(0.0, p)
                grid[(h, a)] = p
                total += p
        if total > 0:
            for k in grid:
                grid[k] /= total
        return grid

    def predict_totals(self, home_org_name, away_org_name, line=2.5):
        home_key = self._team_key(home_org_name)
        away_key = self._team_key(away_org_name)
        lambda_home, lambda_away = self._lambdas(home_key, away_key)

        grid = self._grid(lambda_home, lambda_away)
        p_over = sum(p for (h, a), p in grid.items() if h + a > line)
        p_under = 1 - p_over
        p_btts = sum(p for (h, a), p in grid.items() if h >= 1 and a >= 1)
        top_scores = sorted(grid.items(), key=lambda kv: kv[1], reverse=True)[:5]

        return {
            "p_over": p_over, "p_under": p_under,
            "lambda_home": lambda_home, "lambda_away": lambda_away,
            "p_btts": p_btts, "top_scores": top_scores,
            "home_known": home_key is not None, "away_known": away_key is not None,
        }
