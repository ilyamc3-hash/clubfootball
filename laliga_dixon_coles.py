"""
Football Prediction Lab -> La Liga, Фаза 2: Dixon-Coles модель на РЕАЛЬНЫХ
голах (не xG - та тема закрыта тремя независимыми отрицательными тестами).

Архитектурное отличие от всего, что тестировалось раньше: вместо одного
Elo-рейтинга на команду - РАЗДЕЛЬНЫЕ параметры атаки и защиты для каждой
команды, подогнанные через maximum likelihood (Dixon & Coles, 1997), плюс
поправка (tau) на систематически заниженную вероятность низкоскоринговых
результатов (0:0, 1:0, 0:1, 1:1) у простого произведения Пуассонов.

  log(lambda_home) = mu + home_adv + attack[home] - defense[away]
  log(lambda_away) = mu +            attack[away] - defense[home]

Для идентифицируемости (иначе attack и mu путаются) первая по алфавиту
команда в датасете фиксируется как референс: attack=defense=0.

МЕТОДОЛОГИЧЕСКОЕ ДОПУЩЕНИЕ (не было явно подтверждено пользователем -
дефолт с обоснованием, легко поменять через константу REFIT_INTERVAL_DAYS):
рефит модели происходит не на каждый матч (было бы дорого - тысячи MLE-
оптимизаций) и не раз в сезон (Elo так делает, но для DC это означало бы
"забывать" о 9 месяцах результатов), а каждые REFIT_INTERVAL_DAYS дней на
РАСШИРЯЮЩЕМСЯ окне всей прошлой истории с экспоненциальным затуханием по
давности матча (time decay, стандартная практика Dixon-Coles) - более
свежие матчи получают больший вес в функции правдоподобия.

Скорость затухания (xi) - гиперпараметр, выбирается ТОЛЬКО по train
(перебор нескольких кандидатов), как и everywhere else в проекте.

Оценивается ЧЕСТНО против рынка сразу по двум рынкам:
  - П1/Х/П2 (сравнение с текущей Elo-моделью: 0.1944, рынок: 0.1902)
  - Over/Under 2.5 (сравнение с грубой Elo-калибровкой тоталов: 0.2461,
    рынок: 0.2374 - см. laliga_totals.py)

ПРЕДУПРЕЖДЕНИЕ: скрипт делает десятки MLE-оптимизаций по ~60 параметрам
каждая (для ~30 команд за 12 сезонов) - полный прогон с перебором xi может
занять от нескольких минут до ~получаса в зависимости от машины. Прогресс
рефитов печатается в консоль, чтобы было видно, что скрипт не завис.

Требует: numpy, scipy (pip install numpy scipy --user, если ещё не стоят)

Использование:
    py laliga_dixon_coles.py
"""

import csv
import math
from datetime import datetime
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize

HISTORY_FILE = "laliga_matches_combined_with_ou.csv"

TRAIN_SEASONS = {"1415", "1516", "1617", "1718", "1819", "1920", "2021", "2122"}
TEST_SEASONS = {"2223", "2324", "2425", "2526"}

# --- допущения, требующие явного обоснования ---
REFIT_INTERVAL_DAYS = 21          # см. docstring выше
MIN_MATCHES_FOR_FIRST_FIT = 150   # ~половина сезона - до этого прогноз "средний по лиге"
XI_CANDIDATES = [0.0, 0.0015, 0.003]  # 0.0 = без затухания (весь train равнозначен)
MAX_GOALS = 8
LEAGUE_AVG_GOALS = 1.35
TOTALS_LINE = 2.5

PARAM_BOUNDS_ATTACK_DEFENSE = (-3.0, 3.0)
PARAM_BOUNDS_HOME_ADV = (-1.0, 1.0)
PARAM_BOUNDS_RHO = (-0.3, 0.3)
PARAM_BOUNDS_MU = (-2.0, 2.0)


# ---------- загрузка ----------

def find_ou_column_pairs(fieldnames):
    """Находит любые пары колонок вида '<букмекер>>2.5' / '<букмекер><2.5' -
    не завязано на конкретные названия, чтобы пережить любую конвенцию
    именования, которую использовал скрипт слияния (Avg, BbAv, B365, P...)."""
    pairs = []
    for col in fieldnames:
        if ">2.5" in col:
            prefix = col.split(">2.5")[0]
            under_col = prefix + "<2.5"
            if under_col in fieldnames:
                pairs.append((col, under_col))
    return pairs


def load_matches(path):
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        ou_pairs = find_ou_column_pairs(fieldnames)
        rows_raw = list(reader)

    if ou_pairs:
        print(f"Найдены пары колонок O/U: {ou_pairs}")
    else:
        print("!! Колонки O/U не найдены - оценка тоталов против рынка будет пропущена.")

    rows = []
    for r in rows_raw:
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

        avg_h, avg_d, avg_a = r.get("AvgH"), r.get("AvgD"), r.get("AvgA")

        over_odd = under_odd = None
        for over_col, under_col in ou_pairs:
            if r.get(over_col) and r.get(under_col):
                try:
                    over_odd = float(r[over_col])
                    under_odd = float(r[under_col])
                    break
                except (ValueError, TypeError):
                    continue

        rows.append({
            "date": date, "home": r["HomeTeam"].strip(), "away": r["AwayTeam"].strip(),
            "hg": hg, "ag": ag, "season": r.get("SeasonCode", ""),
            "avg_h": float(avg_h) if avg_h else None,
            "avg_d": float(avg_d) if avg_d else None,
            "avg_a": float(avg_a) if avg_a else None,
            "over_odd": over_odd, "under_odd": under_odd,
        })
    rows.sort(key=lambda x: x["date"])
    return rows


def odds_to_probs_3way(oh, od, oa):
    if not (oh and od and oa):
        return None
    raw_h, raw_d, raw_a = 1 / oh, 1 / od, 1 / oa
    total = raw_h + raw_d + raw_a
    return raw_h / total, raw_d / total, raw_a / total


def odds_to_probs_2way(o1, o2):
    if not (o1 and o2):
        return None
    raw1, raw2 = 1 / o1, 1 / o2
    total = raw1 + raw2
    return raw1 / total, raw2 / total


# ---------- Dixon-Coles: правдоподобие и фит ----------

def tau_correction(hg, ag, lam_home, lam_away, rho):
    """Поправка Dixon-Coles на низкоскоринговые результаты. Возвращает
    множитель к произведению двух независимых Пуассонов."""
    if hg == 0 and ag == 0:
        return 1 - lam_home * lam_away * rho
    elif hg == 0 and ag == 1:
        return 1 + lam_home * rho
    elif hg == 1 and ag == 0:
        return 1 + lam_away * rho
    elif hg == 1 and ag == 1:
        return 1 - rho
    return 1.0


def fit_dixon_coles(matches_subset, as_of_date, xi, team_index, n_teams):
    """MLE-фит параметров атаки/защиты на подвыборке матчей (строго до
    as_of_date), с весом exp(-xi * дней_с_матча) - более свежие матчи
    получают больший вес. Первая команда (team_index[...]==0) - референс,
    attack=defense=0, не оптимизируется."""
    n = len(matches_subset)
    home_idx = np.array([team_index[m["home"]] for m in matches_subset])
    away_idx = np.array([team_index[m["away"]] for m in matches_subset])
    hg = np.array([m["hg"] for m in matches_subset], dtype=float)
    ag = np.array([m["ag"] for m in matches_subset], dtype=float)
    days_ago = np.array([(as_of_date - m["date"]).days for m in matches_subset], dtype=float)
    weights = np.exp(-xi * days_ago)

    n_free = n_teams - 1  # первая команда - референс

    def unpack(theta):
        mu = theta[0]
        home_adv = theta[1]
        rho = theta[2]
        attack_free = theta[3:3 + n_free]
        defense_free = theta[3 + n_free:3 + 2 * n_free]
        attack = np.concatenate(([0.0], attack_free))
        defense = np.concatenate(([0.0], defense_free))
        return mu, home_adv, rho, attack, defense

    def neg_log_likelihood(theta):
        mu, home_adv, rho, attack, defense = unpack(theta)
        log_lam_home = mu + home_adv + attack[home_idx] - defense[away_idx]
        log_lam_away = mu + attack[away_idx] - defense[home_idx]
        log_lam_home = np.clip(log_lam_home, -5, 3)
        log_lam_away = np.clip(log_lam_away, -5, 3)
        lam_home = np.exp(log_lam_home)
        lam_away = np.exp(log_lam_away)

        # log Poisson pmf: k*log(lam) - lam - lgamma(k+1)
        log_p_home = hg * np.log(lam_home) - lam_home - np.array([math.lgamma(k + 1) for k in hg])
        log_p_away = ag * np.log(lam_away) - lam_away - np.array([math.lgamma(k + 1) for k in ag])
        log_p = log_p_home + log_p_away

        tau = np.array([
            tau_correction(int(h), int(a), lh, la, rho)
            for h, a, lh, la in zip(hg, ag, lam_home, lam_away)
        ])
        tau = np.clip(tau, 1e-6, None)  # защита от log(0) при экстремальном rho
        log_p_total = log_p + np.log(tau)

        return -np.sum(weights * log_p_total)

    x0 = np.zeros(3 + 2 * n_free)
    x0[0] = math.log(LEAGUE_AVG_GOALS)  # разумный старт для mu
    x0[1] = 0.1   # разумный старт для home_adv
    x0[2] = -0.05  # типичное значение rho в литературе

    bounds = (
        [PARAM_BOUNDS_MU, PARAM_BOUNDS_HOME_ADV, PARAM_BOUNDS_RHO]
        + [PARAM_BOUNDS_ATTACK_DEFENSE] * (2 * n_free)
    )

    result = minimize(neg_log_likelihood, x0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 300})

    mu, home_adv, rho, attack, defense = unpack(result.x)
    return {"mu": mu, "home_adv": home_adv, "rho": rho, "attack": attack, "defense": defense}


def predict_lambdas(params, home_key, away_key, team_index):
    if params is None or home_key not in team_index or away_key not in team_index:
        return LEAGUE_AVG_GOALS, LEAGUE_AVG_GOALS
    hi, ai = team_index[home_key], team_index[away_key]
    log_lam_home = params["mu"] + params["home_adv"] + params["attack"][hi] - params["defense"][ai]
    log_lam_away = params["mu"] + params["attack"][ai] - params["defense"][hi]
    return math.exp(max(-5, min(3, log_lam_home))), math.exp(max(-5, min(3, log_lam_away)))


def match_probs(lam_home, lam_away, rho, max_goals=MAX_GOALS):
    """Полная сетка вероятностей h x a с поправкой tau - используется и для
    П1/Х/П2, и для Over/Under (суммированием диагоналей/областей сетки)."""
    grid = np.zeros((max_goals + 1, max_goals + 1))
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p_h = (lam_home ** h) * math.exp(-lam_home) / math.factorial(h)
            p_a = (lam_away ** a) * math.exp(-lam_away) / math.factorial(a)
            grid[h, a] = p_h * p_a * tau_correction(h, a, lam_home, lam_away, rho)
    grid = np.clip(grid, 0, None)
    grid /= grid.sum()
    return grid


def probs_from_grid(grid):
    p_home = np.tril(grid, -1).sum()
    p_draw = np.trace(grid)
    p_away = np.triu(grid, 1).sum()
    return p_home, p_draw, p_away


def over_under_from_grid(grid, line=TOTALS_LINE):
    max_goals = grid.shape[0] - 1
    p_over = 0.0
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            if h + a > line:
                p_over += grid[h, a]
    return p_over, 1 - p_over


# ---------- walk-forward backtest ----------

def run_backtest(matches, xi):
    all_teams = sorted({m["home"] for m in matches} | {m["away"] for m in matches})
    team_index = {t: i for i, t in enumerate(all_teams)}
    n_teams = len(all_teams)
    print(f"  Команд в датасете: {n_teams} (референс: {all_teams[0]})")

    params = None
    last_fit_date = None
    n_refits = 0

    records = []  # (bucket, grid probs H/D/A, market H/D/A, over/under probs, market O/U, actual)

    for i, m in enumerate(matches):
        need_refit = (
            params is None and i >= MIN_MATCHES_FOR_FIRST_FIT
        ) or (
            last_fit_date is not None and (m["date"] - last_fit_date).days >= REFIT_INTERVAL_DAYS
        )
        if need_refit:
            history = matches[:i]
            params = fit_dixon_coles(history, m["date"], xi, team_index, n_teams)
            last_fit_date = m["date"]
            n_refits += 1
            if n_refits % 10 == 0:
                print(f"    ...рефит #{n_refits}, дата {m['date'].date()}, обработано {i}/{len(matches)} матчей")

        lam_home, lam_away = predict_lambdas(params, m["home"], m["away"], team_index)
        rho = params["rho"] if params else -0.05
        grid = match_probs(lam_home, lam_away, rho)
        p_home, p_draw, p_away = probs_from_grid(grid)
        p_over, p_under = over_under_from_grid(grid)

        bucket = "train" if m["season"] in TRAIN_SEASONS else ("test" if m["season"] in TEST_SEASONS else None)
        if bucket:
            actual_1x2 = "H" if m["hg"] > m["ag"] else ("D" if m["hg"] == m["ag"] else "A")
            actual_over = 1.0 if (m["hg"] + m["ag"]) > TOTALS_LINE else 0.0
            market_1x2 = odds_to_probs_3way(m["avg_h"], m["avg_d"], m["avg_a"]) if bucket == "test" else None
            market_ou = odds_to_probs_2way(m["over_odd"], m["under_odd"]) if bucket == "test" else None
            records.append({
                "bucket": bucket, "p_home": p_home, "p_draw": p_draw, "p_away": p_away,
                "actual_1x2": actual_1x2, "p_over": p_over, "p_under": p_under,
                "actual_over": actual_over, "market_1x2": market_1x2, "market_ou": market_ou,
            })

    print(f"  Всего рефитов за прогон: {n_refits}")
    return records


def brier_1x2(records, bucket):
    total, n, correct = 0.0, 0, 0
    for r in records:
        if r["bucket"] != bucket:
            continue
        probs = {"H": r["p_home"], "D": r["p_draw"], "A": r["p_away"]}
        for k in probs:
            t = 1.0 if k == r["actual_1x2"] else 0.0
            total += (probs[k] - t) ** 2
        if max(probs, key=probs.get) == r["actual_1x2"]:
            correct += 1
        n += 1
    return (total / n / 3, correct / n, n) if n else (None, None, 0)


def brier_market_1x2(records):
    total, n, correct = 0.0, 0, 0
    for r in records:
        if r["bucket"] != "test" or r["market_1x2"] is None:
            continue
        mh, md, ma = r["market_1x2"]
        probs = {"H": mh, "D": md, "A": ma}
        for k in probs:
            t = 1.0 if k == r["actual_1x2"] else 0.0
            total += (probs[k] - t) ** 2
        if max(probs, key=probs.get) == r["actual_1x2"]:
            correct += 1
        n += 1
    return (total / n / 3, correct / n, n) if n else (None, None, 0)


def brier_ou(records, bucket):
    total, n, correct = 0.0, 0, 0
    for r in records:
        if r["bucket"] != bucket:
            continue
        total += (r["p_over"] - r["actual_over"]) ** 2 + (r["p_under"] - (1 - r["actual_over"])) ** 2
        if (r["p_over"] >= 0.5) == (r["actual_over"] == 1.0):
            correct += 1
        n += 1
    return (total / n / 2, correct / n, n) if n else (None, None, 0)


def brier_market_ou(records):
    total, n, correct = 0.0, 0, 0
    for r in records:
        if r["bucket"] != "test" or r["market_ou"] is None:
            continue
        mo, mu = r["market_ou"]
        total += (mo - r["actual_over"]) ** 2 + (mu - (1 - r["actual_over"])) ** 2
        if (mo >= 0.5) == (r["actual_over"] == 1.0):
            correct += 1
        n += 1
    return (total / n / 2, correct / n, n) if n else (None, None, 0)


def main():
    matches = load_matches(HISTORY_FILE)
    print(f"Матчей: {len(matches)}\n")

    print("Калибровка time-decay (xi) - выбор ТОЛЬКО по train-Brier для П1/Х/П2:")
    best_xi, best_train_brier, best_records = None, None, None
    for xi in XI_CANDIDATES:
        print(f"\n--- xi={xi} ---")
        records = run_backtest(matches, xi)
        train_brier, train_acc, train_n = brier_1x2(records, "train")
        print(f"  Train Brier (П1/Х/П2): {train_brier:.4f} (accuracy {train_acc*100:.1f}%, n={train_n})")
        if best_train_brier is None or train_brier < best_train_brier:
            best_xi, best_train_brier, best_records = xi, train_brier, records

    print(f"\nЛучший xi: {best_xi} (train Brier {best_train_brier:.4f})")

    print("\n" + "=" * 60)
    print("ЧЕСТНАЯ оценка на test-окне (2022/23-2025/26):")
    print("=" * 60)

    test_brier_1x2, test_acc_1x2, n_1x2 = brier_1x2(best_records, "test")
    market_brier_1x2, market_acc_1x2, n_market_1x2 = brier_market_1x2(best_records)

    print(f"\nП1/Х/П2 ({n_1x2} матчей):")
    print(f"  Dixon-Coles: Brier {test_brier_1x2:.4f}, accuracy {test_acc_1x2*100:.1f}%")
    if n_market_1x2:
        print(f"  Рынок:       Brier {market_brier_1x2:.4f}, accuracy {market_acc_1x2*100:.1f}% (n={n_market_1x2})")
        gap = test_brier_1x2 - market_brier_1x2
        print(f"  Разрыв к рынку: {gap:+.4f}")
    print("  Для сравнения: текущая Elo-модель - Brier 0.1944, разрыв к рынку +0.0042")
    improvement_vs_elo = 0.1944 - test_brier_1x2
    print(f"  Улучшение относительно Elo: {improvement_vs_elo:+.4f}")

    test_brier_ou, test_acc_ou, n_ou = brier_ou(best_records, "test")
    market_brier_ou, market_acc_ou, n_market_ou = brier_market_ou(best_records)

    print(f"\nOver/Under 2.5 ({n_ou} матчей):")
    print(f"  Dixon-Coles: Brier {test_brier_ou:.4f}, accuracy {test_acc_ou*100:.1f}%")
    if n_market_ou:
        print(f"  Рынок:       Brier {market_brier_ou:.4f}, accuracy {market_acc_ou*100:.1f}% (n={n_market_ou})")
        gap_ou = test_brier_ou - market_brier_ou
        print(f"  Разрыв к рынку: {gap_ou:+.4f}")
    print("  Для сравнения: грубая Elo-калибровка тоталов - Brier 0.2461, разрыв к рынку +0.0087")
    improvement_vs_elo_ou = 0.2461 - test_brier_ou
    print(f"  Улучшение относительно Elo-калибровки: {improvement_vs_elo_ou:+.4f}")

    print("\n" + "=" * 60)
    if improvement_vs_elo >= 0.001:
        print("П1/Х/П2: КРИТЕРИЙ ВЫПОЛНЕН - Dixon-Coles лучше Elo. Стоит внедрять.")
    else:
        print("П1/Х/П2: критерий не выполнен (нужно >= 0.001).")
    if improvement_vs_elo_ou >= 0.001:
        print("Тоталы: КРИТЕРИЙ ВЫПОЛНЕН - Dixon-Coles лучше грубой Elo-калибровки.")
    else:
        print("Тоталы: критерий не выполнен (нужно >= 0.001).")


if __name__ == "__main__":
    main()
