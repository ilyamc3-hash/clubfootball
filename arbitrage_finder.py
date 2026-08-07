"""
Football Prediction Lab — поиск вилок (арбитражных ставок).

Идея: если взять ЛУЧШИЙ коэффициент на каждый исход (П1/Х/П2), возможно
от РАЗНЫХ букмекеров, и сумма (1/кэф) по всем трём исходам меньше 1 —
значит, поставив пропорционально на все исходы, можно гарантированно
выйти в плюс независимо от результата матча. Это и есть "вилка".

ВАЖНО, честно:
- Вилки на топовых событиях вроде плей-офф ЧМ встречаются РЕДКО и
  закрываются за секунды-минуты, как только один букмекер замечает
  расхождение и подстраивает линию.
- Большинство букмекеров ограничивают или блокируют аккаунты, замеченные
  в систематическом поиске вилок — это не запрещено законом в большинстве
  юрисдикций, но нарушает пользовательское соглашение букмекера.
- Комиссии, лимиты на ставку и задержки в реальном исполнении часто
  съедают теоретическую прибыль вилки — на бумаге она может выглядеть
  привлекательнее, чем в реальности.

Использование:
    py arbitrage_finder.py
"""

# ---- Впиши сюда коэффициенты по разным букмекерам для одного матча ----
# Формат: {букмекер: {"home": кэф, "draw": кэф, "away": кэф}}
# Не обязательно заполнять всех — пустые (None) просто не участвуют в поиске
# лучшего коэффициента, на результат остальных это не влияет.
BOOKMAKER_ODDS = {
    "Fonbet":       {"home": 3, "draw": 1.5, "away": 2.4},
    "WINLINE":      {"home": 4.20, "draw": 3.60, "away": 1.90},
    "Лига Ставок":  {"home": None, "draw": None, "away": None},
    "PARI":         {"home": None, "draw": None, "away": None},
    "BetBoom":      {"home": None, "draw": None, "away": None},
    "Мелбет":       {"home": None, "draw": None, "away": None},
    "Бетсити":      {"home": None, "draw": None, "away": None},
    "Марафон":      {"home": None, "draw": None, "away": None},
    "Балтбет":      {"home": None, "draw": None, "away": None},
    "Olimpbet":     {"home": None, "draw": None, "away": None},
}

TOTAL_STAKE = 10000  # условная сумма для примера расчёта распределения ставок


def find_best_odds(bookmaker_odds):
    """Для каждого исхода находим лучший коэффициент и у кого он был."""
    best = {}
    for outcome in ["home", "draw", "away"]:
        best_odds, best_book = 0, None
        for book, odds in bookmaker_odds.items():
            o = odds.get(outcome)
            if o and o > best_odds:
                best_odds, best_book = o, book
        best[outcome] = (best_odds, best_book)
    return best


def check_arbitrage(best):
    implied_sum = sum(1 / odds for odds, _ in best.values() if odds > 0)
    return implied_sum, implied_sum < 1.0


def stake_distribution(best, total_stake):
    implied_sum = sum(1 / odds for odds, _ in best.values() if odds > 0)
    stakes = {}
    for outcome, (odds, book) in best.items():
        stake = (1 / odds) / implied_sum * total_stake
        payout = stake * odds
        stakes[outcome] = (stake, book, odds, payout)
    return stakes


def main():
    best = find_best_odds(BOOKMAKER_ODDS)

    print("Лучшие коэффициенты по каждому исходу:")
    labels = {"home": "П1", "draw": "Х", "away": "П2"}
    for outcome, (odds, book) in best.items():
        print(f"  {labels[outcome]}: {odds}  ({book})")

    implied_sum, is_arbitrage = check_arbitrage(best)
    print(f"\nСумма обратных коэффициентов: {implied_sum*100:.2f}%")

    if is_arbitrage:
        margin = (1 - implied_sum) * 100
        print(f"✓ ВИЛКА НАЙДЕНА — теоретическая гарантированная маржа: {margin:.2f}%\n")

        stakes = stake_distribution(best, TOTAL_STAKE)
        print(f"Распределение ставки {TOTAL_STAKE} на каждый исход:")
        for outcome, (stake, book, odds, payout) in stakes.items():
            print(f"  {labels[outcome]} ({book}, кэф {odds}): поставить {stake:.0f}  → выплата при этом исходе {payout:.0f}")

        guaranteed_profit = min(payout for _, _, _, payout in stakes.values()) - TOTAL_STAKE
        print(f"\nГарантированная прибыль (минимум по всем исходам): {guaranteed_profit:.0f}")
    else:
        overround = (implied_sum - 1) * 100
        print(f"✗ Вилки нет — суммарная маржа рынка {overround:.2f}% (обычный оверраунд букмекеров)")

    print("\n⚠ Напоминание: вилки на топовых матчах закрываются очень быстро,")
    print("  а систематический поиск нарушает условия большинства букмекеров.")


if __name__ == "__main__":
    main()
