"""
Football Prediction Lab — стартовые рейтинги.

Официальный рейтинг ФИФА на 11 июня 2026 (последнее обновление перед
стартом турнира). С 2018 года рейтинг ФИФА сам устроен по Elo-подобной
формуле — то есть это не просто "позиция в таблице", а уже готовая
оценка силы команды, накопленная за годы результатов. Логично
использовать её как стартовую точку вместо одинаковых 1500 для всех.

Источник: bombardir.ru/rating-fifa (по официальным данным FIFA/Coca-Cola
World Ranking, обновление от 11.06.2026).

Все остальные скрипты импортируют STARTING_ELO из этого файла:
    from fifa_ratings import STARTING_ELO
"""

STARTING_ELO = {
    "Argentina": 1877,
    "Spain": 1875,
    "France": 1871,
    "England": 1828,
    "Portugal": 1768,
    "Brazil": 1766,
    "Morocco": 1755,
    "Netherlands": 1754,
    "Belgium": 1742,
    "Germany": 1736,
    "Croatia": 1715,
    "Colombia": 1698,
    "Mexico": 1687,
    "Senegal": 1684,
    "Uruguay": 1673,
    "United States": 1671,
    "Japan": 1662,
    "Switzerland": 1650,
    "Iran": 1620,
    "Turkey": 1606,
    "Ecuador": 1599,
    "Austria": 1597,
    "South Korea": 1592,
    "Australia": 1579,
    "Algeria": 1571,
    "Egypt": 1562,
    "Canada": 1559,
    "Norway": 1557,
    "Ivory Coast": 1541,
    "Panama": 1539,
    "Sweden": 1510,
    "Czechia": 1506,
    "Paraguay": 1505,
    "Scotland": 1503,
    "Tunisia": 1476,
    "Congo DR": 1474,
    "South Africa": 1428,
    "Saudi Arabia": 1424,
    "Jordan": 1388,
    "Bosnia-Herzegovina": 1387,
    "Uzbekistan": 1459,
    "Iraq": 1446,
    "New Zealand": 1276,
    "Cape Verde Islands": 1371,
    "Curaçao": 1295,
    "Haiti": 1293,
    "Qatar": 1450,
    "Ghana": 1347,
}

BASE_ELO_FALLBACK = 1500  # на случай, если команды нет в списке (не должно происходить для ЧМ-2026)


def get_starting_elo(team_name):
    return STARTING_ELO.get(team_name, BASE_ELO_FALLBACK)
