"""
Проверка: сколько сезонов Ла Лиги реально отдаёт бесплатный тариф football-data.org.
Это riskiest assumption плана по клубному футболу — проверяем до начала работ.
"""
import os
import requests

from dotenv import load_dotenv
load_dotenv()

API_TOKEN = os.environ["FOOTBALL_DATA_TOKEN"]
HEADERS = {"X-Auth-Token": API_TOKEN}
BASE = "https://api.football-data.org/v4"

# 1. Метаданные: какие сезоны вообще существуют в системе
r = requests.get(f"{BASE}/competitions/PD", headers=HEADERS, timeout=15)
print(f"Метаданные PD: HTTP {r.status_code}")
if r.status_code == 200:
    seasons = r.json().get("seasons", [])
    print(f"Сезонов в метаданных: {len(seasons)}")
    for s in seasons[:6]:
        print(f"  {s.get('startDate','?')[:4]}/{s.get('endDate','?')[:4]}")

print()

# 2. Реальный доступ: пробуем запросить матчи за конкретные прошлые сезоны
for year in [2025, 2024, 2023, 2022, 2021]:
    r = requests.get(
        f"{BASE}/competitions/PD/matches",
        headers=HEADERS,
        params={"season": year},
        timeout=15,
    )
    if r.status_code == 200:
        n = len(r.json().get("matches", []))
        print(f"Сезон {year}/{year+1}: ДОСТУПЕН, матчей: {n}")
    else:
        msg = ""
        try:
            msg = r.json().get("message", "")[:80]
        except Exception:
            pass
        print(f"Сезон {year}/{year+1}: HTTP {r.status_code} {msg}")
