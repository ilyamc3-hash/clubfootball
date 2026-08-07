"""
Диагностика: смотрим, что реально присылает API для проблемных матчей —
чтобы понять, баг ли это в нашем коде, или сам football-data.org
ещё не назначил команды.
"""
import os
import requests
import json

from dotenv import load_dotenv
load_dotenv()

API_TOKEN = os.environ["FOOTBALL_DATA_TOKEN"]
BASE_URL = "https://api.football-data.org/v4"
HEADERS = {"X-Auth-Token": API_TOKEN}

TARGET_IDS = [537388, 537389, 537390]

resp = requests.get(f"{BASE_URL}/competitions/WC/matches", headers=HEADERS, timeout=15)
resp.raise_for_status()
matches = resp.json().get("matches", [])

for m in matches:
    if m["id"] in TARGET_IDS:
        print(f"--- match id={m['id']} ---")
        print(f"stage: {m.get('stage')}")
        print(f"status: {m.get('status')}")
        print(f"utcDate: {m.get('utcDate')}")
        print(f"homeTeam: {m.get('homeTeam')}")
        print(f"awayTeam: {m.get('awayTeam')}")
        print()
