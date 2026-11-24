"""
fetch_bbref.py – NBA advanced stats per season from Basketball-Reference
(BPM, OBPM, DBPM, VORP, WS, WS/48, PER), linked to the NBA player ids.

Source : https://www.basketball-reference.com/leagues/NBA_{YYYY}_advanced.html
         (YYYY = end year of the season: 2019 = 2018-19).  One page per season;
         Basketball-Reference allows ~20 requests / minute, we wait DELAY seconds.

Traded players have one row per team plus a combined row ("2TM", "3TM"…, "TOT" before 2024):
all rows are stored; the season total is the row with the most minutes.

Usage
-----
  python main.py --step bbref [--seasons 2014-15,...]
"""

import logging
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

from config import DB_PATH, REQUEST_TIMEOUT
from database import get_connection, is_done, set_checkpoint, upsert_df
from fetch_college import norm_name

logger = logging.getLogger(__name__)

URL = "https://www.basketball-reference.com/leagues/NBA_{year}_advanced.html"
DELAY = 4.0
NUMERIC = {"age": "age", "games": "g", "mp": "mp", "per": "per", "ws": "ws",
           "ws_per_48": "ws_per_48", "obpm": "obpm", "dbpm": "dbpm", "bpm": "bpm", "vorp": "vorp"}


def _parse(html: str, season: str) -> pd.DataFrame:
    table = BeautifulSoup(html, "lxml").find("table", id="advanced")
    if table is None:
        raise ValueError("table #advanced not found")
    rows = []
    for tr in table.tbody.find_all("tr"):
        name = tr.find(attrs={"data-stat": "name_display"})
        if name is None or not name.get("data-append-csv"):
            continue                                    # header / separator rows
        cell = lambda s: (tr.find(attrs={"data-stat": s}) or {}).get_text(strip=True) if tr.find(attrs={"data-stat": s}) else None
        row = {"bbref_id": name["data-append-csv"], "season": season,
               "player_name": name.get_text(strip=True).rstrip("*"),
               "team": cell("team_name_abbr"), "pos": cell("pos")}
        for stat, col in NUMERIC.items():
            row[col] = cell(stat)
        rows.append(row)
    df = pd.DataFrame(rows)
    for col in NUMERIC.values():
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _match_ids(df: pd.DataFrame, conn) -> pd.Series:
    """NBA player_id by (normalised name, season) among the players of that season."""
    nba = pd.read_sql_query(
        "SELECT DISTINCT t.player_id, t.season, p.full_name FROM player_stats_traditional t "
        "JOIN players p USING (player_id)", conn)
    nba["key"] = nba["full_name"].map(norm_name)
    nba = nba.drop_duplicates(["key", "season"], keep=False)       # ambiguous names → no match
    keys = df.assign(key=df["player_name"].map(norm_name))[["key", "season"]]
    return keys.merge(nba[["key", "season", "player_id"]], on=["key", "season"], how="left")["player_id"].values


def fetch_bbref_advanced(seasons: list[str], db_path: str = DB_PATH) -> int:
    total = 0
    for season in seasons:
        task = f"bbref:{season}"
        if is_done(task, db_path) and season != max(seasons):
            continue
        year = int(season[:4]) + 1
        try:
            r = requests.get(URL.format(year=year), timeout=REQUEST_TIMEOUT * 2,
                             headers={"User-Agent": "nba-draft-analytics (research script)"})
            r.raise_for_status()
            df = _parse(r.content.decode("utf-8"), season)
        except Exception as e:                       # noqa: BLE001
            logger.error(f"[bbref] {season}: {type(e).__name__}: {e}")
            set_checkpoint(task, "failed", str(e)[:200], db_path)
            time.sleep(DELAY)
            continue
        with get_connection(db_path) as conn:
            df["player_id"] = _match_ids(df, conn)
            n = upsert_df(df, "bbref_advanced", conn)
        matched = df.drop_duplicates("bbref_id")["player_id"].notna().mean()
        set_checkpoint(task, "done", f"{n} rows, {matched:.0%} matched", db_path)
        logger.info(f"[bbref] {season}: {n} rows, {matched:.1%} of the players matched to an NBA id")
        total += n
        time.sleep(DELAY)
    return total


def run(seasons: list[str], db_path: str = DB_PATH) -> None:
    fetch_bbref_advanced(seasons, db_path)
