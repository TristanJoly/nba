"""
fetch_college.py – NCAA (Division I) player-seasons from Barttorvik + link to the draft.

Source : https://barttorvik.com/getadvstats.php?year=YYYY&csv=1
         one CSV per college season (YYYY = spring year: 2018 = 2017-18), from 2008,
         no header (column positions below), includes the birth date and the NBA pick.

Tables
------
  college_stats  : one row per (barttorvik pid, season, team)
  college_teams  : team strength per season (Barttorvik team_results: Barthag, SOS…)
  draft_college  : one row per drafted player found in college
                   (NBA player_id ↔ barttorvik pid, birth date, last college season)

Why it matters for the model: the birth date and the college stats are known BEFORE the
draft for every college player, whether or not he later reaches the NBA – unlike the NBA
CommonPlayerInfo birth date, which only exists for players who played (target leak).

Usage
-----
  python main.py --step college
"""

import csv
import io
import logging
import re
import time
import unicodedata
import warnings

import pandas as pd
import requests

from config import DB_PATH, REQUEST_TIMEOUT
from database import get_connection, is_done, set_checkpoint, upsert_df

logger = logging.getLogger(__name__)

URL = "https://barttorvik.com/getadvstats.php?year={year}&csv=1"
FIRST_YEAR = 2008
DELAY = 2.0          # seconds between downloads (be polite)
N_COLS = 78

# position in the CSV → column name (positions checked on Trae Young / Deandre Ayton 2018)
COLUMNS = {
    0: "player_name", 1: "team", 2: "conf", 3: "gp", 4: "min_pct", 5: "ortg", 6: "usg",
    7: "efg", 8: "ts_pct", 9: "orb_pct", 10: "drb_pct", 11: "ast_pct", 12: "tov_pct",
    13: "ftm", 14: "fta", 15: "ft_pct", 16: "fg2m", 17: "fg2a", 18: "fg2_pct",
    19: "fg3m", 20: "fg3a", 21: "fg3_pct", 22: "blk_pct", 23: "stl_pct", 24: "ftr",
    25: "class_yr", 26: "height", 31: "year", 32: "pid", 34: "rec_rank", 45: "nba_pick",
    46: "drtg", 50: "bpm", 51: "obpm", 52: "dbpm", 54: "mpg", 57: "orebpg", 58: "drebpg",
    59: "rebpg", 60: "astpg", 61: "stlpg", 62: "blkpg", 63: "ptspg", 64: "role", 66: "dob",
}
TEXT = {"player_name", "team", "conf", "class_yr", "height", "role", "dob"}

# Conferences whose teams are, year after year, the strongest of Division I
POWER_CONFS = {"ACC", "B10", "B12", "BE", "P12", "SEC"}


# ===========================================================================
# DOWNLOAD
# ===========================================================================

def _parse(text: str) -> pd.DataFrame:
    rows = [r for r in csv.reader(io.StringIO(text)) if len(r) == N_COLS]
    df = pd.DataFrame([[r[i] for i in COLUMNS] for r in rows], columns=list(COLUMNS.values()))
    for c in df.columns.difference(list(TEXT)):
        df[c] = pd.to_numeric(df[c].replace({"": None, "NA": None}), errors="coerce")
    for c in TEXT:
        df[c] = df[c].replace({"": None, "NA": None})
    df["nba_pick"] = df["nba_pick"].where(df["nba_pick"] > 0)
    return df.dropna(subset=["pid", "year"])


def fetch_college_stats(last_year: int, db_path: str = DB_PATH, first_year: int = FIRST_YEAR) -> int:
    """Download every season first_year … last_year (checkpointed: done seasons are skipped,
    except the last one, which may still be in progress)."""
    total = 0
    for year in range(first_year, last_year + 1):
        task = f"college:{year}"
        if is_done(task, db_path) and year != last_year:
            continue
        try:
            r = requests.get(URL.format(year=year), timeout=REQUEST_TIMEOUT * 2,
                             headers={"User-Agent": "nba-draft-analytics (research script)"})
            r.raise_for_status()
            df = _parse(r.text)
            if df.empty:
                raise ValueError("empty CSV")
        except Exception as e:                       # noqa: BLE001
            logger.error(f"[college] {year}: {type(e).__name__}: {e}")
            set_checkpoint(task, "failed", str(e)[:200], db_path)
            continue
        with get_connection(db_path) as conn:
            n = upsert_df(df, "college_stats", conn)
        set_checkpoint(task, "done", f"{n} rows", db_path)
        logger.info(f"[college] {year}: {n} player-seasons")
        total += n
        time.sleep(DELAY)
    return total


TEAM_URL = "https://barttorvik.com/{year}_team_results.csv"
TEAM_COLUMNS = {"team": "team", "conf": "conf", "adjoe": "adjoe", "adjde": "adjde",
                "barthag": "barthag", "sos": "sos", "rank": "rank"}


def fetch_college_teams(last_year: int, db_path: str = DB_PATH, first_year: int = FIRST_YEAR) -> int:
    """Team strength per season (adjusted efficiencies, Barthag = win probability vs an
    average D-I team, strength of schedule).  Same team names as the player CSV."""
    total = 0
    for year in range(first_year, last_year + 1):
        task = f"college_teams:{year}"
        if is_done(task, db_path) and year != last_year:
            continue
        try:
            r = requests.get(TEAM_URL.format(year=year), timeout=REQUEST_TIMEOUT * 2,
                             headers={"User-Agent": "nba-draft-analytics (research script)"})
            r.raise_for_status()
            # older files: one more field per row than in the header ("Fun Rk, adjt") → index_col=False
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", pd.errors.ParserWarning)
                df = pd.read_csv(io.StringIO(r.text), index_col=False)
            df = df.loc[:, ~df.columns.duplicated()][list(TEAM_COLUMNS)].rename(columns=TEAM_COLUMNS)
            if df.empty:
                raise ValueError("empty CSV")
            df["year"] = year
        except Exception as e:                       # noqa: BLE001
            logger.error(f"[college_teams] {year}: {type(e).__name__}: {e}")
            set_checkpoint(task, "failed", str(e)[:200], db_path)
            continue
        with get_connection(db_path) as conn:
            n = upsert_df(df, "college_teams", conn)
        set_checkpoint(task, "done", f"{n} teams", db_path)
        logger.info(f"[college_teams] {year}: {n} teams")
        total += n
        time.sleep(DELAY)
    return total


# ===========================================================================
# MATCH WITH THE DRAFT
# ===========================================================================

def norm_name(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[.'’`-]", "", s).replace(",", " ")
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def match_draft(db_path: str = DB_PATH) -> pd.DataFrame:
    """
    Link each drafted player to his barttorvik pid.
      1. "pick"  : same normalised name + barttorvik's NBA pick == overall pick
      2. "name"  : same normalised name, college season within the 6 years before the draft,
                   and a single candidate pid
    Only seasons <= draft year are kept (college career before the draft).
    """
    conn = get_connection(db_path)
    draft = pd.read_sql_query(
        "SELECT player_id, player_name, CAST(season AS INT) AS draft_year, overall_pick, "
        "organization, organization_type FROM draft_history WHERE CAST(season AS INT) >= ?",
        conn, params=(FIRST_YEAR + 1,))
    col = pd.read_sql_query("SELECT pid, year, player_name, team, nba_pick, dob FROM college_stats", conn)

    draft["key"] = draft["player_name"].map(norm_name)
    col["key"] = col["player_name"].map(norm_name)
    cand = draft.merge(col.drop(columns="player_name"), on="key")
    cand = cand[(cand["year"] <= cand["draft_year"]) & (cand["year"] >= cand["draft_year"] - 5)]

    by_pick = cand[cand["nba_pick"] == cand["overall_pick"]].assign(method="pick")
    rest = cand[~cand["player_id"].isin(by_pick["player_id"])]
    n_pid = rest.groupby("player_id")["pid"].nunique()
    by_name = rest[rest["player_id"].isin(n_pid[n_pid == 1].index)].assign(method="name")

    # 3. nicknames (Bam / Edrice Adebayo, Mo / Mohamed Bamba…): same last name + same pick
    found = set(by_pick["player_id"]) | set(by_name["player_id"])
    left = draft[~draft["player_id"].isin(found)].assign(last=lambda d: d["key"].str.split().str[-1])
    col["last"] = col["key"].str.split().str[-1]
    nick = left.merge(col.drop(columns=["player_name", "key"]).dropna(subset=["nba_pick"]),
                      left_on=["last", "overall_pick"], right_on=["last", "nba_pick"])
    nick = nick[(nick["year"] <= nick["draft_year"]) & (nick["year"] >= nick["draft_year"] - 5)]
    n_pid = nick.groupby("player_id")["pid"].nunique()
    nick = nick[nick["player_id"].isin(n_pid[n_pid == 1].index)].assign(method="nickname")

    m = pd.concat([by_pick, by_name, nick])
    out = (m.sort_values("year").groupby("player_id")
             .agg(pid=("pid", "last"), last_college_year=("year", "max"),
                  first_college_year=("year", "min"), dob=("dob", "last"), method=("method", "first"))
             .reset_index())
    upsert_df(out, "draft_college", conn)
    conn.close()

    college = draft[draft["organization_type"] == "College/University"]
    found = college["player_id"].isin(out["player_id"]).mean()
    logger.info(f"✅ draft_college: {len(out)} drafted players linked to college "
                f"({found:.0%} of the college draftees since {FIRST_YEAR + 1}; "
                f"{(out['method'] == 'pick').sum()} by pick, {(out['method'] == 'name').sum()} by name)")
    return out


def run(last_year: int, db_path: str = DB_PATH) -> None:
    fetch_college_stats(last_year, db_path)
    fetch_college_teams(last_year, db_path)
    match_draft(db_path)
