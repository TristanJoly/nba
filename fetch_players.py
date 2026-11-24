"""
fetch_players.py – Fetch player-level data and persist to SQLite.

Endpoints used
--------------
  CommonAllPlayers                  → players (list)
  CommonPlayerInfo                  → players (draft, school, physical info)
  LeagueDashPlayerStats (Base)      → player_stats_traditional
  LeagueDashPlayerStats (Advanced)  → player_stats_advanced
  LeagueDashPlayerStats (Scoring)   → player_stats_scoring
  LeagueHustleStatsPlayer           → player_stats_hustle   (available from 2015-16)
  LeagueDashPtStats                 → player_stats_tracking (one row per measure type)
  PlayerCareerStats                 → player_career_stats
"""

import json
import logging
from datetime import datetime

import pandas as pd

from config import (
    SEASONS, CURRENT_SEASON, LEAGUE_ID, TRACKING_MEASURES,
    PER_MODE_PER_GAME, SEASON_TYPE_REGULAR,
)
from database import get_connection, upsert_df, set_checkpoint, is_done, done_tasks
from utils import api_call, safe_df, nba_kwargs, progress

logger = logging.getLogger(__name__)


# ===========================================================================
# API WRAPPERS
# ===========================================================================

@api_call
def _all_players():
    from nba_api.stats.endpoints import CommonAllPlayers
    return CommonAllPlayers(is_only_current_season=0, league_id=LEAGUE_ID,
                            season=CURRENT_SEASON, **nba_kwargs())


@api_call
def _player_info(player_id: int):
    from nba_api.stats.endpoints import CommonPlayerInfo
    return CommonPlayerInfo(player_id=player_id, **nba_kwargs())


@api_call
def _league_player_stats(season: str, measure_type: str):
    from nba_api.stats.endpoints import LeagueDashPlayerStats
    return LeagueDashPlayerStats(
        season=season,
        season_type_all_star=SEASON_TYPE_REGULAR,
        per_mode_detailed=PER_MODE_PER_GAME,
        measure_type_detailed_defense=measure_type,
        league_id_nullable=LEAGUE_ID,
        **nba_kwargs(),
    )


@api_call
def _hustle(season: str):
    from nba_api.stats.endpoints import LeagueHustleStatsPlayer
    return LeagueHustleStatsPlayer(
        season=season,
        season_type_all_star=SEASON_TYPE_REGULAR,
        per_mode_time=PER_MODE_PER_GAME,
        league_id_nullable=LEAGUE_ID,
        **nba_kwargs(),
    )


@api_call
def _tracking(season: str, measure: str):
    from nba_api.stats.endpoints import LeagueDashPtStats
    return LeagueDashPtStats(
        season=season,
        season_type_all_star=SEASON_TYPE_REGULAR,
        per_mode_simple=PER_MODE_PER_GAME,
        player_or_team="Player",
        pt_measure_type=measure,
        league_id_nullable=LEAGUE_ID,
        **nba_kwargs(),
    )


@api_call
def _career(player_id: int):
    from nba_api.stats.endpoints import PlayerCareerStats
    return PlayerCareerStats(player_id=player_id, **nba_kwargs())


# ===========================================================================
# PUBLIC ORCHESTRATORS
# ===========================================================================

def fetch_all_players_list(db_path: str = "nba_draft_analytics.db") -> None:
    """
    Add every player who played in (or after) the first configured season
    to *players*. Basic fields only – see fetch_player_info_batch().
    Existing rows are left alone so their detailed fields are not wiped.
    """
    task = f"players_list:{CURRENT_SEASON}"
    if is_done(task, db_path):
        logger.info("[SKIP] Player list already fetched.")
        return

    logger.info("👤 Fetching player list…")
    df = safe_df(_all_players())
    if df.empty:
        set_checkpoint(task, "failed", db_path=db_path)
        logger.error("No player list returned.")
        return

    first_year = int(SEASONS[0][:4])
    df = df[pd.to_numeric(df["TO_YEAR"], errors="coerce") >= first_year].copy()

    names = df["DISPLAY_LAST_COMMA_FIRST"].fillna("").str.split(", ", n=1)
    df["last_name"]  = names.str[0]
    df["first_name"] = names.str[1]
    df = df.rename(columns={
        "PERSON_ID":          "player_id",
        "DISPLAY_FIRST_LAST": "full_name",
        "ROSTERSTATUS":       "is_active",
    })
    df["last_updated"] = datetime.now().isoformat(timespec="seconds")

    conn = get_connection(db_path)
    known = {r["player_id"] for r in conn.execute("SELECT player_id FROM players")}
    n = upsert_df(df[~df["player_id"].isin(known)], "players", conn)
    conn.close()
    set_checkpoint(task, "done", db_path=db_path)
    logger.info(f"✅ {n} new players stored (active since {SEASONS[0]}).")


def fetch_player_info_batch(player_ids: list[int],
                            db_path: str = "nba_draft_analytics.db") -> None:
    """Enrich *players* rows with CommonPlayerInfo (one API call per player)."""
    done = done_tasks("player_info:", db_path)
    todo = [p for p in player_ids if f"player_info:{p}" not in done]
    logger.info(f"   {len(player_ids) - len(todo)} already done, {len(todo)} to fetch.")

    conn = get_connection(db_path)
    ok = failed = 0
    with progress(todo, desc="Player info", unit="player") as bar:
        for pid in bar:
            task = f"player_info:{pid}"
            result = _player_info(pid)
            df = result.common_player_info.get_data_frame() if result else None
            if df is None or df.empty:
                failed += 1
                set_checkpoint(task, "failed", db_path=db_path)
            else:
                r = df.iloc[0]

                def _int(v):
                    try:
                        return int(v)
                    except (TypeError, ValueError):
                        return None   # "Undrafted", "", None…

                conn.execute(
                    """UPDATE players SET first_name=?, last_name=?, full_name=?, team_id=?,
                       team_name=?, team_abbreviation=?, jersey=?, position=?, height=?,
                       weight=?, birthdate=?, country=?, school=?, draft_year=?,
                       draft_round=?, draft_number=?, years_experience=?, last_updated=?
                       WHERE player_id=?""",
                    (r["FIRST_NAME"], r["LAST_NAME"], r["DISPLAY_FIRST_LAST"], _int(r["TEAM_ID"]),
                     r["TEAM_NAME"], r["TEAM_ABBREVIATION"], r["JERSEY"], r["POSITION"],
                     r["HEIGHT"], r["WEIGHT"], str(r["BIRTHDATE"])[:10], r["COUNTRY"],
                     r["SCHOOL"], _int(r["DRAFT_YEAR"]), _int(r["DRAFT_ROUND"]),
                     _int(r["DRAFT_NUMBER"]), _int(r["SEASON_EXP"]),
                     datetime.now().isoformat(timespec="seconds"), pid),
                )
                conn.commit()
                ok += 1
                set_checkpoint(task, "done", db_path=db_path)
            bar.set_postfix(ok=ok, failed=failed)

    conn.close()
    logger.info(f"✅ Player info: {ok} updated, {failed} failed.")


_SEASON_STATS = [
    ("Base",     "player_stats_traditional"),
    ("Advanced", "player_stats_advanced"),
    ("Scoring",  "player_stats_scoring"),
]


def fetch_all_season_stats(seasons: list[str] = SEASONS,
                           db_path: str = "nba_draft_analytics.db") -> None:
    """Traditional + Advanced + Scoring stats for every player, one call per season × measure."""
    jobs = [(s, m, t) for s in seasons for m, t in _SEASON_STATS]
    conn = get_connection(db_path)
    rows = 0
    with progress(jobs, desc="Season stats", unit="call") as bar:
        for season, measure, table in bar:
            bar.set_postfix(season=season, measure=measure, rows=rows)
            task = f"{table}:{season}"
            if is_done(task, db_path):
                continue
            df = safe_df(_league_player_stats(season, measure))
            if df.empty:
                set_checkpoint(task, "failed", db_path=db_path)
                continue
            df["season"] = season
            rows += upsert_df(df, table, conn)
            set_checkpoint(task, "done", db_path=db_path)
            bar.set_postfix(season=season, measure=measure, rows=rows)
    conn.close()
    logger.info(f"✅ Season stats complete ({rows:,} rows).")


def fetch_hustle_stats(season: str, db_path: str = "nba_draft_analytics.db") -> int:
    """Hustle stats for one season (the NBA only tracks them since 2015-16)."""
    task = f"hustle:{season}"
    if is_done(task, db_path):
        return 0
    df = safe_df(_hustle(season))
    if df.empty:
        logger.warning(f"   hustle {season}: no data")
        set_checkpoint(task, "failed", db_path=db_path)
        return 0
    df = df.rename(columns={
        "G": "gp",
        "OFF_LOOSE_BALLS_RECOVERED": "off_loose_balls_rec",
        "DEF_LOOSE_BALLS_RECOVERED": "def_loose_balls_rec",
        "LOOSE_BALLS_RECOVERED":     "loose_balls_rec",
    })
    df["season"] = season
    conn = get_connection(db_path)
    n = upsert_df(df, "player_stats_hustle", conn)
    conn.close()
    set_checkpoint(task, "done", db_path=db_path)
    logger.debug(f"   hustle {season}: {n} rows")
    return n


def fetch_tracking_stats(season: str, db_path: str = "nba_draft_analytics.db") -> int:
    """Player tracking stats for one season: one call per measure in TRACKING_MEASURES."""
    conn = get_connection(db_path)
    rows = 0
    with progress(TRACKING_MEASURES, desc=f"  Tracking {season}", unit="call", leave=False) as bar:
        for measure in bar:
            task = f"tracking:{season}:{measure}"
            if is_done(task, db_path):
                continue
            df = safe_df(_tracking(season, measure))
            if df.empty:
                set_checkpoint(task, "failed", db_path=db_path)
                continue
            records = df.to_dict(orient="records")
            out = [{
                "player_id":    rec.get("PLAYER_ID"),
                "season":       season,
                "team_id":      rec.get("TEAM_ID"),
                "measure_type": measure,
                "data_json":    json.dumps(rec, default=str),
            } for rec in records]
            rows += upsert_df(pd.DataFrame(out), "player_stats_tracking", conn)
            set_checkpoint(task, "done", db_path=db_path)
            bar.set_postfix(rows=rows)
    conn.close()
    return rows


def fetch_player_career_batch(player_ids: list[int],
                              db_path: str = "nba_draft_analytics.db") -> None:
    """Regular-season totals by season for each player (one call per player)."""
    done = done_tasks("career:", db_path)
    todo = [p for p in player_ids if f"career:{p}" not in done]
    logger.info(f"   {len(player_ids) - len(todo)} already done, {len(todo)} to fetch.")

    conn = get_connection(db_path)
    rows = failed = 0
    with progress(todo, desc="Career stats", unit="player") as bar:
        for pid in bar:
            task = f"career:{pid}"
            result = _career(pid)
            if result is None:
                failed += 1
                set_checkpoint(task, "failed", db_path=db_path)
            else:
                df = result.season_totals_regular_season.get_data_frame()
                rows += upsert_df(df, "player_career_stats", conn)
                set_checkpoint(task, "done", db_path=db_path)
            bar.set_postfix(rows=rows, failed=failed)
    conn.close()
    logger.info(f"✅ Career stats: {rows:,} rows, {failed} players failed.")
