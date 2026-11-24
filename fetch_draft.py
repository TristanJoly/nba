"""
fetch_draft.py – Fetch all draft-related data and persist to SQLite.

Endpoints used
--------------
  DraftCombinePlayerAnthro          → draft_combine_anthro
  DraftCombineDrillResults          → draft_combine_drills
  DraftCombineNonStationaryShooting → draft_combine_shooting (off-dribble / on-move columns)
  DraftCombineSpotShooting          → draft_combine_shooting (spot-up columns)
  DraftHistory                      → draft_history

Note: DraftCombineStats is *not* college averages – it returns the same
measurements as the four endpoints above, so it is not fetched.
"""

import logging

import pandas as pd

from config import DRAFT_COMBINE_SEASON, DRAFT_YEARS, LEAGUE_ID
from database import get_connection, upsert_df, set_checkpoint, is_done, done_tasks
from utils import api_call, safe_df, nba_kwargs, progress

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Individual fetchers (each wrapped with @api_call)
# ---------------------------------------------------------------------------

@api_call
def _fetch_anthro(season: str):
    from nba_api.stats.endpoints import DraftCombinePlayerAnthro
    return DraftCombinePlayerAnthro(season_year=season, league_id=LEAGUE_ID, **nba_kwargs())


@api_call
def _fetch_drills(season: str):
    from nba_api.stats.endpoints import DraftCombineDrillResults
    return DraftCombineDrillResults(season_year=season, league_id=LEAGUE_ID, **nba_kwargs())


@api_call
def _fetch_nss(season: str):
    """Non-stationary shooting (off-dribble)."""
    from nba_api.stats.endpoints import DraftCombineNonStationaryShooting
    return DraftCombineNonStationaryShooting(season_year=season, league_id=LEAGUE_ID, **nba_kwargs())


@api_call
def _fetch_spot(season: str):
    """Spot-up shooting."""
    from nba_api.stats.endpoints import DraftCombineSpotShooting
    return DraftCombineSpotShooting(season_year=season, league_id=LEAGUE_ID, **nba_kwargs())


@api_call
def _fetch_draft_history(season_year: int):
    from nba_api.stats.endpoints import DraftHistory
    return DraftHistory(league_id=LEAGUE_ID, season_year_nullable=str(season_year), **nba_kwargs())


# ---------------------------------------------------------------------------
# Public orchestrators
# ---------------------------------------------------------------------------

_FT_IN_RENAME = {
    "HEIGHT_WO_SHOES_FT_IN": "height_wo_shoes_ft",
    "HEIGHT_W_SHOES_FT_IN":  "height_w_shoes_ft",
    "WINGSPAN_FT_IN":        "wingspan_ft",
    "STANDING_REACH_FT_IN":  "standing_reach_ft",
}


def fetch_draft_combine(season: str = DRAFT_COMBINE_SEASON,
                        db_path: str = "nba_draft_analytics.db") -> int:
    """
    Fetch all draft combine sub-tables for a given season year string
    (e.g. "2024-25") and write them to the database.
    Returns the number of rows stored. The season is only checkpointed as
    'done' if every API call succeeded, so failed seasons are retried next run.
    """
    task = f"draft_combine:{season}"
    if is_done(task, db_path):
        logger.info(f"[SKIP] Draft combine {season} already fetched.")
        return 0

    set_checkpoint(task, "in_progress", db_path=db_path)
    logger.info(f"📋 Fetching draft combine data – season {season}")
    conn = get_connection(db_path)
    results = {
        "anthro": _fetch_anthro(season),
        "drills": _fetch_drills(season),
        "nss":    _fetch_nss(season),
        "spot":   _fetch_spot(season),
    }
    rows = 0

    # ── Anthropometric measurements ──────────────────────────────────────────
    df = safe_df(results["anthro"])
    if not df.empty:
        df = df.rename(columns=_FT_IN_RENAME)
        df["season"] = season
        n = upsert_df(df, "draft_combine_anthro", conn)
        rows += n
        logger.info(f"   anthro: {n} rows")

    # ── Athletic drills ──────────────────────────────────────────────────────
    df = safe_df(results["drills"])
    if not df.empty:
        df["season"] = season
        n = upsert_df(df, "draft_combine_drills", conn)
        rows += n
        logger.info(f"   drills: {n} rows")

    # ── Shooting (merge non-stationary + spot-up into one row per player) ────
    nss_df  = safe_df(results["nss"])
    spot_df = safe_df(results["spot"])
    frames = [d.drop_duplicates("PLAYER_ID").set_index("PLAYER_ID") for d in (spot_df, nss_df) if not d.empty]
    if frames:
        shooting = frames[0]
        if len(frames) == 2:
            shooting = shooting.combine_first(frames[1])
        shooting = shooting.reset_index()
        shooting["season"] = season
        n = upsert_df(shooting, "draft_combine_shooting", conn)
        rows += n
        logger.info(f"   shooting: {n} rows")

    conn.close()

    failed = [name for name, r in results.items() if r is None]
    if failed:
        set_checkpoint(task, "failed", details=",".join(failed), db_path=db_path)
        logger.warning(f"⚠️  Draft combine {season}: failed calls {failed} – will retry next run.")
    else:
        set_checkpoint(task, "done", db_path=db_path)
        logger.info(f"✅ Draft combine {season} complete ({rows} rows).")
    return rows


def fetch_all_combine_seasons(seasons_map: dict[str, int] | None = None,
                              db_path: str = "nba_draft_analytics.db") -> None:
    """
    Fetch draft combine data for every season in DRAFT_YEARS.
    seasons_map: optional {season_str: year_int} override.
    """
    if seasons_map is None:
        seasons_map = {f"{y}-{str(y+1)[-2:]}": y for y in DRAFT_YEARS}

    done = done_tasks("draft_combine:", db_path)
    total_rows = 0
    with progress(seasons_map, desc="Draft combine", unit="season") as bar:
        for season_str in bar:
            bar.set_postfix(season=season_str, rows=total_rows)
            if f"draft_combine:{season_str}" in done:
                continue
            total_rows += fetch_draft_combine(season=season_str, db_path=db_path)
            bar.set_postfix(season=season_str, rows=total_rows)


def fetch_draft_history(years: list[int] = DRAFT_YEARS,
                        db_path: str = "nba_draft_analytics.db") -> None:
    """
    Fetch the draft history for every year in *years* and persist to
    draft_history. Each year has its own checkpoint.
    """
    done = done_tasks("draft_history:", db_path)
    years = [y for y in years if f"draft_history:{y}" not in done]
    if not years:
        logger.info("[SKIP] Draft history already fetched.")
        return

    logger.info("📋 Fetching draft history…")
    conn = get_connection(db_path)

    rename = {
        "PERSON_ID":          "player_id",
        "PLAYER_NAME":        "player_name",
        "SEASON":             "season",
        "ROUND_NUMBER":       "round_number",
        "ROUND_PICK":         "round_pick",
        "OVERALL_PICK":       "overall_pick",
        "TEAM_ID":            "team_id",
        "TEAM_CITY":          "team_city",
        "TEAM_NAME":          "team_name",
        "ORGANIZATION":       "organization",
        "ORGANIZATION_TYPE":  "organization_type",
    }

    failed_years = []
    total = 0
    with progress(years, desc="Draft history", unit="year") as bar:
        for year in bar:
            df = safe_df(_fetch_draft_history(year))
            if df.empty:
                failed_years.append(year)
                set_checkpoint(f"draft_history:{year}", "failed", db_path=db_path)
                logger.warning(f"   No draft data for {year}")
                continue

            df = df.rename(columns=rename)
            df["season"] = str(year)   # API value is the same year; force TEXT

            n = upsert_df(df, "draft_history", conn)
            total += n
            set_checkpoint(f"draft_history:{year}", "done", db_path=db_path)
            bar.set_postfix(year=year, picks=total)
            logger.debug(f"   {year} draft: {n} picks stored.")

    conn.close()
    if failed_years:
        logger.warning(f"⚠️  Draft history incomplete (missing {failed_years}) – will retry next run.")
    else:
        logger.info(f"✅ Draft history complete ({total} picks).")
