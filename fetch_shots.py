"""
fetch_shots.py – Fetch shot-by-shot data (ShotChartDetail) per player/season.

Each row in *shot_chart* represents one shot attempt and includes:
  • coordinates (loc_x, loc_y)
  • shot distance (feet)
  • zone (basic, area, range)
  • shot type (2PT / 3PT)
  • action type (Jump Shot, Layup, Dunk, Hook Shot, …)
  • period + time remaining
  • made / missed flag
  • closest defender + distance (when available)

Usage
-----
  # Fetch shots for every player found in player_career_stats for seasons 2020-25:
  from fetch_shots import fetch_shots_for_draft_players
  fetch_shots_for_draft_players(db_path="nba_draft_analytics.db")

  # Or target a specific player × season:
  from fetch_shots import fetch_shot_chart
  fetch_shot_chart(player_id=2544, season="2023-24")

WARNING
-------
Shot chart data is LARGE.  Fetching all players × all seasons can produce
10M+ rows and take many hours.  The functions below offer targeted helpers
to limit scope to players relevant to the predictive model (draft picks).
"""

import logging
from typing import Optional

import pandas as pd

from config import SEASONS, LEAGUE_ID, SEASON_TYPE_REGULAR
from database import get_connection, upsert_df, set_checkpoint, is_done, done_tasks, played_seasons
from utils import api_call, nba_kwargs, progress

logger = logging.getLogger(__name__)

_SHOT_RENAME = {
    "GAME_ID":                 "game_id",
    "PLAYER_ID":               "player_id",
    "PLAYER_NAME":             "player_name",
    "TEAM_ID":                 "team_id",
    "PERIOD":                  "period",
    "MINUTES_REMAINING":       "minutes_remaining",
    "SECONDS_REMAINING":       "seconds_remaining",
    "ACTION_TYPE":             "action_type",
    "SHOT_TYPE":               "shot_type",
    "SHOT_ZONE_BASIC":         "shot_zone_basic",
    "SHOT_ZONE_AREA":          "shot_zone_area",
    "SHOT_ZONE_RANGE":         "shot_zone_range",
    "SHOT_DISTANCE":           "shot_distance",
    "LOC_X":                   "loc_x",
    "LOC_Y":                   "loc_y",
    "SHOT_MADE_FLAG":          "shot_made_flag",
    "CLOSEST_DEFENDER":        "closest_defender",
    "CLOSEST_DEFENDER_PLAYER_ID": "closest_defender_id",
    "CLOSE_DEF_DIST":          "closest_defender_dist",
}


@api_call
def _shot_chart_detail(player_id: int, season: str, team_id: int = 0):
    from nba_api.stats.endpoints import ShotChartDetail
    return ShotChartDetail(
        player_id=player_id,
        team_id=team_id,
        season_nullable=season,
        season_type_all_star=SEASON_TYPE_REGULAR,
        league_id=LEAGUE_ID,
        context_measure_simple="FGA",  # shot attempts
        **nba_kwargs(),
    )


def fetch_shot_chart(
    player_id: int,
    season: str,
    db_path: str = "nba_draft_analytics.db",
    team_id: int = 0,
) -> int:
    """
    Fetch and store all shot attempts for one player in one season.
    Returns the number of rows inserted (0 on failure / already done).
    """
    task = f"shots:{player_id}:{season}"
    if is_done(task, db_path):
        return 0

    result = _shot_chart_detail(player_id, season, team_id)
    df = result.shot_chart_detail.get_data_frame() if result else pd.DataFrame()

    if df.empty:
        logger.debug(f"shots player={player_id} season={season}: no data")
        set_checkpoint(task, "failed", db_path=db_path)
        return 0

    df["season"] = season
    df.rename(columns={k: v for k, v in _SHOT_RENAME.items() if k in df.columns},
              inplace=True)

    # Keep only columns that exist in the schema
    keep = list(_SHOT_RENAME.values()) + ["season"]
    df = df[[c for c in keep if c in df.columns]]

    conn = get_connection(db_path)
    # shot_chart uses AUTOINCREMENT id → use INSERT OR IGNORE (avoid duplication on resume)
    cols = df.columns.tolist()
    col_str  = ", ".join(f'"{c}"' for c in cols)
    ph_str   = ", ".join("?" for _ in cols)
    sql      = f"INSERT OR IGNORE INTO shot_chart ({col_str}) VALUES ({ph_str})"

    data = [
        [None if pd.isna(v) else v for v in row]
        for row in df.astype(object).itertuples(index=False, name=None)
    ]
    conn.executemany(sql, data)
    conn.commit()
    n = len(data)
    conn.close()

    set_checkpoint(task, "done", db_path=db_path)
    logger.debug(f"shots player={player_id} season={season}: {n} rows stored")
    return n


def _run_shot_jobs(jobs: list[tuple[int, str]], desc: str, db_path: str) -> int:
    """Fetch every (player_id, season) not yet checkpointed, with a progress bar."""
    # Skip players with no stats in a season: the API would return no shots
    played = played_seasons({s for _, s in jobs}, db_path)
    covered = {s for _, s in played}
    jobs = [(p, s) for p, s in jobs if s not in covered or (p, s) in played]

    done = done_tasks("shots:", db_path)
    todo = [(p, s) for p, s in jobs if f"shots:{p}:{s}" not in done]
    logger.info(f"🏀 {desc}: {len(jobs) - len(todo)} already done, {len(todo)} calls to make")

    rows = 0
    with progress(todo, desc=desc, unit="call") as bar:
        for pid, season in bar:
            rows += fetch_shot_chart(pid, season, db_path)
            bar.set_postfix(player=pid, season=season, shots=f"{rows:,}")
    logger.info(f"✅ {desc} done. Shots stored: {rows:,}")
    return rows


def fetch_shots_for_player_list(
    player_ids: list[int],
    seasons: list[str] = SEASONS,
    db_path: str = "nba_draft_analytics.db",
) -> None:
    """
    Fetch shots for every (player_id, season) combination in *player_ids × seasons*.
    Respects checkpoints so it can be safely interrupted and resumed.
    """
    total = len(player_ids) * len(seasons)
    logger.info(f"🏀 Fetching shot charts: {len(player_ids)} players × {len(seasons)} seasons = {total} calls")

    _run_shot_jobs([(pid, s) for pid in player_ids for s in seasons], "Shot charts", db_path)


def fetch_shots_for_draft_players(
    draft_years_range: tuple[int, int] = (2014, 2025),
    seasons: list[str] = SEASONS,
    db_path: str = "nba_draft_analytics.db",
) -> None:
    """
    Convenience function: look up all players who appear in *draft_history*
    for the given year range, then fetch their shot charts for all seasons
    after their draft year.
    """
    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT player_id, season FROM draft_history WHERE CAST(season AS INTEGER) >= ? AND CAST(season AS INTEGER) <= ?",
        (draft_years_range[0], draft_years_range[1]),
    ).fetchall()
    conn.close()

    if not rows:
        logger.warning("No draft history found. Run fetch_draft_history() first.")
        return

    # Build player → first available season mapping
    from collections import defaultdict
    player_first_season: dict[int, int] = defaultdict(lambda: 9999)
    for row in rows:
        pid   = row["player_id"]
        dyear = int(row["season"])
        player_first_season[pid] = min(player_first_season[pid], dyear)

    player_ids = sorted(player_first_season.keys())

    # Filter SEASONS to those at or after the player's draft year
    def _season_year(s: str) -> int:
        return int(s.split("-")[0])

    jobs = [(pid, s) for pid in player_ids for s in seasons
            if _season_year(s) >= player_first_season[pid]]
    _run_shot_jobs(jobs, "Draft player shots", db_path)


def fetch_shots_current_combine(
    seasons: list[str] = SEASONS[-3:],
    db_path: str = "nba_draft_analytics.db",
) -> None:
    """
    Fetch shot charts for the combine players of *seasons* who already have
    college or professional shooting data accessible through the API.
    (Some prospects may have no NBA data yet—those calls will return empty.)
    """
    conn = get_connection(db_path)
    marks = ", ".join("?" for _ in seasons)
    rows = conn.execute(
        f"SELECT DISTINCT player_id FROM draft_combine_anthro WHERE season IN ({marks})",
        seasons,
    ).fetchall()
    conn.close()

    player_ids = [r["player_id"] for r in rows]
    if not player_ids:
        logger.warning("No combine players in DB yet. Run fetch_draft_combine() first.")
        return

    logger.info(f"🏀 Fetching shot data for {len(player_ids)} combine players…")
    _run_shot_jobs([(pid, s) for pid in player_ids for s in seasons],
                   "Combine player shots", db_path)
