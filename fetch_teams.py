"""
fetch_teams.py – Fetch team-level data and persist to SQLite.

Endpoints used
--------------
  nba_api.stats.static.teams          → teams table (static list)
  TeamInfoCommon                      → teams table (conference, division)
  CommonTeamRoster                    → team_roster + coaches tables
  LeagueDashTeamStats (Base)          → team_stats_traditional
  LeagueDashTeamStats (Advanced)      → team_stats_advanced
  TeamDashboardByGeneralSplits        → team_splits (Home/Road, W/L, Pre/Post All-Star)
"""

import logging

import pandas as pd

from config import (
    SEASONS, CURRENT_SEASON, LEAGUE_ID,
    PER_MODE_PER_GAME, SEASON_TYPE_REGULAR,
)
from database import get_connection, upsert_df, set_checkpoint, is_done, done_tasks
from utils import api_call, safe_df, nba_kwargs, progress

logger = logging.getLogger(__name__)


# ===========================================================================
# API WRAPPERS
# ===========================================================================

@api_call
def _team_info(team_id: int, season: str):
    from nba_api.stats.endpoints import TeamInfoCommon
    return TeamInfoCommon(team_id=team_id, season_nullable=season, league_id=LEAGUE_ID,
                          **nba_kwargs())


@api_call
def _team_roster(team_id: int, season: str):
    from nba_api.stats.endpoints import CommonTeamRoster
    return CommonTeamRoster(team_id=team_id, season=season, league_id_nullable=LEAGUE_ID,
                            **nba_kwargs())


@api_call
def _league_team_traditional(season: str):
    from nba_api.stats.endpoints import LeagueDashTeamStats
    return LeagueDashTeamStats(
        season=season,
        season_type_all_star=SEASON_TYPE_REGULAR,
        per_mode_detailed=PER_MODE_PER_GAME,
        measure_type_detailed_defense="Base",
        league_id_nullable=LEAGUE_ID,
        **nba_kwargs(),
    )


@api_call
def _league_team_advanced(season: str):
    from nba_api.stats.endpoints import LeagueDashTeamStats
    return LeagueDashTeamStats(
        season=season,
        season_type_all_star=SEASON_TYPE_REGULAR,
        per_mode_detailed=PER_MODE_PER_GAME,
        measure_type_detailed_defense="Advanced",
        league_id_nullable=LEAGUE_ID,
        **nba_kwargs(),
    )


@api_call
def _team_splits(team_id: int, season: str, measure_type: str = "Base"):
    from nba_api.stats.endpoints import TeamDashboardByGeneralSplits
    return TeamDashboardByGeneralSplits(
        team_id=team_id,
        season=season,
        season_type_all_star=SEASON_TYPE_REGULAR,
        per_mode_detailed=PER_MODE_PER_GAME,
        measure_type_detailed_defense=measure_type,
        league_id_nullable=LEAGUE_ID,
        **nba_kwargs(),
    )


# ===========================================================================
# PUBLIC ORCHESTRATORS
# ===========================================================================

def fetch_all_teams(db_path: str = "nba_draft_analytics.db") -> list[dict]:
    """
    Load the static team list from nba_api, enrich with TeamInfoCommon for
    the current season (conference, division), and store in *teams* table.
    Returns a list of team dicts.
    """
    task = "teams_static"
    if is_done(task, db_path):
        logger.info("[SKIP] Static team list already loaded.")
        conn = get_connection(db_path)
        rows = conn.execute("SELECT * FROM teams").fetchall()
        conn.close()
        return [dict(r) for r in rows]

    from nba_api.stats.static import teams as nba_static_teams
    teams = nba_static_teams.get_teams()

    conn = get_connection(db_path)
    for t in teams:
        conn.execute(
            "INSERT OR REPLACE INTO teams (team_id, city, name, full_name, abbreviation) "
            "VALUES (?, ?, ?, ?, ?)",
            (t["id"], t["city"], t["nickname"], t["full_name"], t["abbreviation"]),
        )
    conn.commit()
    logger.info(f"   {len(teams)} teams stored (static).")

    # Enrich with conference/division/year_founded via TeamInfoCommon
    logger.info("   Enriching with TeamInfoCommon…")
    for t in progress(teams, desc="Team info", unit="team"):
        result = _team_info(t["id"], CURRENT_SEASON)
        if result is None:
            continue
        df = result.team_info_common.get_data_frame()
        if df.empty:
            continue
        row = df.iloc[0]

        def _g(col: str):
            for v in (col, col.upper(), col.lower()):
                if v in row.index:
                    val = row[v]
                    return None if str(val) in ("", "nan", "None") else val
            return None

        conn.execute(
            "UPDATE teams SET conference=?, division=?, year_founded=? WHERE team_id=?",
            (_g("TEAM_CONFERENCE"), _g("TEAM_DIVISION"), _g("MIN_YEAR"), t["id"]),
        )
    conn.commit()
    conn.close()
    set_checkpoint(task, "done", db_path=db_path)
    logger.info("✅ Teams complete.")
    return teams


def fetch_rosters_and_coaches(seasons: list[str] = SEASONS,
                               db_path: str = "nba_draft_analytics.db") -> None:
    """
    For every team × season, fetch the roster and coaching staff.
    Writes to *team_roster* and *coaches*.
    """
    from nba_api.stats.static import teams as nba_static_teams
    all_teams = nba_static_teams.get_teams()

    conn = get_connection(db_path)

    done = done_tasks("roster:", db_path)
    jobs = [(s, t) for s in seasons for t in all_teams
            if f"roster:{t['id']}:{s}" not in done]
    bar = progress(jobs, desc="Rosters & coaches", unit="team")
    for season, t in bar:
        bar.set_postfix(season=season, team=t["abbreviation"])
        task = f"roster:{t['id']}:{season}"

        result = _team_roster(t["id"], season)
        if result is None:
            set_checkpoint(task, "failed", db_path=db_path)
            continue

        roster_df = result.common_team_roster.get_data_frame()
        coach_df  = result.coaches.get_data_frame()

        # ── Roster ────────────────────────────────────────────────────
        if not roster_df.empty:
            roster_df["team_id"] = t["id"]
            roster_df["season"]  = season
            rename = {
                "PLAYER_ID":    "player_id",
                "PLAYER":       "player_name",
                "NUM":          "jersey_num",
                "POSITION":     "position",
                "HOW_ACQUIRED": "how_acquired",
            }
            roster_df.rename(columns={k: v for k, v in rename.items()
                                      if k in roster_df.columns}, inplace=True)
            upsert_df(roster_df, "team_roster", conn)

        # ── Coaches ───────────────────────────────────────────────────
        if not coach_df.empty:
            coach_df["team_id"] = t["id"]
            coach_df["season"]  = season
            rename = {
                "COACH_ID":       "coach_id",
                "FIRST_NAME":     "first_name",
                "LAST_NAME":      "last_name",
                "COACH_TYPE":     "coach_type",
                "SORT_SEQUENCE":  "sort_sequence",
            }
            coach_df.rename(columns={k: v for k, v in rename.items()
                                     if k in coach_df.columns}, inplace=True)
            upsert_df(coach_df, "coaches", conn)

        set_checkpoint(task, "done", db_path=db_path)

    conn.close()
    logger.info("✅ Rosters and coaches complete.")


def fetch_team_stats(seasons: list[str] = SEASONS,
                     db_path: str = "nba_draft_analytics.db") -> None:
    """
    Fetch Traditional + Advanced team stats for all teams, per season.
    Uses batch endpoints (one call per season per measure type).
    """
    conn = get_connection(db_path)

    for season in progress(seasons, desc="Team stats", unit="season"):
        # Traditional
        task = f"team_stats_traditional:{season}"
        if not is_done(task, db_path):
            df = safe_df(_league_team_traditional(season))
            if not df.empty:
                df["season"] = season
                rename = {
                    "TEAM_ID": "team_id", "TEAM_NAME": "team_name",
                    "TEAM_ABBREVIATION": "team_abbreviation",
                    "GP": "gp", "W": "w", "L": "l", "W_PCT": "w_pct",
                    "MIN": "min", "FGM": "fgm", "FGA": "fga", "FG_PCT": "fg_pct",
                    "FG3M": "fg3m", "FG3A": "fg3a", "FG3_PCT": "fg3_pct",
                    "FTM": "ftm", "FTA": "fta", "FT_PCT": "ft_pct",
                    "OREB": "oreb", "DREB": "dreb", "REB": "reb",
                    "AST": "ast", "TOV": "tov", "STL": "stl", "BLK": "blk",
                    "PTS": "pts", "PLUS_MINUS": "plus_minus",
                }
                df.rename(columns={k: v for k, v in rename.items() if k in df.columns},
                          inplace=True)
                upsert_df(df, "team_stats_traditional", conn)
                logger.info(f"   [{season}] team traditional: {len(df)} rows")
                set_checkpoint(task, "done", db_path=db_path)

        # Advanced
        task = f"team_stats_advanced:{season}"
        if not is_done(task, db_path):
            df = safe_df(_league_team_advanced(season))
            if not df.empty:
                df["season"] = season
                rename = {
                    "TEAM_ID": "team_id", "TEAM_NAME": "team_name",
                    "TEAM_ABBREVIATION": "team_abbreviation",
                    "GP": "gp", "W": "w", "L": "l", "MIN": "min",
                    "OFF_RATING": "off_rating", "DEF_RATING": "def_rating",
                    "NET_RATING": "net_rating",
                    "AST_PCT": "ast_pct", "AST_TO": "ast_to", "AST_RATIO": "ast_ratio",
                    "OREB_PCT": "oreb_pct", "DREB_PCT": "dreb_pct", "REB_PCT": "reb_pct",
                    "TM_TOV_PCT": "tm_tov_pct", "EFG_PCT": "efg_pct",
                    "TS_PCT": "ts_pct", "PACE": "pace", "PIE": "pie",
                }
                df.rename(columns={k: v for k, v in rename.items() if k in df.columns},
                          inplace=True)
                upsert_df(df, "team_stats_advanced", conn)
                logger.info(f"   [{season}] team advanced:     {len(df)} rows")
                set_checkpoint(task, "done", db_path=db_path)

    conn.close()


def fetch_team_splits(seasons: list[str] = SEASONS,
                      db_path: str = "nba_draft_analytics.db") -> None:
    """
    Fetch home/road, win/loss, and pre/post All-Star splits for every team.
    TeamDashboardByGeneralSplits returns multiple DataFrames; we iterate them.
    """
    from nba_api.stats.static import teams as nba_static_teams
    all_teams = nba_static_teams.get_teams()

    # split_type → TeamDashboardByGeneralSplits dataset attribute
    SPLIT_MAP = {
        "Location":       "location_team_dashboard",
        "WinsLosses":     "wins_losses_team_dashboard",
        "Month":          "month_team_dashboard",
        "PrePostAllStar": "pre_post_all_star_team_dashboard",
    }

    conn = get_connection(db_path)

    done = done_tasks("splits:", db_path)
    jobs = [(s, t) for s in seasons for t in all_teams
            if f"splits:{t['id']}:{s}" not in done]
    bar = progress(jobs, desc="Team splits", unit="team")
    for season, t in bar:
        bar.set_postfix(season=season, team=t["abbreviation"])
        task = f"splits:{t['id']}:{season}"

        result = _team_splits(t["id"], season)
        if result is None:
            set_checkpoint(task, "failed", db_path=db_path)
            continue

        for split_type, attr in SPLIT_MAP.items():
            df = getattr(result, attr).get_data_frame()
            if df.empty:
                continue
            df = df.drop(columns=["GROUP_SET"], errors="ignore")
            df = df.rename(columns={"GROUP_VALUE": "split_value"})
            df["team_id"]    = t["id"]
            df["season"]     = season
            df["split_type"] = split_type
            upsert_df(df, "team_splits", conn)

        set_checkpoint(task, "done", db_path=db_path)

    conn.close()
    logger.info("✅ Team splits complete.")


def fetch_all_team_data(seasons: list[str] = SEASONS,
                        db_path: str = "nba_draft_analytics.db") -> None:
    """Convenience wrapper: runs all team-related fetches in order."""
    fetch_all_teams(db_path=db_path)
    fetch_rosters_and_coaches(seasons=seasons, db_path=db_path)
    fetch_team_stats(seasons=seasons, db_path=db_path)
    fetch_team_splits(seasons=seasons, db_path=db_path)
    logger.info("✅ All team data complete.")
