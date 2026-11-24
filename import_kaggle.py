"""
import_kaggle.py – Fill the database from the Kaggle "NBA Database" (wyattowalsh/basketball)
instead of calling stats.nba.com (which may block scripts).

Download (free Kaggle account needed):
  https://www.kaggle.com/datasets/wyattowalsh/basketball  →  "Download"  →  archive.zip

Usage
-----
  python main.py --step import_kaggle --source "C:\\Users\\...\\Downloads\\archive.zip"
  (the .zip or the extracted nba.sqlite both work)

Kaggle table             → our table(s)
--------------------------------------------------------------
  draft_history          → draft_history
  draft_combine_stats    → draft_combine_anthro / _drills / _shooting
  player                 → players
  common_player_info     → players (draft, school, physical info)
  team (+ team_info_common) → teams

The Kaggle file has no season stats, rosters, coaches or shot charts:
those steps still need the NBA API.
"""

import logging
import re
import sqlite3
import zipfile
from pathlib import Path

import pandas as pd

from database import get_connection, upsert_df, set_checkpoint, _shooting_columns
from utils import progress

logger = logging.getLogger(__name__)


# ===========================================================================
# SOURCE FILE
# ===========================================================================

def _resolve_sqlite(source: str) -> Path:
    """Return the path of nba.sqlite, extracting it next to the zip if needed."""
    path = Path(source).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix.lower() != ".zip":
        return path

    with zipfile.ZipFile(path) as zf:
        members = [m for m in zf.namelist() if m.lower().endswith((".sqlite", ".db"))]
        if not members:
            raise FileNotFoundError(f"No .sqlite file inside {path.name}")
        member = max(members, key=lambda m: zf.getinfo(m).file_size)
        target = path.parent / Path(member).name
        if target.exists() and target.stat().st_size == zf.getinfo(member).file_size:
            logger.info(f"   Already extracted: {target}")
            return target

        size = zf.getinfo(member).file_size
        logger.info(f"📦 Extracting {member} ({size / 1e9:.1f} GB) → {target}")
        with zf.open(member) as src, open(target, "wb") as dst, \
                progress(total=size, desc="Extracting", unit="B",
                         unit_scale=True, unit_divisor=1024) as bar:
            while chunk := src.read(8 * 1024 * 1024):
                dst.write(chunk)
                bar.update(len(chunk))
    return target


def _read(src: sqlite3.Connection, table: str) -> pd.DataFrame:
    """Read a whole source table with lower-cased columns (empty if missing)."""
    exists = src.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    if not exists:
        logger.warning(f"   Source table '{table}' not found – skipped.")
        return pd.DataFrame()
    df = pd.read_sql_query(f'SELECT * FROM "{table}"', src)
    df.columns = [c.lower() for c in df.columns]
    return df


# ===========================================================================
# HELPERS
# ===========================================================================

def _season_label(value) -> str | None:
    """2019 / "2019" / "2019-20" → "2019-20" (same format as fetch_draft.py)."""
    m = re.match(r"^\s*(\d{4})", str(value))
    if not m:
        return None
    y = int(m.group(1))
    return f"{y}-{str(y + 1)[-2:]}"


def _to_int(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None   # "Undrafted", "", None…


def _split_made_attempt(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """'3-5' → (3, 5, 0.6). Anything else → NaN."""
    parts = series.astype("string").str.extract(r"^\s*(\d+)\s*-\s*(\d+)\s*$")
    made = pd.to_numeric(parts[0], errors="coerce")
    att = pd.to_numeric(parts[1], errors="coerce")
    pct = (made / att).where(att > 0)
    return made, att, pct


def _drop_empty(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Drop rows where every non-key column is null (player skipped that part)."""
    metrics = [c for c in df.columns if c not in keys]
    return df.dropna(subset=metrics, how="all") if metrics else df


# ===========================================================================
# IMPORTERS
# ===========================================================================

def _import_draft_history(src, conn) -> int:
    df = _read(src, "draft_history")
    if df.empty:
        return 0
    df = df.rename(columns={"person_id": "player_id"})
    df["season"] = df["season"].astype(str).str[:4]   # same as fetch_draft: "2019"
    n = upsert_df(df, "draft_history", conn)
    set_checkpoint("draft_history", "done", details="kaggle", db_path=_db(conn))
    return n


def _import_draft_combine(src, conn) -> dict[str, int]:
    df = _read(src, "draft_combine_stats")
    if df.empty:
        return {}
    df["season"] = df["season"].map(_season_label)
    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce")
    df = df.dropna(subset=["player_id", "season"])
    df["player_id"] = df["player_id"].astype("int64")
    keys = ["player_id", "season", "player_name", "position"]
    counts = {}

    # ── Anthropometrics ───────────────────────────────────────────────────
    anthro = df.rename(columns={
        "height_wo_shoes_ft_in": "height_wo_shoes_ft",
        "height_w_shoes_ft_in":  "height_w_shoes_ft",
        "wingspan_ft_in":        "wingspan_ft",
        "standing_reach_ft_in":  "standing_reach_ft",
    })
    cols = keys + [c for c in ("height_wo_shoes", "height_wo_shoes_ft", "height_w_shoes",
                               "height_w_shoes_ft", "weight", "wingspan", "wingspan_ft",
                               "standing_reach", "standing_reach_ft", "body_fat_pct",
                               "hand_length", "hand_width") if c in anthro.columns]
    counts["draft_combine_anthro"] = upsert_df(
        _drop_empty(anthro[cols], keys), "draft_combine_anthro", conn)

    # ── Drills ────────────────────────────────────────────────────────────
    cols = keys + [c for c in ("standing_vertical_leap", "max_vertical_leap",
                               "lane_agility_time", "modified_lane_agility_time",
                               "three_quarter_sprint", "bench_press") if c in df.columns]
    counts["draft_combine_drills"] = upsert_df(
        _drop_empty(df[cols], keys), "draft_combine_drills", conn)

    # ── Shooting: "made-attempt" strings → made / attempt / pct columns ───
    shooting = df[keys].copy()
    for target in _shooting_columns():
        if not target.endswith("_made"):
            continue
        base = target[: -len("_made")]
        source = base if base.startswith(("off_drib_", "on_move_")) else f"spot_{base}"
        if source in df.columns:
            made, att, pct = _split_made_attempt(df[source])
            shooting[f"{base}_made"], shooting[f"{base}_attempt"], shooting[f"{base}_pct"] = made, att, pct
    counts["draft_combine_shooting"] = upsert_df(
        _drop_empty(shooting, keys), "draft_combine_shooting", conn)

    for season in df["season"].unique():
        set_checkpoint(f"draft_combine:{season}", "done", details="kaggle", db_path=_db(conn))
    return counts


def _import_players(src, conn) -> int:
    players = _read(src, "player")
    info = _read(src, "common_player_info")
    if players.empty and info.empty:
        return 0

    players = players.rename(columns={"id": "player_id"})
    info = info.rename(columns={
        "person_id":          "player_id",
        "display_first_last": "full_name",
        "season_exp":         "years_experience",
        "rosterstatus":       "roster_status",
    })
    if not info.empty:
        info["birthdate"] = info["birthdate"].astype(str).str[:10]
        for c in ("draft_year", "draft_round", "draft_number", "years_experience", "team_id"):
            if c in info.columns:
                info[c] = info[c].map(_to_int).astype(object)
        if "is_active" not in info.columns and "roster_status" in info.columns:
            info["is_active"] = (info["roster_status"].astype(str) == "Active").astype(int)

    # Kaggle stores ids as TEXT in some tables and INTEGER in others
    for d in (players, info):
        if not d.empty:
            d["player_id"] = pd.to_numeric(d["player_id"], errors="coerce")
            d.dropna(subset=["player_id"], inplace=True)
            d["player_id"] = d["player_id"].astype("int64")

    if players.empty:
        df = info
    elif info.empty:
        df = players
    else:
        # common_player_info wins where both have a value
        df = info.set_index("player_id").combine_first(players.set_index("player_id")).reset_index()

    df["last_updated"] = pd.Timestamp.now().isoformat(timespec="seconds")
    n = upsert_df(df, "players", conn)
    set_checkpoint("players_list", "done", details="kaggle", db_path=_db(conn))
    return n


def _import_teams(src, conn) -> int:
    teams = _read(src, "team")
    if teams.empty:
        return 0
    teams = teams.rename(columns={"id": "team_id", "nickname": "name"})
    teams["team_id"] = pd.to_numeric(teams["team_id"], errors="coerce")
    info = _read(src, "team_info_common")
    if not info.empty and {"team_id", "team_conference", "team_division"} <= set(info.columns):
        info["team_id"] = pd.to_numeric(info["team_id"], errors="coerce")
        info = (info.rename(columns={"team_conference": "conference", "team_division": "division"})
                    [["team_id", "conference", "division"]].drop_duplicates("team_id", keep="last"))
        teams = teams.merge(info, on="team_id", how="left")
    n = upsert_df(teams, "teams", conn)
    set_checkpoint("teams_static", "done", details="kaggle", db_path=_db(conn))
    return n


def _db(conn: sqlite3.Connection) -> str:
    return conn.execute("PRAGMA database_list").fetchone()[2]


# ===========================================================================
# PUBLIC ENTRY POINT
# ===========================================================================

def import_kaggle(source: str, db_path: str = "nba_draft_analytics.db") -> None:
    sqlite_path = _resolve_sqlite(source)
    logger.info(f"📥 Importing from {sqlite_path}")

    src = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    conn = get_connection(db_path)
    steps = [
        ("draft_history", _import_draft_history),
        ("draft_combine", _import_draft_combine),
        ("players",       _import_players),
        ("teams",         _import_teams),
    ]
    try:
        with progress(steps, desc="Kaggle import", unit="table") as bar:
            for name, fn in bar:
                bar.set_postfix(table=name)
                result = fn(src, conn)
                if isinstance(result, dict):
                    for table, n in result.items():
                        logger.info(f"   {table:<26s} {n:>7,} rows")
                else:
                    logger.info(f"   {name:<26s} {result:>7,} rows")
    finally:
        src.close()
        conn.close()
    logger.info("✅ Kaggle import complete.")
