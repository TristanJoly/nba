"""
main.py – CLI orchestrator for the NBA draft analytics database pipeline.

Usage examples
--------------
  # Setup schema only (always run first):
  python main.py --step setup

  # Full pipeline (several hours):
  python main.py --step all

  # Individual steps:
  python main.py --step draft_combine
  python main.py --step draft_history
  python main.py --step teams
  python main.py --step player_list
  python main.py --step season_stats
  python main.py --step hustle
  python main.py --step tracking
  python main.py --step career
  python main.py --step shots_draft        # shots for drafted players only
  python main.py --step shots_combine      # shots for current combine players
  python main.py --step graph
  python main.py --step college            # NCAA stats + birth dates (Barttorvik, no NBA API)
  python main.py --step ages               # birth date of every draftee (Barttorvik + Wikidata)
  python main.py --step bbref              # NBA BPM / VORP per season (Basketball-Reference)
  python main.py --step model              # prospect table + predictive model (offline)
  python main.py --step progression        # season-to-season BPM jump model + next-season predictions

  # Limit seasons (faster for testing):
  python main.py --step season_stats --seasons 2023-24,2024-25

  # Offline import from the Kaggle "NBA Database" (if stats.nba.com blocks you):
  python main.py --step import_kaggle --source "C:/Users/me/Downloads/archive.zip"

  # Custom DB path:
  python main.py --step all --db my_nba.db
"""

import argparse
import logging
import sys
import time

from utils import setup_logging, check_api_connection, progress, ApiUnavailable
from config import SEASONS, CURRENT_SEASON, DB_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NBA Draft Analytics – Database Builder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--step",
        required=True,
        choices=[
            "setup",
            "draft_combine",
            "draft_history",
            "teams",
            "player_list",
            "player_info",
            "season_stats",
            "hustle",
            "tracking",
            "career",
            "shots_draft",
            "shots_combine",
            "graph",
            "college",
            "ages",
            "bbref",
            "model",
            "progression",
            "import_kaggle",
            "all",
        ],
        help="Which pipeline step to run.",
    )
    parser.add_argument(
        "--seasons",
        default=None,
        help='Comma-separated season strings, e.g. "2022-23,2023-24". '
             "Defaults to all 10 seasons defined in config.py.",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Kaggle archive.zip or nba.sqlite (for --step import_kaggle).",
    )
    parser.add_argument(
        "--db",
        default=DB_PATH,
        help=f"Path to the SQLite database file (default: {DB_PATH}).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return parser.parse_args()


def resolve_seasons(seasons_arg: str | None) -> list[str]:
    if seasons_arg:
        return [s.strip() for s in seasons_arg.split(",") if s.strip()]
    return SEASONS


# ===========================================================================
# STEP RUNNERS
# ===========================================================================

def step_setup(db_path: str, **_) -> None:
    from database import create_tables
    create_tables(db_path)
    print(f"\n✅ Schema created in '{db_path}'. You can now run other steps.\n")


def step_draft_combine(db_path: str, seasons: list[str], **_) -> None:
    from fetch_draft import fetch_all_combine_seasons
    fetch_all_combine_seasons({s: int(s[:4]) for s in seasons}, db_path=db_path)


def step_draft_history(db_path: str, seasons: list[str], **_) -> None:
    from fetch_draft import fetch_draft_history
    fetch_draft_history([int(s[:4]) for s in seasons], db_path=db_path)


def step_teams(db_path: str, seasons: list[str], **_) -> None:
    from fetch_teams import fetch_all_team_data
    fetch_all_team_data(seasons=seasons, db_path=db_path)


def step_player_list(db_path: str, **_) -> None:
    from fetch_players import fetch_all_players_list
    fetch_all_players_list(db_path=db_path)


def step_player_info(db_path: str, seasons: list[str], **_) -> None:
    """Enrich the players who played in *seasons* (one call per player)."""
    from database import played_seasons
    from fetch_players import fetch_player_info_batch

    player_ids = sorted({p for p, _s in played_seasons(seasons, db_path)})
    logging.getLogger(__name__).info(f"Will fetch CommonPlayerInfo for {len(player_ids)} players.")
    fetch_player_info_batch(player_ids, db_path=db_path)


def step_season_stats(db_path: str, seasons: list[str], **_) -> None:
    from fetch_players import fetch_all_season_stats
    fetch_all_season_stats(seasons=seasons, db_path=db_path)


def step_hustle(db_path: str, seasons: list[str], **_) -> None:
    from fetch_players import fetch_hustle_stats
    rows = 0
    with progress(seasons, desc="Hustle stats", unit="season") as bar:
        for s in bar:
            rows += fetch_hustle_stats(s, db_path=db_path)
            bar.set_postfix(season=s, rows=rows)


def step_tracking(db_path: str, seasons: list[str], **_) -> None:
    from fetch_players import fetch_tracking_stats
    rows = 0
    with progress(seasons, desc="Tracking stats", unit="season") as bar:
        for s in bar:
            rows += fetch_tracking_stats(s, db_path=db_path)
            bar.set_postfix(season=s, rows=rows)


def step_career(db_path: str, seasons: list[str], **_) -> None:
    """Fetch career stats for the drafted players who played in *seasons*."""
    from database import get_connection, played_seasons
    from fetch_players import fetch_player_career_batch

    conn = get_connection(db_path)
    drafted = {r["player_id"] for r in conn.execute("SELECT DISTINCT player_id FROM draft_history")}
    conn.close()

    player_ids = sorted(drafted & {p for p, _s in played_seasons(seasons, db_path)})
    logging.getLogger(__name__).info(
        f"Fetching career stats for {len(player_ids)} drafted players."
    )
    fetch_player_career_batch(player_ids, db_path=db_path)


def step_shots_draft(db_path: str, seasons: list[str], **_) -> None:
    from fetch_shots import fetch_shots_for_draft_players
    fetch_shots_for_draft_players(seasons=seasons, db_path=db_path)


def step_shots_combine(db_path: str, seasons: list[str], **_) -> None:
    from fetch_shots import fetch_shots_current_combine
    fetch_shots_current_combine(seasons=seasons, db_path=db_path)


def step_graph(db_path: str, **_) -> None:
    from graph_builder import build_graph, persist_graph_edges, analyse_graph, export_gexf
    G = build_graph(db_path=db_path)
    persist_graph_edges(G, db_path=db_path)
    analyse_graph(G)
    export_gexf(G, "nba_graph.gexf")


def step_college(db_path: str, seasons: list[str], **_) -> None:
    from fetch_college import run
    run(last_year=max(int(s[:4]) for s in seasons) + 1, db_path=db_path)


def step_ages(db_path: str, **_) -> None:
    from fetch_ages import run
    run(db_path=db_path)


def step_bbref(db_path: str, seasons: list[str], **_) -> None:
    from fetch_bbref import run
    run(seasons=seasons, db_path=db_path)


def step_model(db_path: str, **_) -> None:
    from modeling import run
    run(db_path=db_path)


def step_progression(db_path: str, **_) -> None:
    from progression import run
    run(db_path=db_path)


def step_import_kaggle(db_path: str, source: str | None = None, **_) -> None:
    from import_kaggle import import_kaggle
    if not source:
        raise ValueError('--source is required, e.g. --source "C:/Users/me/Downloads/archive.zip"')
    import_kaggle(source, db_path=db_path)


# ===========================================================================
# MAIN
# ===========================================================================

STEP_MAP = {
    "setup":         step_setup,
    "draft_combine": step_draft_combine,
    "draft_history": step_draft_history,
    "teams":         step_teams,
    "player_list":   step_player_list,
    "player_info":   step_player_info,
    "season_stats":  step_season_stats,
    "hustle":        step_hustle,
    "tracking":      step_tracking,
    "career":        step_career,
    "shots_draft":   step_shots_draft,
    "shots_combine": step_shots_combine,
    "graph":         step_graph,
    "college":       step_college,
    "ages":          step_ages,
    "bbref":         step_bbref,
    "model":         step_model,
    "progression":   step_progression,
    "import_kaggle": step_import_kaggle,
}

ALL_STEPS_ORDERED = [
    "setup",
    "draft_combine",
    "draft_history",
    "teams",
    "player_list",
    "season_stats",
    "player_info",   # after season_stats: only players of the selected seasons
    "hustle",
    "tracking",
    "career",
    "shots_draft",
    "shots_combine",
    "graph",
    "college",
    "ages",
    "bbref",
]


# Steps that do not call the NBA API
OFFLINE_STEPS = {"setup", "graph", "college", "ages", "bbref", "model", "progression", "import_kaggle"}


def print_summary(db_path: str, counts_before: dict[str, int], elapsed: float) -> None:
    from database import table_counts
    counts = table_counts(db_path)
    print(f"\n📦 Database summary ({db_path}) – {elapsed / 60:.1f} min")
    print(f"   {'table':<28s} {'rows':>12s} {'new':>10s}")
    for table, n in counts.items():
        new = n - counts_before.get(table, 0)
        print(f"   {table:<28s} {n:>12,} {('+' + format(new, ',')) if new else '':>10s}")
    print()


def main() -> None:
    args = parse_args()
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)

    seasons = resolve_seasons(args.seasons)
    logger.info(f"DB path : {args.db}")
    logger.info(f"Seasons : {seasons}")
    logger.info(f"Step    : {args.step}")

    kwargs = {"db_path": args.db, "seasons": seasons, "source": args.source}
    steps = ALL_STEPS_ORDERED if args.step == "all" else [args.step]

    # Always make sure the schema exists / is up to date before fetching
    from database import create_tables, table_counts
    create_tables(args.db)
    counts_before = table_counts(args.db)
    t0 = time.time()

    if any(s not in OFFLINE_STEPS for s in steps):
        try:
            check_api_connection()
        except ApiUnavailable as exc:
            logger.error(f"❌ {exc}")
            sys.exit(2)

    failed_steps = []
    for i, step_name in enumerate(steps, 1):
        logger.info(f"\n{'=' * 60}\n  STEP {i}/{len(steps)}: {step_name}\n{'=' * 60}")
        try:
            STEP_MAP[step_name](**kwargs)
        except ApiUnavailable as exc:
            logger.error(f"❌ Aborting at step '{step_name}': {exc}")
            logger.error("   Re-run the same command later: finished work is checkpointed and skipped.")
            print_summary(args.db, counts_before, time.time() - t0)
            sys.exit(2)
        except KeyboardInterrupt:
            logger.warning("Interrupted – re-run the same command to resume.")
            print_summary(args.db, counts_before, time.time() - t0)
            sys.exit(130)
        except Exception as exc:
            logger.error(f"Step '{step_name}' failed: {exc}", exc_info=True)
            failed_steps.append(step_name)
            if len(steps) > 1:
                logger.warning("Continuing to next step…")

    print_summary(args.db, counts_before, time.time() - t0)
    if failed_steps:
        logger.error(f"Steps with errors: {failed_steps}")
        sys.exit(1)
    logger.info("🏆 Done!")


if __name__ == "__main__":
    main()
