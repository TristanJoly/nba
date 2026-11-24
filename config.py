"""
config.py – Global configuration for the NBA draft analytics pipeline.
Edit these values to customize behaviour without touching business logic.
"""
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DB_PATH   = os.environ.get("NBA_DB_PATH", "nba_draft_analytics.db")
LOG_FILE  = "nba_fetch.log"

# ---------------------------------------------------------------------------
# NBA API rate limiting
# ---------------------------------------------------------------------------
RATE_LIMIT_DELAY = 1.5    # seconds between every API call (avoid HTTP 429)
MAX_RETRIES      = 3      # maximum retry attempts on failure
RETRY_DELAY      = 10     # base seconds before a retry (multiplied by attempt #)
REQUEST_TIMEOUT  = 30     # socket timeout in seconds
# stats.nba.com blocks many VPN / cloud / some ISP IPs (requests hang forever).
# If so, set an HTTP(S) proxy, e.g.  $env:NBA_PROXY = "http://user:pass@host:port"
PROXY            = os.environ.get("NBA_PROXY") or None
# Abort the run after this many API calls in a row failed all their retries
MAX_CONSECUTIVE_FAILURES = 3

# ---------------------------------------------------------------------------
# Seasons – last 10 regular seasons + current
# ---------------------------------------------------------------------------
def _season_str(start_year: int) -> str:
    return f"{start_year}-{str(start_year + 1)[-2:]}"

SEASONS: list[str] = [_season_str(y) for y in range(2014, 2027)]  # "2014-15" … "2026-27"
CURRENT_SEASON       = "2026-27"   # season in progress (rosters, player list)
LAST_COMPLETE_SEASON = "2025-26"   # last finished regular season (model targets)
DRAFT_COMBINE_SEASON = "2026-27"   # year string used by DraftCombine endpoints (2026 combine)
DRAFT_YEARS: list[int] = list(range(2014, 2027))

# ---------------------------------------------------------------------------
# NBA API parameter strings
# ---------------------------------------------------------------------------
SEASON_TYPE_REGULAR  = "Regular Season"
SEASON_TYPE_PLAYOFFS = "Playoffs"
PER_MODE_PER_GAME    = "PerGame"
PER_MODE_TOTALS      = "Totals"
PER_MODE_PER36       = "Per36"
LEAGUE_ID            = "00"        # 00 = NBA

# ---------------------------------------------------------------------------
# Tracking stat measure types (used with LeagueDashPtStats)
# ---------------------------------------------------------------------------
TRACKING_MEASURES = [
    "Possessions",
    "Passing",
    "CatchShoot",
    "PullUpShot",
    "Defense",
    "Drives",
    "Efficiency",
    "SpeedDistance",
    "ElbowTouch",
    "PostTouch",
    "PaintTouch",
]
