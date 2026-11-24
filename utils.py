"""
utils.py – Shared utilities used across all fetcher modules.

  • setup_logging()        – configure console + file handlers (tqdm-friendly)
  • progress()             – tqdm progress bar with project-wide defaults
  • nba_kwargs()           – timeout / proxy kwargs passed to every endpoint
  • api_call               – decorator: rate limit + exponential-backoff retry
  • check_api_connection() – fail fast if stats.nba.com is unreachable
  • safe_df()              – safely pull a DataFrame from an nba_api result object
  • chunk_list()           – split a list into fixed-size batches
"""

import sys
import time
import logging
import functools
from typing import Any, Callable

import pandas as pd
import requests
from tqdm import tqdm

from config import (
    RATE_LIMIT_DELAY, MAX_RETRIES, RETRY_DELAY, LOG_FILE,
    REQUEST_TIMEOUT, PROXY, MAX_CONSECUTIVE_FAILURES,
)


class ApiUnavailable(RuntimeError):
    """Raised when stats.nba.com keeps failing – continuing would only waste hours."""


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class _TqdmHandler(logging.StreamHandler):
    """Console handler that prints through tqdm.write so progress bars stay intact."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record), file=sys.stderr)
        except Exception:
            self.handleError(record)


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with both a file handler and a coloured console handler."""
    fmt = "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    # Avoid UnicodeEncodeError on Windows consoles for the emoji in log messages
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

    sh = _TqdmHandler()
    try:
        import colorlog
        sh.setFormatter(
            colorlog.ColoredFormatter(
                "%(log_color)s" + fmt,
                datefmt=datefmt,
                log_colors={
                    "DEBUG": "cyan",
                    "INFO": "green",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "bold_red",
                },
            )
        )
    except ImportError:
        sh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        handlers=[file_handler, sh],
        force=True,
    )
    # urllib3 is very chatty about retries – keep it quiet
    logging.getLogger("urllib3").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Progress bars
# ---------------------------------------------------------------------------

def progress(iterable=None, desc: str = "", unit: str = "call", **kwargs) -> tqdm:
    """tqdm bar with ETA, rate and elapsed time; use bar.set_postfix(...) for row counts."""
    kwargs.setdefault("dynamic_ncols", True)
    kwargs.setdefault("bar_format",
                      "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]{postfix}")
    return tqdm(iterable, desc=desc, unit=unit, **kwargs)


# ---------------------------------------------------------------------------
# API call decorator
# ---------------------------------------------------------------------------

def nba_kwargs() -> dict:
    """Common kwargs for every nba_api endpoint constructor."""
    return {"timeout": REQUEST_TIMEOUT, "proxy": PROXY}


_NETWORK_ERRORS = (requests.exceptions.ConnectionError, requests.exceptions.Timeout)
_consecutive_failures = 0


def api_call(func: Callable) -> Callable:
    """
    Decorator that wraps an NBA API call with:
      1. A mandatory sleep before each call (respect rate limits).
      2. Exponential-backoff retry on network / HTTP / parsing errors.
      3. Returns None on total failure (caller must handle None gracefully).
      4. Raises ApiUnavailable after MAX_CONSECUTIVE_FAILURES network failures
         in a row (the API is down or blocking this IP).
    Programming errors (TypeError, e.g. a wrong parameter name) are raised
    immediately instead of being retried.
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        global _consecutive_failures
        logger = logging.getLogger(func.__module__ or __name__)
        time.sleep(RATE_LIMIT_DELAY)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                result = func(*args, **kwargs)
                _consecutive_failures = 0
                return result
            except (TypeError, NameError, AttributeError):
                raise
            except Exception as exc:
                wait = RETRY_DELAY * attempt
                if attempt < MAX_RETRIES:
                    logger.warning(
                        f"[{func.__name__}] attempt {attempt}/{MAX_RETRIES} failed "
                        f"({type(exc).__name__}). Retrying in {wait}s…"
                    )
                    time.sleep(wait)
                    continue

                logger.error(
                    f"[{func.__name__}] gave up after {MAX_RETRIES} attempts. "
                    f"Last error: {type(exc).__name__}: {exc}"
                )
                if isinstance(exc, _NETWORK_ERRORS):
                    _consecutive_failures += 1
                    if _consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        raise ApiUnavailable(
                            f"{_consecutive_failures} API calls in a row failed with network "
                            "errors – stats.nba.com is unreachable from this connection."
                        ) from exc
                return None

    return wrapper


def check_api_connection() -> None:
    """Make one quick request; raise ApiUnavailable with a helpful message if it fails."""
    from nba_api.stats.endpoints import TeamInfoCommon
    logger = logging.getLogger(__name__)
    logger.info("🔌 Checking connection to stats.nba.com…")
    t0 = time.time()
    try:
        TeamInfoCommon(team_id=1610612747, timeout=20, proxy=PROXY)
    except Exception as exc:
        raise ApiUnavailable(
            f"stats.nba.com did not answer ({type(exc).__name__}). This usually means the "
            "NBA blocks your IP (VPN, cloud host, some ISPs/networks) or you are offline.\n"
            "  → disable your VPN / try another network (e.g. phone hotspot),\n"
            "  → or set a proxy:  $env:NBA_PROXY = \"http://host:port\""
        ) from exc
    logger.info(f"   OK ({time.time() - t0:.1f}s)")


# ---------------------------------------------------------------------------
# Safe DataFrame extractor
# ---------------------------------------------------------------------------

def safe_df(result: Any, index: int = 0) -> pd.DataFrame:
    """
    Extract the DataFrame at *index* from an nba_api result object.
    Returns an empty DataFrame on any failure.
    """
    try:
        if result is None:
            return pd.DataFrame()
        dfs = result.get_data_frames()
        if not dfs or index >= len(dfs):
            return pd.DataFrame()
        return dfs[index]
    except Exception as exc:
        logging.getLogger(__name__).debug(f"safe_df error: {exc}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Batch helper
# ---------------------------------------------------------------------------

def chunk_list(lst: list, size: int):
    """Yield successive *size*-sized chunks from *lst*."""
    for i in range(0, len(lst), size):
        yield lst[i : i + size]
