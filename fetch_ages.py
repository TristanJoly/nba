"""
fetch_ages.py – birth date of every drafted player, from sources known BEFORE the draft.

  1. Barttorvik (college players)            → table draft_college (fetch_college.py)
     (drafts 2009-2010: year only, placeholder YYYY-10-15 → Wikidata when possible)
  2. Wikidata (everyone else: internationals, G League, JUCO…)
       ONE SPARQL query: every basketball player (occupation Q3665646) born 1984-2008, with
       his English label, birth date and whether he has a "drafted by" claim (P647).
       Matched locally on the normalised name; among the namesakes whose age at the draft
       is plausible (17-27) we keep the single one with a "drafted by" claim, otherwise
       the single plausible one.  Ambiguous or unknown → no age.

Never uses the NBA CommonPlayerInfo birth date: it only exists for players who reached
the NBA, so a missing value would mean "never played" (target leak).

Table draft_birthdates : player_id, dob, source ('barttorvik' | 'barttorvik_approx' | 'wikidata'), wikidata_id

Usage
-----
  python main.py --step ages
"""

import io
import logging

import pandas as pd
import requests

from config import DB_PATH
from database import get_connection, upsert_df
from fetch_college import norm_name

logger = logging.getLogger(__name__)

SPARQL = "https://query.wikidata.org/sparql"
HEADERS = {"User-Agent": "nba-draft-analytics/1.0 (research script)", "Accept": "text/csv"}
FIRST_DRAFT_YEAR = 2009
QUERY = """
SELECT ?p ?label ?dob ?drafted WHERE {
  ?p wdt:P106 wd:Q3665646 ; wdt:P569 ?dob .
  FILTER(?dob >= "1984-01-01T00:00:00Z"^^xsd:dateTime && ?dob < "2009-01-01T00:00:00Z"^^xsd:dateTime)
  ?p rdfs:label ?label . FILTER(LANG(?label) = "en")
  BIND(EXISTS { ?p wdt:P647 ?team } AS ?drafted)
}"""


def wikidata_players() -> pd.DataFrame:
    r = requests.get(SPARQL, params={"query": QUERY}, headers=HEADERS, timeout=180)
    r.raise_for_status()
    wd = pd.read_csv(io.StringIO(r.text))
    wd["wikidata_id"] = wd["p"].str.rsplit("/", n=1).str[-1]
    wd["dob"] = wd["dob"].str[:10]
    wd["drafted"] = wd["drafted"].astype(str).str.lower() == "true"
    wd["key"] = wd["label"].map(norm_name)
    return wd.drop_duplicates(["wikidata_id", "dob"])


def match_wikidata(draft: pd.DataFrame, wd: pd.DataFrame) -> pd.DataFrame:
    """draft: player_id, player_name, draft_year → player_id, dob, wikidata_id."""
    m = draft.assign(key=draft["player_name"].map(norm_name)).merge(wd, on="key")
    age = m["draft_year"] - m["dob"].str[:4].astype(int)
    m = m[(age >= 17) & (age <= 27)]
    out = []
    for pid, g in m.groupby("player_id"):
        g = g.drop_duplicates("wikidata_id")
        drafted = g[g["drafted"]]
        pick = drafted if len(drafted) == 1 else g if len(g) == 1 else None
        if pick is not None:
            out.append({"player_id": pid, "dob": pick["dob"].iloc[0], "wikidata_id": pick["wikidata_id"].iloc[0]})
    return pd.DataFrame(out, columns=["player_id", "dob", "wikidata_id"])


def fetch_ages(db_path: str = DB_PATH, first_year: int = FIRST_DRAFT_YEAR) -> int:
    conn = get_connection(db_path)
    draft = pd.read_sql_query(
        "SELECT h.player_id, h.player_name, CAST(h.season AS INT) AS draft_year, dc.dob "
        "FROM draft_history h LEFT JOIN draft_college dc USING (player_id) "
        "WHERE CAST(h.season AS INT) >= ?", conn, params=(first_year,))
    conn.close()

    wd = wikidata_players()
    logger.info(f"[ages] Wikidata: {len(wd):,} basketball players born 1984-2008")
    found = match_wikidata(draft[["player_id", "player_name", "draft_year"]], wd)

    # Barttorvik's oldest seasons (drafts 2009-2010) only know the birth YEAR: the date is a
    # placeholder "YYYY-10-15" → replaced by Wikidata when it has the player
    placeholder = draft["dob"].fillna("").str.endswith("-10-15")

    # quality check on the college players, whose Barttorvik birth date is known
    chk = found.merge(draft[~placeholder].dropna(subset=["dob"]), on="player_id", suffixes=("", "_bart"))
    if len(chk):
        same = (pd.to_datetime(chk["dob"]) - pd.to_datetime(chk["dob_bart"])).dt.days.abs() <= 31
        logger.info(f"[ages] Wikidata vs Barttorvik on {len(chk)} college players: {same.mean():.1%} agree (±31 days)")

    bart = draft.dropna(subset=["dob"]).assign(source="barttorvik", wikidata_id=None)
    bart.loc[placeholder[bart.index], "source"] = "barttorvik_approx"
    replace = bart["source"].eq("barttorvik_approx") & bart["player_id"].isin(found["player_id"])
    bart = bart[~replace]
    wiki = found[~found["player_id"].isin(bart["player_id"])].assign(source="wikidata")
    rows = pd.concat([bart[["player_id", "dob", "source", "wikidata_id"]], wiki])
    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM draft_birthdates")
        upsert_df(rows, "draft_birthdates", conn)
    missing = draft[~draft["player_id"].isin(rows["player_id"])]
    logger.info(f"✅ draft_birthdates: {len(bart)} from Barttorvik, {len(wiki)} from Wikidata, "
                f"{len(missing)} drafted players still without a birth date")
    return len(wiki)


def run(db_path: str = DB_PATH) -> None:
    fetch_ages(db_path)
