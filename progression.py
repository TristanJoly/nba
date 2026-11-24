"""
progression.py – Predict a player's analytical jump from one season to the next.

Target (pairs of consecutive seasons t → t+1, both with >= MIN_MP minutes):
    d_bpm     = BPM(t+1) − BPM(t)          (Basketball-Reference Box Plus/Minus)
    breakout  = 1 if d_bpm >= BREAKOUT     (a real "leap")

Features — everything known before season t+1 starts:
  joueur    : age, experience, draft pick, height, weight, position
  box       : per-36 box score, shooting %, plus-minus (NBA traditional)
  avance    : usage, TS%, AST%, REB%, ratings, PIE (NBA advanced) + BPM/OBPM/DBPM/PER/WS48/VORP (BBRef)
  scoring   : shot / points mix (NBA scoring)
  hustle    : deflections, contests, screen assists… (from 2015-16)
  tracking  : touches, drives, catch & shoot, pull-ups, speed, defense… (11 NBA tracking families)
  tirs      : zone mix and FG% by zone from the shot chart (2021-22 onwards)
  tendance  : previous season values and changes t-1 → t
  (avance also holds the success index of indice.py and its three pillars)
  contexte  : team of t+1 (city / market size, conference, team strength at t), team change,
              minutes left free on the new team, head coach of t+1 (change, tenure, staff,
              past "development effect" of the coach and of the franchise)

A second target, d_indice = change of the success index (indice.py), is built the same way
(build_pairs(target="indice")); it depends much less on 3-point luck than ΔBPM.

Validation is temporal (rolling origin): each target season is predicted by a model
trained only on the earlier pairs.

Usage
-----
  from progression import build_pairs, evaluate_rolling, predict_next
"""

import json
import logging

import numpy as np
import pandas as pd

from config import DB_PATH, LAST_COMPLETE_SEASON
from database import get_connection
from indice import INDEX_COLS, add_index

logger = logging.getLogger(__name__)

MIN_MP = 500          # minutes in BOTH seasons for a pair to count
BREAKOUT = 2.0        # BPM gain that counts as a leap
SHRINK = 30           # shrinkage (pseudo-pairs) of the coach / franchise effects

# Metropolitan-area population (millions, ≈ 2020 census / Statistics Canada) – market size
MARKET = {"ATL": 6.1, "BOS": 4.9, "BKN": 19.8, "CHA": 2.7, "CHI": 9.6, "CLE": 2.1, "DAL": 7.6,
          "DEN": 3.0, "DET": 4.4, "GSW": 4.7, "HOU": 7.1, "IND": 2.1, "LAC": 13.2, "LAL": 13.2,
          "MEM": 1.3, "MIA": 6.1, "MIL": 1.6, "MIN": 3.7, "NOP": 1.3, "NYK": 19.8, "OKC": 1.4,
          "ORL": 2.7, "PHI": 6.2, "PHX": 4.8, "POR": 2.5, "SAC": 2.4, "SAS": 2.6, "TOR": 6.2,
          "UTA": 1.3, "WAS": 6.3}
ALTITUDE = {"DEN", "UTA"}

FAMILIES = ["joueur", "box", "avance", "scoring", "hustle", "tracking", "tirs", "tendance", "contexte"]


def _season(y: int) -> str:
    return f"{y}-{str(y + 1)[-2:]}"


def _height_in(h) -> float:
    try:
        ft, inch = str(h).split("-")
        return int(ft) * 12 + int(inch)
    except ValueError:
        return np.nan


# ===========================================================================
# PLAYER-SEASON PANEL
# ===========================================================================

def load_panel(db_path: str = DB_PATH) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """One row per (player, season) with every season-level feature, and the column
    list of each feature family."""
    conn = get_connection(db_path)
    q = lambda sql: pd.read_sql_query(sql, conn)
    fam: dict[str, list[str]] = {f: [] for f in FAMILIES}

    trad = q("SELECT * FROM player_stats_traditional")
    trad["start"] = trad["season"].str[:4].astype(int)
    p = trad[["player_id", "season", "start", "team_id", "team_abbreviation", "age", "gp", "gs", "min"]].copy()
    p["mp_nba"] = trad["gp"] * trad["min"]
    per36 = ["pts", "reb", "oreb", "dreb", "ast", "tov", "stl", "blk", "blka", "pf", "pfd",
             "fga", "fg3a", "fta", "fgm", "fg3m", "ftm", "plus_minus"]
    for c in per36:
        p[f"box_{c}36"] = trad[c] * 36 / trad["min"].replace(0, np.nan)
    for c in ["fg_pct", "fg3_pct", "ft_pct"]:
        p[f"box_{c}"] = trad[c]
    p["box_min"], p["box_gp"] = trad["min"], trad["gp"]
    p["box_gs_rate"] = trad["gs"] / trad["gp"].replace(0, np.nan)
    p["box_fg3a_rate"] = trad["fg3a"] / trad["fga"].replace(0, np.nan)
    p["box_fta_rate"] = trad["fta"] / trad["fga"].replace(0, np.nan)
    p["box_ast_tov"] = trad["ast"] / trad["tov"].replace(0, np.nan)
    fam["box"] = [c for c in p.columns if c.startswith("box_")]

    # NBA advanced + Basketball-Reference
    adv = q("SELECT * FROM player_stats_advanced").drop(columns=["team_id", "team_abbreviation", "age", "gp", "min"])
    adv = adv.rename(columns={c: f"adv_{c}" for c in adv.columns if c not in ("player_id", "season")})
    p = p.merge(adv, on=["player_id", "season"], how="left")
    bb = q("SELECT player_id, season, mp, g, per, ws, ws_per_48, obpm, dbpm, bpm, vorp, pos "
           "FROM bbref_advanced WHERE player_id IS NOT NULL")
    bb = bb.sort_values("mp").groupby(["player_id", "season"]).tail(1)        # traded: season total
    bb = bb.rename(columns={c: f"adv_bb_{c}" for c in ["per", "ws", "ws_per_48", "obpm", "dbpm", "vorp"]})
    p = p.merge(bb, on=["player_id", "season"], how="left")
    fam["avance"] = [c for c in p.columns if c.startswith("adv_")] + ["bpm"]

    sco = q("SELECT * FROM player_stats_scoring").drop(columns=["team_id"])
    sco = sco.rename(columns={c: f"sco_{c}" for c in sco.columns if c not in ("player_id", "season")})
    p = p.merge(sco, on=["player_id", "season"], how="left")
    fam["scoring"] = [c for c in sco.columns if c.startswith("sco_")]

    hus = q("SELECT * FROM player_stats_hustle").drop(columns=["team_id", "team_abbreviation", "gp"])
    hmin = hus.pop("min").replace(0, np.nan)
    for c in hus.columns.drop(["player_id", "season"]):
        hus[c] = hus[c] * 36 / hmin
    hus = hus.rename(columns={c: f"hus_{c}36" for c in hus.columns if c not in ("player_id", "season")})
    p = p.merge(hus, on=["player_id", "season"], how="left")
    fam["hustle"] = [c for c in hus.columns if c.startswith("hus_")]

    # tracking: 11 families stored as JSON
    trk = q("SELECT player_id, season, measure_type, data_json FROM player_stats_tracking")
    skip = {"PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION", "GP", "W", "L", "MIN",
            "PLAYER_LAST_TEAM_ID", "PLAYER_LAST_TEAM_ABBREVIATION"}
    recs = {}
    for r in trk.itertuples(index=False):
        d = json.loads(r.data_json)
        row = recs.setdefault((r.player_id, r.season), {})
        for k, v in d.items():
            if k not in skip and isinstance(v, (int, float)):
                row[f"trk_{r.measure_type}_{k.lower()}"] = v
    trk = pd.DataFrame.from_dict(recs, orient="index")
    trk.index = pd.MultiIndex.from_tuples(trk.index, names=["player_id", "season"])
    trk = trk.loc[:, trk.notna().mean() > 0.5].reset_index()
    p = p.merge(trk, on=["player_id", "season"], how="left")
    fam["tracking"] = [c for c in trk.columns if c.startswith("trk_")]

    # shot chart (2021-22+): zone mix and accuracy
    sh = q("""SELECT player_id, season, shot_zone_basic AS z, COUNT(*) AS n, AVG(shot_made_flag) AS pct,
                     AVG(shot_distance) AS dist
              FROM shot_chart GROUP BY 1, 2, 3""")
    if not sh.empty:
        tot = sh.groupby(["player_id", "season"])["n"].transform("sum")
        sh["share"] = sh["n"] / tot
        zones = {"Restricted Area": "rim", "In The Paint (Non-RA)": "paint", "Mid-Range": "mid",
                 "Left Corner 3": "corner3", "Right Corner 3": "corner3", "Above the Break 3": "ab3"}
        sh["z"] = sh["z"].map(zones)
        sh = sh.dropna(subset=["z"])
        sh["made"] = sh["pct"] * sh["n"]
        g = sh.groupby(["player_id", "season", "z"]).agg(n=("n", "sum"), made=("made", "sum"), share=("share", "sum"))
        g["pct"] = g["made"] / g["n"]
        wide = g[["share", "pct"]].unstack("z")
        wide.columns = [f"tir_{z}_{m}" for m, z in wide.columns]
        dist = q("SELECT player_id, season, AVG(shot_distance) AS tir_distance FROM shot_chart GROUP BY 1, 2")
        wide = wide.reset_index().merge(dist, on=["player_id", "season"], how="left")
        p = p.merge(wide, on=["player_id", "season"], how="left")
        fam["tirs"] = [c for c in wide.columns if c.startswith("tir_")]

    # player bio
    pl = q("SELECT player_id, height, weight, draft_year, draft_number, birthdate FROM players")
    pl["taille"] = pl["height"].map(_height_in)
    pl["poids"] = pd.to_numeric(pl["weight"], errors="coerce")
    pl["pick"] = pd.to_numeric(pl["draft_number"], errors="coerce")
    pl["non_drafte"] = pl["pick"].isna().astype(int)
    pl["draft_year"] = pd.to_numeric(pl["draft_year"], errors="coerce")
    p = p.merge(pl[["player_id", "taille", "poids", "pick", "non_drafte", "draft_year"]], on="player_id", how="left")
    first = p.groupby("player_id")["start"].transform("min")
    p["experience"] = (p["start"] - p["draft_year"].where(p["draft_year"] <= p["start"], first)).clip(lower=0)
    p["age2"] = (p["age"] - 26) ** 2
    pos = p["pos"].fillna("").str.split("-").str[0]
    for k, v in {"meneur": "PG", "arriere": "SG", "ailier": "SF", "ailier_fort": "PF", "pivot": "C"}.items():
        p[f"poste_{k}"] = (pos == v).astype(int)
    fam["joueur"] = ["age", "age2", "experience", "taille", "poids", "pick", "non_drafte"] + \
                    [c for c in p.columns if c.startswith("poste_")]

    # trends: previous season and change
    p = p.sort_values(["player_id", "start"])
    prev = p.groupby("player_id").shift(1)
    consecutive = (p["start"] - prev["start"]) == 1
    for c in ["bpm", "mp", "adv_usg_pct", "adv_ts_pct", "box_fg3_pct", "box_ft_pct", "box_pts36",
              "adv_bb_obpm", "adv_bb_dbpm", "box_min"]:
        p[f"tend_{c}_prec"] = prev[c].where(consecutive)
        p[f"tend_{c}_delta"] = p[c] - p[f"tend_{c}_prec"]
    p["tend_saisons_dans_panel"] = p.groupby("player_id").cumcount()
    fam["tendance"] = [c for c in p.columns if c.startswith("tend_")]
    conn.close()
    p = add_index(p)
    fam["avance"] += [c for c in INDEX_COLS if c != "indice_volume"] + ["ix_ts_ajoutes36"]
    # trends of the index (computed after the index itself)
    p = p.sort_values(["player_id", "start"])
    prev = p.groupby("player_id").shift(1)
    consecutive = (p["start"] - prev["start"]) == 1
    p["tend_indice_prec"] = prev["indice"].where(consecutive)
    p["tend_indice_delta"] = p["indice"] - p["tend_indice_prec"]
    fam["tendance"] += ["tend_indice_prec", "tend_indice_delta"]
    return p.reset_index(drop=True), fam


# ===========================================================================
# CONTEXT OF SEASON t+1 (team, city, coach)
# ===========================================================================

def _head_coaches(conn) -> pd.DataFrame:
    co = pd.read_sql_query("SELECT team_id, season, coach_id, first_name || ' ' || last_name AS coach, "
                           "coach_type FROM coaches", conn)
    staff = co.groupby(["team_id", "season"]).agg(
        coach_staff=("coach_id", "nunique"),
        coach_dev_staff=("coach_type", lambda s: s.str.contains("Development", case=False).sum()))
    head = co[co["coach_type"] == "Head Coach"].drop_duplicates(["team_id", "season"])
    head = head.set_index(["team_id", "season"])[["coach_id", "coach"]].join(staff, how="outer").reset_index()
    head["start"] = head["season"].str[:4].astype(int)
    head = head.sort_values(["team_id", "start"])
    # tenure: consecutive seasons of the same head coach with this team
    same = head.groupby("team_id")["coach_id"].shift(1).eq(head["coach_id"]) & head["coach_id"].notna()
    head["coach_anciennete"] = same.groupby((~same).cumsum()).cumsum() + 1
    head.loc[head["coach_id"].isna(), "coach_anciennete"] = np.nan
    return head


def build_pairs(db_path: str = DB_PATH, next_rosters: pd.DataFrame | None = None,
                min_mp: int = MIN_MP, target: str = "bpm") -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """
    Pairs t → t+1, target = change of *target* ("bpm" or "indice"; both changes are always
    computed, *target* picks the one used by the past coach / franchise effects and,
    by default, by the evaluation functions).  Pairs whose season t+1 is not complete are kept with NaN target
    (to predict them).  *next_rosters*: optional (player_id, season, team_id) of the coming
    season (if not given: team of t+1 from player_stats_traditional / team_roster).
    """
    panel, fam = load_panel(db_path)
    conn = get_connection(db_path)
    teams = pd.read_sql_query("SELECT team_id, abbreviation, city FROM teams", conn)
    tstat = pd.read_sql_query("SELECT a.team_id, a.season, a.net_rating, a.off_rating, a.def_rating, a.pace, "
                              "t.w_pct FROM team_stats_advanced a JOIN team_stats_traditional t USING (team_id, season)", conn)
    roster = pd.read_sql_query("SELECT team_id, player_id, season FROM team_roster", conn)
    head = _head_coaches(conn)
    conn.close()

    last_full = int(LAST_COMPLETE_SEASON[:4])
    cur = panel.copy()
    nxt = panel[["player_id", "start", "team_id", "bpm", "indice", "mp", "mp_nba", "gp"]].copy()
    nxt["start"] -= 1
    nxt = nxt.rename(columns={"team_id": "team_next", "bpm": "bpm_next", "indice": "indice_next", "mp": "mp_next",
                              "mp_nba": "mp_nba_next", "gp": "gp_next"})
    pairs = cur.merge(nxt, on=["player_id", "start"], how="left")

    # team of t+1: roster of t+1 (start of season t+1 for the coming season), else stats
    ros = roster.assign(start=roster["season"].str[:4].astype(int) - 1)[["player_id", "start", "team_id"]]
    ros = ros.drop_duplicates(["player_id", "start"]).rename(columns={"team_id": "team_roster_next"})
    pairs = pairs.merge(ros, on=["player_id", "start"], how="left")
    pairs["team_next"] = pairs["team_roster_next"].fillna(pairs["team_next"])
    if next_rosters is not None:
        nr = next_rosters.assign(start=next_rosters["season"].str[:4].astype(int) - 1)
        nr = nr.drop_duplicates(["player_id", "start"]).set_index(["player_id", "start"])["team_id"]
        key = pd.MultiIndex.from_frame(pairs[["player_id", "start"]])
        pairs["team_next"] = pd.Series(nr.reindex(key).values, index=pairs.index).fillna(pairs["team_next"])
    pairs = pairs[pairs["team_next"].notna() & (pairs["start"] + 1 <= last_full + 1)]
    pairs["season_next"] = (pairs["start"] + 1).map(_season)

    # ── context: team / city ────────────────────────────────────────────────
    ab = teams.set_index("team_id")["abbreviation"]
    pairs["ctx_equipe"] = pairs["team_next"].map(ab)
    pairs["ctx_marche_millions"] = pairs["ctx_equipe"].map(MARKET)
    pairs["ctx_altitude"] = pairs["ctx_equipe"].isin(ALTITUDE).astype(int)
    pairs["ctx_change_equipe"] = (pairs["team_next"] != pairs["team_id"]).astype(int)
    ts = tstat.rename(columns={c: f"ctx_eq_{c}" for c in ["net_rating", "off_rating", "def_rating", "pace", "w_pct"]})
    pairs = pairs.merge(ts.rename(columns={"team_id": "team_next"}), on=["team_next", "season"], how="left")

    # minutes freed on the team of t+1: minutes of season t whose player is gone
    on_next = set(zip(pairs["player_id"], pairs["season"], pairs["team_next"].astype(int)))
    t_min = panel[["player_id", "season", "team_id", "mp_nba", "adv_usg_pct"]].copy()
    t_min["reste"] = [(pid, s, int(tid)) in on_next
                      for pid, s, tid in zip(t_min["player_id"], t_min["season"], t_min["team_id"])]
    team_tot = t_min.groupby(["team_id", "season"])["mp_nba"].sum()
    gone = t_min[~t_min["reste"]].groupby(["team_id", "season"])["mp_nba"].sum().reindex(team_tot.index).fillna(0)
    freed = (gone / team_tot).rename("ctx_minutes_liberees").reset_index().rename(columns={"team_id": "team_next"})
    pairs = pairs.merge(freed, on=["team_next", "season"], how="left")
    # usage of the teammates who stay (competition for the ball)
    stay = t_min[t_min["reste"]].assign(u=lambda d: d["adv_usg_pct"] * d["mp_nba"])
    comp = stay.groupby(["team_id", "season"]).agg(u=("u", "sum"), m=("mp_nba", "sum"))
    comp = (comp["u"] / comp["m"]).rename("ctx_usage_coequipiers").reset_index().rename(columns={"team_id": "team_next"})
    pairs = pairs.merge(comp, on=["team_next", "season"], how="left")

    # ── context: head coach of t+1 ──────────────────────────────────────────
    hn = head.rename(columns={"team_id": "team_next", "season": "season_next"})
    pairs = pairs.merge(hn[["team_next", "season_next", "coach_id", "coach", "coach_anciennete",
                            "coach_staff", "coach_dev_staff"]], on=["team_next", "season_next"], how="left")
    hp = head[["team_id", "season", "coach_id"]].rename(columns={"team_id": "team_next", "coach_id": "coach_prec"})
    pairs = pairs.merge(hp, on=["team_next", "season"], how="left")
    pairs["ctx_change_coach"] = (pairs["coach_id"] != pairs["coach_prec"]).astype(float)
    pairs.loc[pairs["coach_id"].isna() | pairs["coach_prec"].isna(), "ctx_change_coach"] = np.nan
    pairs = pairs.rename(columns={"coach_anciennete": "ctx_coach_anciennete", "coach_staff": "ctx_coach_staff",
                                  "coach_dev_staff": "ctx_coach_staff_developpement"})

    # ── target ──────────────────────────────────────────────────────────────
    ok = (pairs["mp"] >= min_mp) & (pairs["mp_next"] >= min_mp)
    pairs = pairs[pairs["mp"] >= min_mp].copy()
    pairs["d_bpm"] = np.where(ok[pairs.index], pairs["bpm_next"] - pairs["bpm"], np.nan)
    pairs.loc[pairs["start"] + 1 > last_full, "d_bpm"] = np.nan          # coming season: unknown
    pairs["breakout"] = np.where(pairs["d_bpm"].notna(), (pairs["d_bpm"] >= BREAKOUT).astype(float), np.nan)
    pairs["d_indice"] = np.where(pairs["d_bpm"].notna(), pairs["indice_next"] - pairs["indice"], np.nan)
    pairs["breakout_indice"] = np.where(pairs["d_indice"].notna(),
                                        (pairs["d_indice"] >= BREAKOUT).astype(float), np.nan)

    # ── past-only development effects (coach, franchise) ────────────────────
    y = TARGETS[target]["y"]
    pairs = _past_effects(pairs, "coach_id", "ctx_effet_coach", y, TARGETS[target]["niveau"])
    pairs = _past_effects(pairs, "team_next", "ctx_effet_franchise", y, TARGETS[target]["niveau"])
    pairs.attrs["target"] = target

    fam["contexte"] = ["ctx_marche_millions", "ctx_altitude", "ctx_change_equipe", "ctx_eq_net_rating",
                       "ctx_eq_off_rating", "ctx_eq_def_rating", "ctx_eq_pace", "ctx_eq_w_pct",
                       "ctx_minutes_liberees", "ctx_usage_coequipiers", "ctx_change_coach",
                       "ctx_coach_anciennete", "ctx_coach_staff", "ctx_coach_staff_developpement",
                       "ctx_effet_coach", "ctx_effet_franchise"]
    return pairs.reset_index(drop=True), fam


def _past_effects(pairs: pd.DataFrame, key: str, name: str, y: str = "d_bpm", level: str = "bpm") -> pd.DataFrame:
    """Shrunk mean of the residual jump (change minus what level / age predict) of the
    players who had the same coach / franchise in EARLIER target seasons only."""
    lab = pairs.dropna(subset=[y, level])
    X = np.column_stack([np.ones(len(lab)), lab[level], lab["age"], lab["age"] ** 2])
    beta = np.linalg.lstsq(X, lab[y], rcond=None)[0]
    resid = pd.Series(lab[y] - X @ beta, index=lab.index)
    tab = pd.DataFrame({key: lab[key], "s": lab["start"], "r": resid}).dropna(subset=[key])
    agg = tab.groupby([key, "s"])["r"].agg(["sum", "count"]).reset_index()
    out = np.full(len(pairs), np.nan)
    for i, (k, s) in enumerate(zip(pairs[key], pairs["start"])):
        if pd.isna(k):
            continue
        past = agg[(agg[key] == k) & (agg["s"] < s)]
        out[i] = past["sum"].sum() / (past["count"].sum() + SHRINK)
    pairs[name] = out
    return pairs


def all_features(fam: dict[str, list[str]]) -> list[str]:
    return [c for f in FAMILIES for c in fam[f]]


# ===========================================================================
# MODELS & ROLLING EVALUATION
# ===========================================================================

BASELINE = ["bpm", "age", "age2", "mp", "tend_bpm_prec"]
ENSEMBLE = "Moyenne linéaire + boosting"
# target -> change column, leap column, level column, baseline features
TARGETS = {
    "bpm":    {"y": "d_bpm", "bond": "breakout", "niveau": "bpm", "base": BASELINE},
    "indice": {"y": "d_indice", "bond": "breakout_indice", "niveau": "indice",
               "base": ["indice", "age", "age2", "mp", "tend_indice_prec"]},
}


def _target(pairs: pd.DataFrame, target: str | None) -> dict:
    return TARGETS[target or pairs.attrs.get("target", "bpm")]


def _models(feats: list[str], base: list[str] = BASELINE):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    lin = lambda a: make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), Ridge(alpha=a))
    return {
        "Référence : niveau, âge, minutes (linéaire)": (base, lin(1.0)),
        "Toutes les stats (linéaire)": (feats, lin(300.0)),
        "Toutes les stats (gradient boosting)": (feats, HistGradientBoostingRegressor(
            max_depth=4, learning_rate=0.03, max_iter=400, min_samples_leaf=40,
            l2_regularization=1.0, random_state=0)),
    }


def evaluate_rolling(pairs: pd.DataFrame, feats: list[str], first_test: int = 2019,
                     target: str | None = None) -> dict:
    """Predict each target season >= first_test (start of season t+1) with a model trained on
    the earlier pairs.  Returns per-season and overall scores, and the out-of-sample predictions."""
    from scipy.stats import spearmanr
    from sklearn.base import clone
    from sklearn.metrics import roc_auc_score

    T = _target(pairs, target)
    y, bond = T["y"], T["bond"]
    lab = pairs.dropna(subset=[y])
    rows, preds = [], []
    for test_start in sorted(lab["start"].unique() + 1):
        if test_start < first_test:
            continue
        tr = lab[lab["start"] + 1 < test_start]
        te = lab[lab["start"] + 1 == test_start]
        out = te[["player_id", "season_next", T["niveau"], y, bond]].copy()
        fitted = {}
        for name, (f, model) in _models(feats, T["base"]).items():
            fitted[name] = clone(model).fit(tr[f], tr[y]).predict(te[f])
        fitted[ENSEMBLE] = (fitted["Toutes les stats (linéaire)"] + fitted["Toutes les stats (gradient boosting)"]) / 2
        for name, p in fitted.items():
            out[name] = p
            top = te.assign(p=p).nlargest(25, "p")
            rows.append({"saison testée": _season(test_start), "modèle": name, "n": len(te),
                         "RMSE": float(np.sqrt(np.mean((te[y] - p) ** 2))),
                         "Spearman": spearmanr(te[y], p).statistic,
                         "AUC bond": roc_auc_score(te[bond], p),
                         "précision top 25": top[bond].mean()})
        preds.append(out)
    per = pd.DataFrame(rows)
    overall = per.groupby("modèle")[["RMSE", "Spearman", "AUC bond", "précision top 25"]].mean()
    overall["taux de bonds"] = lab[lab["start"] + 1 >= first_test][bond].mean()
    return {"par_saison": per, "global": overall.sort_values("Spearman", ascending=False),
            "oof": pd.concat(preds)}


def univariate_signal(pairs: pd.DataFrame, feats: list[str], target: str | None = None) -> pd.DataFrame:
    """For every feature: Spearman with the jump AFTER removing what the baseline
    (level, age, minutes, previous level) already explains – i.e. the extra signal of that stat."""
    from scipy.stats import spearmanr
    T = _target(pairs, target)
    y, base = T["y"], T["base"]
    lab = pairs.dropna(subset=[y])
    X = lab[base].fillna(lab[base].median())
    X = np.column_stack([np.ones(len(lab)), X.values])
    resid = lab[y].values - X @ np.linalg.lstsq(X, lab[y].values, rcond=None)[0]
    rows = []
    for c in feats:
        v = lab[c]
        ok = v.notna().values
        if ok.sum() < 300 or v[ok].nunique() < 2:
            continue
        rows.append({"variable": c, "corr. avec le bond (brute)": spearmanr(v[ok], lab[y].values[ok]).statistic,
                     "signal en plus de la référence": spearmanr(v[ok], resid[ok]).statistic,
                     "couverture": ok.mean()})
    out = pd.DataFrame(rows)
    out["|signal|"] = out["signal en plus de la référence"].abs()
    return out.sort_values("|signal|", ascending=False).drop(columns="|signal|").reset_index(drop=True)


def family_importance(pairs: pd.DataFrame, fam: dict[str, list[str]], first_test: int = 2019,
                      n_repeats: int = 3, seed: int = 0, target: str | None = None) -> pd.DataFrame:
    """Grouped permutation importance of the gradient-boosting model, on the rolling test
    seasons: drop in Spearman when all the columns of a family are shuffled together."""
    from scipy.stats import spearmanr
    from sklearn.base import clone
    T = _target(pairs, target)
    y = T["y"]
    rng = np.random.default_rng(seed)
    feats = all_features(fam)
    name = "Toutes les stats (gradient boosting)"
    lab = pairs.dropna(subset=[y])
    drops = {f: [] for f in FAMILIES}
    for test_start in sorted(lab["start"].unique() + 1):
        if test_start < first_test:
            continue
        tr = lab[lab["start"] + 1 < test_start]
        te = lab[lab["start"] + 1 == test_start]
        m = clone(_models(feats)[name][1]).fit(tr[feats], tr[y])
        base = spearmanr(te[y], m.predict(te[feats])).statistic
        for f in FAMILIES:
            cols = [c for c in fam[f] if c in feats]
            if not cols:
                continue
            for _ in range(n_repeats):
                Xp = te[feats].copy()
                Xp[cols] = Xp[cols].values[rng.permutation(len(Xp))]
                drops[f].append(base - spearmanr(te[y], m.predict(Xp)).statistic)
    return (pd.DataFrame({"famille": list(drops), "baisse de Spearman": [np.mean(v) if v else np.nan for v in drops.values()],
                          "variables": [len(fam[f]) for f in drops]})
            .sort_values("baisse de Spearman", ascending=False).reset_index(drop=True))


def predict_next(pairs: pd.DataFrame, feats: list[str], model_name: str = None,
                 target: str | None = None) -> pd.DataFrame:
    """Fit on every complete pair, predict the coming season (pairs with season t = last complete).
    Default: average of the linear and gradient-boosting models (best in the rolling test).
    Output columns: d_<level>_predit and <level>_predit."""
    T = _target(pairs, target)
    y, lvl = T["y"], T["niveau"]
    models = _models(feats, T["base"])
    names = [model_name] if model_name else ["Toutes les stats (linéaire)", "Toutes les stats (gradient boosting)"]
    lab = pairs.dropna(subset=[y])
    last = int(LAST_COMPLETE_SEASON[:4])
    nxt = pairs[pairs["start"] == last].copy()
    preds = []
    for n in names:
        f, model = models[n]
        preds.append(model.fit(lab[f], lab[y]).predict(nxt[f]))
    nxt[f"d_{lvl}_predit"] = np.mean(preds, axis=0)
    nxt[f"{lvl}_predit"] = nxt[lvl] + nxt[f"d_{lvl}_predit"]
    return nxt


# ===========================================================================
# PIPELINE STEP
# ===========================================================================

def run(db_path: str = DB_PATH) -> None:
    pairs, fam = build_pairs(db_path)
    feats = all_features(fam)
    lab = pairs.dropna(subset=["d_bpm"])
    logger.info(f"✅ {len(lab)} season pairs (t → t+1, >= {MIN_MP} min both seasons), "
                f"{lab['breakout'].mean():.0%} leaps (ΔBPM >= {BREAKOUT})")
    res = evaluate_rolling(pairs, feats)
    print("\n📈 Rolling temporal validation (each season predicted from the earlier ones)\n")
    print(res["global"].round(3).to_string())
    nxt = predict_next(pairs, feats)
    # same model on the success index (indice.py); effects of coach / franchise re-estimated on it
    pairs_ix, _ = build_pairs(db_path, target="indice")
    res_ix = evaluate_rolling(pairs_ix, feats)
    print("\n📈 Same validation, target = change of the success index\n")
    print(res_ix["global"].round(3).to_string())
    nx_ix = predict_next(pairs_ix, feats)[["player_id", "d_indice_predit", "indice_predit"]]
    nxt = nxt.merge(nx_ix, on="player_id", how="left")
    conn = get_connection(db_path)
    names = pd.read_sql_query("SELECT player_id, full_name FROM players", conn)
    out = nxt.merge(names, on="player_id", how="left")[
        ["player_id", "full_name", "season_next", "age", "ctx_equipe", "ctx_change_equipe",
         "ctx_change_coach", "mp", "bpm", "d_bpm_predit", "bpm_predit",
         "indice", "ix_efficacite_off", "ix_efficacite_def", "ix_production",
         "d_indice_predit", "indice_predit"]]
    out.to_sql("progression_predictions", conn, if_exists="replace", index=False)
    conn.close()
    print(f"\n🔮 {len(out)} predictions for {out['season_next'].iloc[0]} → table progression_predictions")
    print(out.sort_values("d_bpm_predit", ascending=False).head(15).round(2).to_string(index=False))
