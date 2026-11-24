"""
modeling.py – Prospect table + first predictive model.

One row per drafted player (draft years 2014-2021 by default):
  features : draft pick, age at draft (college players, Barttorvik), origin, position, size,
             Draft Combine measurements (anthropometry + athletic drills)
  target   : minutes played in the first WINDOW NBA seasons, counted from the NBA debut
             (from player_career_stats)
             → `etabli` = 1 if minutes >= MIN_THRESHOLD (≈ 1000 min / season = rotation player)
  quality  : in the same window, summed VORP and minute-weighted BPM (Basketball-Reference)
             and PIE (NBA); BPM / PIE only for players with >= QUALITY_MIN_MINUTES minutes

Usage
-----
  python main.py --step model            # build table `modeling_table` + print evaluation

  from modeling import build_modeling_table, evaluate
  df = build_modeling_table()
  res = evaluate(df)
"""

import logging

import numpy as np
import pandas as pd

from config import DB_PATH, LAST_COMPLETE_SEASON
from database import get_connection
from fetch_college import POWER_CONFS

logger = logging.getLogger(__name__)

DRAFT_YEARS = (2014, 2021)   # cohorts with WINDOW full seasons observed
WINDOW = 4                   # seasons after the draft used for the target
MIN_THRESHOLD = 4000         # minutes in WINDOW seasons → "established" player
QUALITY_MIN_MINUTES = 1000  # BPM / PIE targets only for players with at least these minutes
LAST_SEASON = int(LAST_COMPLETE_SEASON[:4])   # last complete season (2025 → 2025-26)

COMBINE_FEATURES = [
    "wingspan", "standing_reach", "hand_length", "hand_width", "body_fat_pct",
    "envergure_moins_taille",
    "standing_vertical_leap", "max_vertical_leap", "lane_agility_time", "three_quarter_sprint",
]
# age_draft: birth date from Barttorvik (college players) or Wikidata (others) – table
# draft_birthdates, sources that do not depend on reaching the NBA.  NEVER use players.birthdate
# (CommonPlayerInfo exists only for players who reached the NBA → missing age = never played).
BASE_FEATURES = ["pick", "log_pick", "age_draft", "taille", "poids", "a_fait_combine",
                 "origine_universite", "origine_international", "origine_lycee",
                 "poste_arriere", "poste_ailier", "poste_interieur"]
# NCAA (Barttorvik): last college season before the draft + whole college career
COLLEGE_FEATURES = [
    "ncaa_bpm", "ncaa_obpm", "ncaa_dbpm", "ncaa_ptspg", "ncaa_rebpg", "ncaa_astpg",
    "ncaa_stlpg", "ncaa_blkpg", "ncaa_mpg", "ncaa_usg", "ncaa_ts_pct", "ncaa_ast_pct",
    "ncaa_tov_pct", "ncaa_orb_pct", "ncaa_drb_pct", "ncaa_blk_pct", "ncaa_stl_pct",
    "ncaa_fg3_rate", "ncaa_fg3_pct", "ncaa_ft_pct_carriere", "ncaa_saisons",
    "ncaa_conf_majeure", "ncaa_rec_rank", "ncaa_barthag", "ncaa_sos",
]
FEATURES = BASE_FEATURES + COMBINE_FEATURES + COLLEGE_FEATURES
# draft rank enters as pick AND log(pick): log fits the top of the draft, raw the late picks
RANK = ["pick", "log_pick"]
NCAA_SET = ["pick", "log_pick", "age_draft", "origine_universite"] + COLLEGE_FEATURES


def _wavg(values: pd.Series, weights: pd.Series) -> float:
    ok = values.notna() & weights.notna() & (weights > 0)
    return float(np.average(values[ok], weights=weights[ok])) if ok.any() else np.nan


def _season(start_year: int) -> str:
    return f"{start_year}-{str(start_year + 1)[-2:]}"


# ===========================================================================
# BUILD TABLE
# ===========================================================================

def build_modeling_table(db_path: str = DB_PATH,
                         draft_years: tuple[int, int] = DRAFT_YEARS,
                         window: int = WINDOW,
                         window_start: str = "debut",
                         include_unlabeled: bool = False) -> pd.DataFrame:
    """
    One row per drafted player.  With *include_unlabeled* the most recent drafts
    (target not observable yet) are kept too, with `etabli` = NaN.
    *window_start*: "debut" (first NBA season, default) or "draft" (draft year).
    """
    conn = get_connection(db_path)
    q = lambda sql, *p: pd.read_sql_query(sql, conn, params=p)
    last_year = 2100 if include_unlabeled else draft_years[1]

    df = q("""
        SELECT h.player_id, h.player_name, CAST(h.season AS INT) AS draft_year,
               h.overall_pick AS pick, h.organization_type,
               b.dob AS birthdate, b.source AS source_age
        FROM draft_history h LEFT JOIN draft_birthdates b USING (player_id)
        WHERE CAST(h.season AS INT) BETWEEN ? AND ?""", draft_years[0], last_year)

    # ── basic features ──────────────────────────────────────────────────────
    draft_day = pd.to_datetime(df["draft_year"].astype(str) + "-06-25")
    df["age_draft"] = (draft_day - pd.to_datetime(df["birthdate"], errors="coerce")).dt.days / 365.25
    # ~2 % without a birth date, a bit more often among never-played internationals: fill with
    # the median of the same origin so that no "age missing" indicator can carry that signal
    df["age_draft"] = df["age_draft"].fillna(
        df.groupby(df["organization_type"].fillna(""))["age_draft"].transform("median")).fillna(df["age_draft"].median())
    org = df["organization_type"].fillna("")
    df["origine_universite"] = (org == "College/University").astype(int)
    df["origine_international"] = (org == "Other Team/Club").astype(int)
    df["origine_lycee"] = (org == "High School").astype(int)
    df["log_pick"] = np.log(df["pick"])

    # ── Draft Combine (latest combine at or before the draft year) ──────────
    comb = q("""
        SELECT a.player_id, CAST(substr(a.season, 1, 4) AS INT) AS combine_year, a.position AS position_combine,
               a.height_wo_shoes, a.weight AS weight_combine, a.wingspan, a.standing_reach, a.hand_length, a.hand_width, a.body_fat_pct,
               d.standing_vertical_leap, d.max_vertical_leap, d.lane_agility_time, d.three_quarter_sprint
        FROM draft_combine_anthro a
        LEFT JOIN draft_combine_drills d USING (player_id, season)""")
    comb = comb.merge(df[["player_id", "draft_year"]], on="player_id")
    comb = (comb[comb["combine_year"] <= comb["draft_year"]]
            .sort_values("combine_year").groupby("player_id").tail(1)
            .drop(columns=["draft_year", "combine_year"]))
    for c in comb.columns.drop(["player_id", "position_combine"]):
        comb[c] = pd.to_numeric(comb[c], errors="coerce").replace(0, np.nan)
    comb["envergure_moins_taille"] = comb["wingspan"] - comb["height_wo_shoes"]
    df = df.merge(comb, on="player_id", how="left")
    df["a_fait_combine"] = df["wingspan"].notna().astype(int)
    # Size and position come from the Combine ONLY: the NBA listing (CommonPlayerInfo)
    # exists only for players who reached the NBA, so using it would leak the target.
    df["taille"] = df["height_wo_shoes"]
    df["poids"] = df["weight_combine"]

    # ── NCAA (Barttorvik) – only seasons up to the draft (draft_college) ────
    col = q("""
        SELECT dc.player_id, cs.year, cs.conf, cs.gp, cs.mpg, cs.usg, cs.ts_pct, cs.ast_pct, cs.tov_pct,
               cs.orb_pct, cs.drb_pct, cs.blk_pct, cs.stl_pct, cs.bpm, cs.obpm, cs.dbpm,
               cs.ptspg, cs.rebpg, cs.astpg, cs.stlpg, cs.blkpg, cs.ftm, cs.fta,
               cs.fg2a, cs.fg3a, cs.fg3_pct, cs.rec_rank, ct.barthag, ct.sos
        FROM draft_college dc
        JOIN college_stats cs ON cs.pid = dc.pid
                             AND cs.year BETWEEN dc.first_college_year AND dc.last_college_year
        LEFT JOIN college_teams ct ON ct.year = cs.year AND ct.team = cs.team""")
    if not col.empty:
        g = col.groupby("player_id")
        career_ncaa = pd.DataFrame({
            "ncaa_saisons": g["year"].nunique(),
            "ncaa_ft_pct_carriere": g["ftm"].sum() / g["fta"].sum().replace(0, np.nan),
            "ncaa_rec_rank": g["rec_rank"].max(),
        })
        last = col.sort_values(["year", "gp"]).groupby("player_id").tail(1).set_index("player_id")
        last["ncaa_fg3_rate"] = last["fg3a"] / (last["fg2a"] + last["fg3a"]).replace(0, np.nan)
        last["ncaa_conf_majeure"] = last["conf"].isin(POWER_CONFS).astype(int)
        keep = ["bpm", "obpm", "dbpm", "ptspg", "rebpg", "astpg", "stlpg", "blkpg", "mpg", "usg",
                "ts_pct", "ast_pct", "tov_pct", "orb_pct", "drb_pct", "blk_pct", "stl_pct", "fg3_pct",
                "barthag", "sos"]
        last = last.rename(columns={c: f"ncaa_{c}" for c in keep})
        ncaa = last[[f"ncaa_{c}" for c in keep] + ["ncaa_fg3_rate", "ncaa_conf_majeure"]].join(career_ncaa)
        df = df.merge(ncaa, left_on="player_id", right_index=True, how="left")
    for c in COLLEGE_FEATURES:
        if c not in df:
            df[c] = np.nan
    df["ncaa_conf_majeure"] = df["ncaa_conf_majeure"].fillna(0).astype(int)

    # ── position group (Combine listing) ────────────────────────────────────
    poste = df["position_combine"].fillna("").str.split("-").str[0].map(
        {"PG": "arriere", "SG": "arriere", "SF": "ailier", "PF": "interieur", "C": "interieur"})
    for p in ("arriere", "ailier", "interieur"):
        df[f"poste_{p}"] = (poste == p).astype(int)

    # ── target: minutes in the first `window` NBA seasons ──────────────────
    # window_start="draft": seasons draft_year … draft_year+window-1
    # window_start="debut": seasons from the first NBA season played (draft-and-stash
    #                       players such as Jokic are not penalised for staying in Europe)
    career = q("SELECT player_id, season_id, min, gp FROM player_career_stats WHERE team_id != 0")
    career["start"] = career["season_id"].str[:4].astype(int)
    career = career.merge(df[["player_id", "draft_year"]], on="player_id")
    career = career[(career["start"] >= career["draft_year"]) & (career["start"] <= LAST_SEASON)]
    debut = career[career["gp"] > 0].groupby("player_id")["start"].min().rename("debut_year")
    df = df.merge(debut, on="player_id", how="left")
    df["window_start"] = df["draft_year"] if window_start == "draft" else df["debut_year"].fillna(df["draft_year"])
    career = career.merge(df[["player_id", "window_start"]], on="player_id")
    career = career[(career["start"] >= career["window_start"]) & (career["start"] < career["window_start"] + window)]
    tgt = career.groupby("player_id").agg(minutes=("min", "sum"), matchs=("gp", "sum"))
    df = df.merge(tgt, on="player_id", how="left")

    # ── quality targets in the same window ─────────────────────────────────
    # vorp : summed VORP (Basketball-Reference) – quality × playing time
    # bpm  : BPM weighted by minutes (Basketball-Reference) – quality only
    # pie  : PIE weighted by minutes (NBA)
    adv = q("SELECT player_id, season, mp, bpm, vorp FROM bbref_advanced WHERE player_id IS NOT NULL")
    adv = adv.sort_values("mp").groupby(["player_id", "season"]).tail(1)   # traded: combined row
    pie = q("SELECT player_id, season, pie, gp * min AS mp_nba FROM player_stats_advanced")
    adv = adv.merge(pie, on=["player_id", "season"], how="outer")
    adv["start"] = adv["season"].str[:4].astype(int)
    adv = adv[adv["start"] <= LAST_SEASON]                         # season in progress excluded
    adv = adv.merge(df[["player_id", "window_start"]], on="player_id")
    adv = adv[(adv["start"] >= adv["window_start"]) & (adv["start"] < adv["window_start"] + window)]
    g = adv.groupby("player_id")
    qual = pd.DataFrame({"vorp": g["vorp"].sum(min_count=1),
                         "bpm": g.apply(lambda x: _wavg(x["bpm"], x["mp"])),
                         "pie": g.apply(lambda x: _wavg(x["pie"], x["mp_nba"]))})
    df = df.merge(qual, left_on="player_id", right_index=True, how="left")

    # A missing career row means 0 minutes ONLY if every season since the draft is in
    # player_stats_traditional (the player never appears) and no career fetch was missed.
    covered = set(q("SELECT DISTINCT season FROM player_stats_traditional")["season"])
    played = set(q("SELECT DISTINCT player_id FROM player_stats_traditional")["player_id"])
    has_career = set(q("SELECT DISTINCT player_id FROM player_career_stats")["player_id"])
    conn.close()

    def _label(row):
        if row["draft_year"] > draft_years[1]:
            return np.nan, np.nan                       # draft too recent
        if pd.isna(row["minutes"]):
            draft_window = {_season(row["draft_year"] + k) for k in range(window)}
            if not draft_window <= covered:
                return np.nan, np.nan                   # seasons not downloaded
            if row["player_id"] in played and row["player_id"] not in has_career:
                return np.nan, np.nan                   # career fetch failed / missing
            return 0.0, 0.0                             # drafted, never played
        complete = row["window_start"] + window - 1 <= LAST_SEASON
        etabli = float(row["minutes"] >= MIN_THRESHOLD)
        if complete:
            return row["minutes"], etabli
        # late debut: window not over → outcome known only if the threshold is already reached
        return np.nan, (1.0 if etabli else np.nan)
    df["minutes_obs"] = df["minutes"].fillna(0)   # minutes in the window observed so far
    labels = df.apply(_label, axis=1, result_type="expand")
    df["minutes"], df["etabli"] = labels[0], labels[1]
    df["matchs"] = np.where(df["minutes"] == 0, 0, df["matchs"])
    # quality targets: known when the minutes are; VORP = 0 for a player who never played;
    # BPM / PIE only make sense with enough minutes
    df["vorp"] = np.where(df["minutes"].isna(), np.nan, np.where(df["minutes"] == 0, 0.0, df["vorp"].fillna(0)))
    for c in ("bpm", "pie"):
        df[c] = df[c].where(df["minutes"] >= QUALITY_MIN_MINUTES)

    # value of the pick: VORP minus the VORP expected at that draft slot (decreasing
    # isotonic curve of VORP on the pick) → > 0 = better than his draft position
    from sklearn.isotonic import IsotonicRegression
    known = df["vorp"].notna()
    if known.sum() > 20:
        iso = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(df.loc[known, "pick"], df.loc[known, "vorp"])
        df["vorp_attendu"] = iso.predict(df["pick"])
    else:
        df["vorp_attendu"] = np.nan
    df["valeur_vs_rang"] = df["vorp"] - df["vorp_attendu"]

    cols = ["player_id", "player_name", "draft_year", "organization_type", "debut_year", "minutes", "minutes_obs", "matchs", "etabli", "vorp", "bpm", "pie", "vorp_attendu", "valeur_vs_rang", "source_age"] + FEATURES
    return df[cols].sort_values(["draft_year", "pick"]).reset_index(drop=True)


# ===========================================================================
# MODEL
# ===========================================================================

def _models():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        # baseline: draft position only
        "Rang de draft seul (logistique)": (RANK, make_pipeline(StandardScaler(), LogisticRegression())),
        "Rang + âge + NCAA (logistique)": (NCAA_SET, make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
            LogisticRegression(C=0.3, max_iter=2000))),
        "Toutes variables (logistique)": (FEATURES, make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
            LogisticRegression(C=0.3, max_iter=2000))),
        "Toutes variables (gradient boosting)": (FEATURES, HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=20,
            l2_regularization=1.0, random_state=0)),
    }


def evaluate(df: pd.DataFrame, n_repeats: int = 5, seed: int = 0) -> dict:
    """
    Repeated stratified 5-fold CV.  Returns
      scores      : DataFrame model × (AUC mean / std, Brier)
      importance  : permutation importance (AUC drop) of the full logistic model
      oof         : out-of-fold probability of each player for every model
    """
    from sklearn.base import clone
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import brier_score_loss, roc_auc_score
    from sklearn.model_selection import RepeatedStratifiedKFold

    data = df.dropna(subset=["etabli"]).reset_index(drop=True)
    y = data["etabli"].astype(int).values
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=seed)

    rows, oof, imp = [], {}, []
    for name, (feats, model) in _models().items():
        X = data[feats]
        aucs, briers = [], []
        proba = np.zeros(len(data))
        for k, (tr, te) in enumerate(cv.split(X, y)):
            m = clone(model).fit(X.iloc[tr], y[tr])
            p = m.predict_proba(X.iloc[te])[:, 1]
            aucs.append(roc_auc_score(y[te], p)); briers.append(brier_score_loss(y[te], p))
            proba[te] += p / n_repeats
            if name == "Toutes variables (logistique)" and k < 5:   # importance on first repeat
                r = permutation_importance(m, X.iloc[te], y[te], scoring="roc_auc",
                                           n_repeats=10, random_state=seed)
                imp.append(r.importances_mean)
        rows.append({"modèle": name, "AUC": np.mean(aucs), "AUC écart-type": np.std(aucs),
                     "Brier": np.mean(briers)})
        oof[name] = proba

    importance = (pd.Series(np.mean(imp, axis=0), index=FEATURES)
                  .sort_values(ascending=False).rename("baisse d'AUC"))
    out = data[["player_id", "player_name", "draft_year", "pick", "minutes", "etabli"]].copy()
    for name, p in oof.items():
        out[name] = p
    return {"scores": pd.DataFrame(rows).set_index("modèle"), "importance": importance,
            "oof": out, "n": len(data), "taux_etablis": y.mean()}


def evaluate_temporal(df: pd.DataFrame, train_until: int = 2019) -> pd.DataFrame:
    """
    Train on drafts <= *train_until*, test on the later drafts – a real
    "predict the future" setting.  Returns AUC / Brier per model on the test drafts.
    """
    from sklearn.metrics import brier_score_loss, roc_auc_score

    data = df.dropna(subset=["etabli"])
    train, test = data[data["draft_year"] <= train_until], data[data["draft_year"] > train_until]
    rows = []
    for name, (feats, model) in _models().items():
        model.fit(train[feats], train["etabli"].astype(int))
        p = model.predict_proba(test[feats])[:, 1]
        rows.append({"modèle": name, "AUC test": roc_auc_score(test["etabli"], p),
                     "Brier test": brier_score_loss(test["etabli"], p)})
    out = pd.DataFrame(rows).set_index("modèle")
    out.attrs.update(n_train=len(train), n_test=len(test),
                     test_years=f"{test['draft_year'].min()}-{test['draft_year'].max()}")
    return out


def _regressors():
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "Rang de draft seul (linéaire)": (RANK, make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
        "Rang + âge + NCAA (linéaire)": (NCAA_SET, make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), Ridge(alpha=10.0))),
        "Toutes variables (linéaire)": (FEATURES, make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), Ridge(alpha=10.0))),
        "Toutes variables (gradient boosting)": (FEATURES, HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=20,
            l2_regularization=1.0, random_state=0)),
    }


TARGETS = {
    "minutes": "log(1 + minutes) sur la fenêtre",
    "vorp": "VORP cumulé sur la fenêtre (0 si jamais joué)",
    "bpm": f"BPM pondéré par les minutes (joueurs >= {QUALITY_MIN_MINUTES} min)",
    "pie": f"PIE pondéré par les minutes (joueurs >= {QUALITY_MIN_MINUTES} min)",
    "valeur_vs_rang": "VORP moins le VORP attendu à ce rang de draft",
}


def evaluate_regression(df: pd.DataFrame, target: str = "minutes",
                        n_repeats: int = 5, seed: int = 0) -> pd.DataFrame:
    """
    Continuous target (see TARGETS), repeated 5-fold CV.  "minutes" is modelled as
    log(1 + minutes).  R² on that scale; Spearman = how well the model *ranks* the players.
    """
    from scipy.stats import spearmanr
    from sklearn.base import clone
    from sklearn.metrics import r2_score
    from sklearn.model_selection import RepeatedKFold

    data = df.dropna(subset=[target]).reset_index(drop=True)
    y = data[target].values.astype(float)
    if target == "minutes":
        y = np.log1p(y)
    cv = RepeatedKFold(n_splits=5, n_repeats=n_repeats, random_state=seed)
    rows = []
    for name, (feats, model) in _regressors().items():
        r2, rho = [], []
        for tr, te in cv.split(data):
            m = clone(model).fit(data[feats].iloc[tr], y[tr])
            p = m.predict(data[feats].iloc[te])
            r2.append(r2_score(y[te], p)); rho.append(spearmanr(y[te], p).statistic)
        rows.append({"modèle": name, "R²": np.mean(r2), "Spearman": np.mean(rho),
                     "Spearman écart-type": np.std(rho)})
    out = pd.DataFrame(rows).set_index("modèle")
    out.attrs.update(n=len(data), target=TARGETS[target])
    return out


def predict_recent_value(df_all: pd.DataFrame, target: str = "valeur_vs_rang",
                         model_name: str = "Rang + âge + NCAA (linéaire)") -> pd.DataFrame:
    """Fit a regressor on the labeled players, predict *target* for the recent drafts."""
    feats, model = _regressors()[model_name]
    train = df_all.dropna(subset=[target])
    model.fit(train[feats], train[target])
    recent = df_all[df_all["draft_year"] > DRAFT_YEARS[1]].copy()
    recent[f"{target}_predit"] = model.predict(recent[feats])
    return recent


def predict_recent(df_all: pd.DataFrame, model_name: str = "Toutes variables (logistique)") -> pd.DataFrame:
    """Fit on labeled players, score the drafts whose outcome is not known yet."""
    feats, model = _models()[model_name]
    train = df_all.dropna(subset=["etabli"])
    model.fit(train[feats], train["etabli"].astype(int))
    recent = df_all[df_all["draft_year"] > DRAFT_YEARS[1]].copy()
    recent["proba_etabli"] = model.predict_proba(recent[feats])[:, 1]
    return recent


# ===========================================================================
# PIPELINE STEP
# ===========================================================================

def run(db_path: str = DB_PATH) -> None:
    df = build_modeling_table(db_path, include_unlabeled=True)
    conn = get_connection(db_path)
    df.to_sql("modeling_table", conn, if_exists="replace", index=False)
    conn.close()
    labeled = df.dropna(subset=["etabli"])
    logger.info(f"✅ modeling_table: {len(df)} drafted players, {len(labeled)} with a known outcome "
                f"({labeled['etabli'].mean():.0%} established).")

    res = evaluate(df)
    print(f"\n📈 Model evaluation ({res['n']} players, {res['taux_etablis']:.0%} established, "
          f"repeated 5-fold CV)\n")
    print(res["scores"].round(3).to_string())
    print("\nTop features (permutation importance, full logistic model):")
    print(res["importance"].head(10).round(4).to_string())

    tmp = evaluate_temporal(df)
    print(f"\n⏳ Temporal validation: train {DRAFT_YEARS[0]}-2019 ({tmp.attrs['n_train']} players), "
          f"test {tmp.attrs['test_years']} ({tmp.attrs['n_test']} players)\n")
    print(tmp.round(3).to_string())

    for target in TARGETS:
        reg = evaluate_regression(df, target)
        print(f"\n📏 Continuous target: {reg.attrs['target']} ({reg.attrs['n']} players, repeated 5-fold CV)\n")
        print(reg.round(3).to_string())
