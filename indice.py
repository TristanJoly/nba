"""
indice.py – "Indice de réussite": a per-minute success index of a player-season.

Three pillars, each the mean of season-standardised components (z-scores weighted by
minutes, so the reference is the average NBA minute of that season):

  efficacite_off : scoring efficiency × volume (points added vs league TS%, with the 3P%
                   shrunk toward the league mean – 3P% is mostly noise on small samples),
                   offensive rating on court, ball security (team turnover share, inverted)
  efficacite_def : defensive rating on court (inverted), steals, blocks, defensive
                   rebound %, deflections
  production     : raw box score per 36 minutes – points, rebounds, assists, turnovers (inverted)

indice = mean of the three pillars, put on a BPM-like scale (0 = average minute,
same spread as BPM among players with >= 500 minutes).  It is a RATE: two players with
the same index produce the same per minute, whatever their playing time.
indice_volume = indice × minutes / average minutes of the season (rate × playing time).
"""

import numpy as np
import pandas as pd

REF_MIN = 250          # players used to set the season reference (mean / SD)
SHRINK_3PA = 250       # pseudo-attempts at league 3P% added to every shooter
CLIP = 3.0             # z-scores clipped at ±3

PILLARS = {
    "efficacite_off": {"ix_ts_ajoutes36": 1, "adv_off_rating": 1, "adv_tm_tov_pct": -1},
    "efficacite_def": {"adv_def_rating": -1, "box_stl36": 1, "box_blk36": 1, "adv_dreb_pct": 1,
                       "hus_deflections36": 1},
    "production":     {"box_pts36": 1, "box_reb36": 1, "box_ast36": 1, "box_tov36": -1},
}
INDEX_COLS = ["indice", "indice_volume"] + [f"ix_{p}" for p in PILLARS]


def _wz(x: pd.Series, w: pd.Series, ref: pd.Series) -> pd.Series:
    """z-score with the minute-weighted mean / SD of the reference players."""
    ok = ref & x.notna()
    if ok.sum() < 20:
        return pd.Series(np.nan, index=x.index)
    m = np.average(x[ok], weights=w[ok])
    sd = np.sqrt(np.average((x[ok] - m) ** 2, weights=w[ok]))
    return ((x - m) / sd).clip(-CLIP, CLIP)


def add_index(p: pd.DataFrame) -> pd.DataFrame:
    """Adds the index columns to a player-season panel (progression.load_panel)."""
    p = p.copy()
    mp = p["mp_nba"]
    fg3a, fg3m = p["box_fg3a36"], p["box_fg3m36"]
    fga, fta, ftm, fgm = p["box_fga36"], p["box_fta36"], p["box_ftm36"], p["box_fgm36"]
    out = []
    for _, g in p.groupby("season"):
        ref = g["mp_nba"] >= REF_MIN
        w = g["mp_nba"]
        lg3 = (g.loc[ref, "box_fg3m36"] * w[ref]).sum() / (g.loc[ref, "box_fg3a36"] * w[ref]).sum()
        # 3P made recomputed with a shrunk 3P% (attempts of the whole season, not per 36)
        att = g["box_fg3a36"] * w / 36
        p3 = (g["box_fg3m36"] * w / 36 + SHRINK_3PA * lg3) / (att + SHRINK_3PA)
        fg3m_adj = g["box_fg3a36"] * p3
        pts = 2 * (g["box_fgm36"] - g["box_fg3m36"]) + 3 * fg3m_adj + g["box_ftm36"]
        tsa = g["box_fga36"] + 0.44 * g["box_fta36"]
        lg_ts = (pts[ref] * w[ref]).sum() / (2 * (tsa[ref] * w[ref]).sum())
        g = g.assign(ix_ts_ajoutes36=pts - 2 * lg_ts * tsa)
        z = {}
        for pil, comps in PILLARS.items():
            cols = [sign * _wz(g[c], w, ref) for c, sign in comps.items() if c in g]
            z[f"ix_{pil}"] = pd.concat(cols, axis=1).mean(axis=1, skipna=True)
        zz = pd.DataFrame(z, index=g.index)
        zz["ix_ts_ajoutes36"] = g["ix_ts_ajoutes36"]
        zz["raw"] = zz[[f"ix_{q}" for q in PILLARS]].mean(axis=1)
        zz["mp_rel"] = w / w[ref].mean()
        out.append(zz)
    z = pd.concat(out)
    # BPM-like scale, fixed on the pooled players with >= 500 minutes
    big = (mp >= 500) & p["bpm"].notna()
    m_raw, sd_raw = np.average(z.loc[big, "raw"], weights=mp[big]), z.loc[big, "raw"].std()
    sd_bpm = p.loc[big, "bpm"].std()
    p["ix_ts_ajoutes36"] = z["ix_ts_ajoutes36"]
    for q in PILLARS:
        p[f"ix_{q}"] = z[f"ix_{q}"]
    p["indice"] = (z["raw"] - m_raw) / sd_raw * sd_bpm
    p["indice_volume"] = p["indice"] * z["mp_rel"]
    return p
