"""Metric tree and conversion-rate attribution for the UCI Online Shoppers data.

The SQL files in ../sql hold the analysis logic (DuckDB). This module runs
them, adds an independent pandas/numpy implementation of the shift-share
decomposition (used to cross-check the SQL), a multinomial bootstrap for
uncertainty, and a plain-text driver summary.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SQL_DIR = ROOT / "sql"
DIMENSIONS = ["visitor_type", "traffic_source", "intent", "weekend", "region"]
MONTH_ORDER = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# --------------------------------------------------------------------------- SQL
def connect(csv_path: str | Path) -> duckdb.DuckDBPyConnection:
    """Create an in-memory DuckDB database and run all SQL files in order."""
    con = duckdb.connect()
    for sql_file in sorted(SQL_DIR.glob("*.sql")):
        sql = sql_file.read_text(encoding="utf-8").replace(
            "{csv_path}", Path(csv_path).as_posix())
        con.execute(sql)
    return con


def _months(m) -> list[str]:
    return [m] if isinstance(m, str) else list(m)


def label(m) -> str:
    m = _months(m)
    return m[0] if len(m) == 1 else "+".join(m)


def shift_share_sql(con, dimension: str, base, comp) -> pd.DataFrame:
    """base / comp: one month label or a list of months (pooled period)."""
    return con.execute("SELECT * FROM shift_share(?, ?, ?)",
                       [dimension, _months(base), _months(comp)]).df()


def summary_sql(con, base, comp) -> pd.DataFrame:
    return con.execute("SELECT * FROM shift_share_summary(?, ?)",
                       [_months(base), _months(comp)]).df()


# ------------------------------------------------------- reference implementation
def shift_share_np(n0, k0, n1, k1):
    """Vectorised symmetric shift-share. Inputs are per-segment arrays; the
    last axis indexes segments, so leading axes can hold bootstrap draws.

    Returns (mix_effect, rate_effect) per segment with the same shape.
    """
    n0, k0, n1, k1 = (np.asarray(a, dtype=float) for a in (n0, k0, n1, k1))
    N0, N1 = n0.sum(-1, keepdims=True), n1.sum(-1, keepdims=True)
    w0, w1 = n0 / N0, n1 / N1
    with np.errstate(invalid="ignore", divide="ignore"):
        r0_obs, r1_obs = k0 / n0, k1 / n1
    r0 = np.where(n0 > 0, r0_obs, r1_obs)
    r1 = np.where(n1 > 0, r1_obs, r0_obs)
    r0, r1 = np.nan_to_num(r0), np.nan_to_num(r1)  # segment empty in both
    cvr0 = k0.sum(-1, keepdims=True) / N0
    cvr1 = k1.sum(-1, keepdims=True) / N1
    mix = (w1 - w0) * ((r0 + r1) / 2 - (cvr0 + cvr1) / 2)
    rate = (r1 - r0) * (w0 + w1) / 2
    return mix, rate


def cells(con, dimension: str, base, comp) -> pd.DataFrame:
    """Per-segment counts for two periods, computed in pandas from the
    session-level table (independent of the SQL macro)."""
    d = con.execute("SELECT * FROM session_dims WHERE dimension = ?", [dimension]).df()
    out = {}
    for tag, months in (("0", _months(base)), ("1", _months(comp))):
        g = d[d.month_raw.isin(months)].groupby("segment").purchase.agg(["size", "sum"])
        out["n" + tag], out["k" + tag] = g["size"], g["sum"]
    return pd.DataFrame(out).fillna(0).astype(int).sort_index()


def bootstrap_summary(con, base, comp, reps: int = 2000,
                      seed: int = 42) -> pd.DataFrame:
    """Multinomial bootstrap of sessions within each month.

    Each month's sessions are resampled with replacement (fixed month total),
    which is equivalent to a multinomial draw over (segment x purchase) cells.
    The same draws are reused for every dimension of a month by resampling
    session rows, so dimension summaries are comparable.
    """
    rng = np.random.default_rng(seed)
    periods = {"base": _months(base), "comp": _months(comp)}
    sess = con.execute(
        "SELECT session_id, month_raw, dimension, segment, purchase FROM session_dims").df()
    sess["period"] = np.select([sess.month_raw.isin(periods["base"]),
                                sess.month_raw.isin(periods["comp"])], ["base", "comp"], "")
    sess = sess[sess.period != ""]
    base, comp = "base", "comp"
    rows = []
    # Resampling is stratified by month, so pooled periods keep their month mix.
    month_ids = {m: np.sort(sess.loc[sess.period == m, "session_id"].unique())
                 for m in (base, comp)}
    draws = {m: rng.integers(0, len(ids), size=(reps, len(ids)))
             for m, ids in month_ids.items()}
    for dim in DIMENSIONS:
        d = sess[sess.dimension == dim]
        segs = np.sort(d.segment.unique())
        seg_code = {m: pd.Categorical(
            d[d.period == m].set_index("session_id").loc[month_ids[m], "segment"],
            categories=segs).codes for m in (base, comp)}
        y = {m: d[d.period == m].set_index("session_id").loc[month_ids[m], "purchase"]
             .to_numpy() for m in (base, comp)}
        counts = {}
        for m in (base, comp):
            idx = draws[m]
            s, yy = seg_code[m][idx], y[m][idx]
            n = np.stack([(s == j).sum(1) for j in range(len(segs))], -1)
            k = np.stack([((s == j) & (yy == 1)).sum(1) for j in range(len(segs))], -1)
            counts[m] = (n, k)
        mix, rate = shift_share_np(*counts[base], *counts[comp])
        mix_t, rate_t = mix.sum(-1), rate.sum(-1)
        rows.append({"dimension": dim,
                     "mix_lo": np.percentile(mix_t, 2.5), "mix_hi": np.percentile(mix_t, 97.5),
                     "rate_lo": np.percentile(rate_t, 2.5), "rate_hi": np.percentile(rate_t, 97.5),
                     "delta_lo": np.percentile(mix_t + rate_t, 2.5),
                     "delta_hi": np.percentile(mix_t + rate_t, 97.5)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ narrative
def period_cvr(con, months) -> float:
    return con.execute("SELECT sum(purchases) / sum(sessions) FROM kpi_month "
                       "WHERE list_contains(?, month_raw)", [_months(months)]).fetchone()[0]


def driver_summary(con, base, comp, top_n: int = 3) -> str:
    """Deterministic text summary of the main drivers (input for a later LLM step)."""
    summ = summary_sql(con, base, comp)
    c0, c1 = period_cvr(con, base), period_cvr(con, comp)
    lines = [f"CVR {label(base)} {c0:.2%} -> {label(comp)} {c1:.2%} "
             f"({(c1 - c0) * 100:+.2f} pp)."]
    top_mix = summ.iloc[summ.mix_effect.abs().argmax()]
    lines.append(
        f"Largest mix effect: {top_mix.dimension} ({top_mix.mix_effect * 100:+.2f} pp, "
        f"{top_mix.mix_share_of_change:.0%} of the change); the rest is within-segment rate change.")
    for dim in DIMENSIONS:
        seg = shift_share_sql(con, dim, base, comp).head(top_n)
        parts = [f"{r.segment} {r.total_effect * 100:+.2f} pp "
                 f"(share {r.w0:.0%}->{r.w1:.0%}, CVR {r.r0:.1%}->{r.r1:.1%})"
                 for r in seg.itertuples()]
        lines.append(f"- {dim}: " + "; ".join(parts))
    return "\n".join(lines)


if __name__ == "__main__":  # quick smoke run
    import sys
    con = connect(sys.argv[1])
    print(con.execute("SELECT * FROM kpi_month").df().round(4).to_string())
    print(summary_sql(con, "Nov", "Dec").round(4).to_string())
    print(driver_summary(con, "Nov", "Dec"))
