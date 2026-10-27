"""Checks that the SQL logic is exact and agrees with an independent implementation.

Run:  ONLINE_SHOPPERS_CSV=path/to/online_shoppers_intention.csv pytest -q tests
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import attribution as A  # noqa: E402

CSV = os.environ.get(
    "ONLINE_SHOPPERS_CSV",
    str(Path(__file__).resolve().parents[1] / "data" / "online_shoppers_intention.csv"))
PAIRS = [("Nov", "Dec"), ("Oct", "Nov"), ("Mar", "May"),
         (["Jul", "Aug", "Sep", "Oct"], "Nov")]


@pytest.fixture(scope="module")
def con():
    return A.connect(CSV)


def test_row_counts(con):
    assert con.execute("SELECT count(*), sum(purchase) FROM sessions").fetchone() == (12330, 1908)
    # every dimension partitions every session exactly once
    per_dim = con.execute(
        "SELECT dimension, count(*), count(DISTINCT session_id) FROM session_dims GROUP BY 1").fetchall()
    assert all(n == 12330 and d == 12330 for _, n, d in per_dim)
    assert len(per_dim) == len(A.DIMENSIONS)


def test_metric_tree_identity(con):
    gaps = con.execute("SELECT max(abs(identity_gap)) FROM kpi_identity_check").fetchone()[0]
    assert gaps < 1e-12
    total = con.execute("SELECT sum(purchases), sum(sessions) FROM kpi_month").fetchone()
    assert total == (1908, 12330)


@pytest.mark.parametrize("base,comp", PAIRS)
def test_decomposition_is_exact(con, base, comp):
    summ = A.summary_sql(con, base, comp)
    delta = A.period_cvr(con, comp) - A.period_cvr(con, base)
    assert np.allclose(summ.mix_effect + summ.rate_effect, delta, atol=1e-12)
    assert np.allclose(summ.delta_cvr, delta, atol=1e-12)


@pytest.mark.parametrize("base,comp", PAIRS)
@pytest.mark.parametrize("dim", A.DIMENSIONS)
def test_sql_matches_numpy(con, dim, base, comp):
    sql = A.shift_share_sql(con, dim, base, comp).set_index("segment").sort_index()
    c = A.cells(con, dim, base, comp)
    mix, rate = A.shift_share_np(c.n0, c.k0, c.n1, c.k1)
    assert list(sql.index) == list(c.index)
    assert np.allclose(sql.mix_effect, mix, atol=1e-12)
    assert np.allclose(sql.rate_effect, rate, atol=1e-12)


def test_bootstrap_brackets_point_estimate(con):
    boot = A.bootstrap_summary(con, "Nov", "Dec", reps=300).set_index("dimension")
    point = A.summary_sql(con, "Nov", "Dec").set_index("dimension")
    for dim in A.DIMENSIONS:
        assert boot.loc[dim, "delta_lo"] <= point.loc[dim, "total_effect"] <= boot.loc[dim, "delta_hi"]
