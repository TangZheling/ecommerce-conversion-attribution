# Conversion Metric Tree & Change Attribution

Personal extension of my DSA5102 course project
[*Understanding Online Shopping Behavior*](https://github.com/TangZheling/DSA5102-Online-Shoppers-Project)
(purchase prediction, session segmentation, latent patterns).
The course project modelled **which sessions buy**. This extension asks the questions an e-commerce
analytics team asks each week: what is the metric tree, which months are actually abnormal,
**why** conversion changed, and whether the metric definitions can be trusted.

**Stack:** DuckDB SQL (all analysis logic) · Python (NumPy cross-check, bootstrap, plots) · pytest

## Key findings (UCI Online Shoppers, 12,330 sessions, 10 months)

| Question | Answer |
|---|---|
| Which months are abnormal? | A Laney p′ control chart (σ_z = 4.2, so monthly variation is ~4× binomial noise) flags **only November** (CVR 25.4% vs 15.5% annual). December's 12.5% is within limits: the Nov→Dec "drop" is mainly a return to normal. |
| What made November high? (+6.9 pp vs Jul–Oct) | **Traffic mix: +2.7 pp (~38%)**, 95% bootstrap CI [+1.3, +4.0]. About half comes from two sources (T08, T10) active only in Mar/Nov/Dec, likely campaign channels, which made up 12.5% of November sessions at 30–37% CVR. The remaining ~62% is a site-wide lift in conversion across nearly every segment. |
| What changed Nov → Dec? (−12.8 pp) | Not traffic mix (slightly positive). **Funnel intent mix: −3.3 pp (~26%)**, CI [−4.6, −2.1]: the share of sessions reaching a value page fell from 26% to 20%. Both funnel paths also converted less (66%→52%, 11%→3%). |
| Can the funnel metric be trusted? | **Data-consistency break:** purchases without a value page are exactly 0% in Feb–May and 13–33% from June onward, which suggests a tracking/attribution change. Funnel components are comparable within Jun–Dec only. |

Figures and full discussion: [`notebooks/metric_attribution.ipynb`](notebooks/metric_attribution.ipynb).

## Method

**Metric tree:** `Purchases = Sessions × CVR`, `CVR = s_hi·CVR_hi + (1−s_hi)·CVR_lo`, where `s_hi` is the
share of sessions that reached a page with PageValues > 0. Identity verified every month.

**Shift-share attribution:** for each dimension (visitor type, traffic source, funnel intent,
weekend, region),

```
ΔCVR = Σ Δwᵢ·(r̄ᵢ − CVR̄)   [mix]   +   Σ Δrᵢ·w̄ᵢ   [rate]
```

This is exact, with no residual. Centring the mix term makes per-segment contributions interpretable.
Periods can be single months or pooled baselines. Uncertainty comes from 2,000 session bootstrap
resamples stratified by month.

**Monitoring:** Laney p′ chart, i.e. binomial limits widened by the moving-range spread of monthly z-scores.

## Repository layout

```
sql/01_base.sql          typed session table, dimension long table, segment×month cells
sql/02_metric_tree.sql   monthly KPI table + identity check
sql/03_shift_share.sql   shift_share(dim, base_months, comp_months) and summary table macros
sql/04_monitoring.sql    Laney p′ control chart
src/attribution.py       runner, NumPy reference implementation, bootstrap, driver_summary()
tests/                   27 checks: row counts, exact identities, SQL == NumPy for 5 dims × 4 comparisons
notebooks/               executed analysis notebook
build_notebook.py        regenerates the notebook
scripts/download_data.py fetches and verifies the dataset
```

## Run

```bash
pip install -r requirements.txt
python scripts/download_data.py          # UCI CSV -> data/, SHA-256 verified
# optional: point to an existing copy instead
# export ONLINE_SHOPPERS_CSV=/path/to/online_shoppers_intention.csv
pytest -q tests
python build_notebook.py && jupyter nbconvert --to notebook --execute --inplace notebooks/metric_attribution.ipynb
```

SQL example:

```sql
SELECT * FROM shift_share('traffic_source', ['Jul','Aug','Sep','Oct'], ['Nov']);
SELECT * FROM shift_share_summary(['Nov'], ['Dec']);
```

## Limitations

These are accounting decompositions of observational data, not causal effects. The data has no user
IDs, revenue amounts or within-month dates, and traffic and region codes are anonymised. Control
limits rest on 10 monthly points. Dimensions are decomposed one at a time and are not additive
across dimensions.

**Planned next:** (B) experiment-analysis module on a randomised e-commerce dataset (SRM, CUPED,
delta method, heterogeneous effects); (C) an LLM layer that turns `driver_summary()` into a written
explanation with automatic number verification.

## Data

Sakar, C. & Kastro, Y. (2018). *Online Shoppers Purchasing Intention Dataset*. UCI Machine Learning
Repository, [doi:10.24432/C5F88Q](https://doi.org/10.24432/C5F88Q). Licensed CC BY 4.0. The CSV is
not committed; `scripts/download_data.py` fetches it.
