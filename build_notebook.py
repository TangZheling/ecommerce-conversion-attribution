"""Builds notebooks/metric_attribution.ipynb (run, then execute with nbconvert)."""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = []

cells.append(md("""# Conversion Metric Tree and Change Attribution
**Personal extension of the DSA5102 Online Shoppers project · Zheling Tang**

The course project asked *which sessions buy* (prediction, segmentation, latent structure).
This extension asks the questions a business analytics team asks every week:

1. **What is the metric tree?** How does the headline conversion rate (CVR) break into components we can monitor?
2. **Which months are actually abnormal?** A control chart that accounts for normal month-to-month variation.
3. **Why did CVR change?** An exact *mix vs rate* (shift-share) decomposition across five dimensions, with bootstrap intervals.
4. **Can we trust the metric definitions?** A data-consistency check on the intent step of the funnel.

All analysis logic lives in SQL (`sql/*.sql`, DuckDB). Python runs it, cross-checks it against an
independent NumPy implementation (`tests/`), adds the bootstrap and draws the figures.

**Data:** [UCI Online Shoppers Purchasing Intention](https://doi.org/10.24432/C5F88Q), 12,330 sessions, one year,
10 months observed (no Jan / Apr). Sessions only: no user IDs, no revenue amounts, no experiment assignment."""))

cells.append(code("""import os, sys
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import display, Markdown
import attribution as A

CSV = os.environ.get('ONLINE_SHOPPERS_CSV', str(ROOT / 'data' / 'online_shoppers_intention.csv'))
con = A.connect(CSV)

BLUE, ORANGE, INK, MUTED, CRIT = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#d03b3b'
plt.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#b5b4ae', 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.grid': True, 'grid.color': '#ebeae6',
                     'grid.linewidth': .6, 'axes.axisbelow': True, 'font.size': 9.5,
                     'axes.titleweight': 'bold', 'axes.titlesize': 10.5,
                     'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb'})
pd.set_option('display.float_format', lambda v: f'{v:,.4f}')
print('sessions, purchases =', con.execute('SELECT count(*), sum(purchase) FROM sessions').fetchone())"""))

# ---------------------------------------------------------------- metric tree
cells.append(md("""## 1. Metric tree

```
Purchases = Sessions × CVR
CVR       = s_hi × CVR_hi + (1 − s_hi) × CVR_lo
            s_hi   : share of sessions that reached a page with PageValues > 0
                     (Google Analytics "page value": pages that preceded a transaction)
            CVR_hi : conversion of those sessions        CVR_lo : conversion of the rest
Guardrails: bounce-heavy share (BounceRates ≥ 0.2), new-visitor share
```

`s_hi` is a funnel step (did the session get close to checkout?), `CVR_hi` is closing efficiency.
The identity holds exactly every month (`kpi_identity_check`, tested)."""))

cells.append(code("""kpi = con.execute('SELECT * FROM kpi_month').df()
assert con.execute('SELECT max(abs(identity_gap)) FROM kpi_identity_check').fetchone()[0] < 1e-12
display(kpi[['month_raw','sessions','purchases','cvr','s_hi','cvr_hi','cvr_lo',
             'bounce_heavy_share','new_visitor_share']].rename(columns={'month_raw': 'month'}))

fig, axes = plt.subplots(1, 3, figsize=(12, 3.1), sharex=True)
for ax, col, title in zip(axes, ['s_hi', 'cvr_hi', 'cvr_lo'],
        ['s_hi: reached a value page', 'CVR_hi: conversion | value page', 'CVR_lo: conversion | no value page']):
    ax.plot(kpi.month_raw, kpi[col], color=BLUE, lw=2, marker='o', ms=5)
    ax.set_title(title); ax.yaxis.set_major_formatter(lambda v, _: f'{v:.0%}')
    ax.set_ylim(0, None); ax.tick_params(axis='x', labelrotation=45)
fig.suptitle('Metric-tree components by month (each panel its own scale)', x=.01, ha='left', fontsize=11)
fig.tight_layout(); plt.show()"""))

# ---------------------------------------------------------------- monitoring
cells.append(md("""## 2. Which months are abnormal? Laney p′-chart

A plain p-chart with thousands of sessions per month flags almost everything, because real
month-to-month variation is far larger than binomial noise. Laney's p′ chart widens the binomial
limits by σ_z, the spread of monthly z-scores (moving-range estimate), so a flag means
"unusual even given ordinary seasonal swings" (`sql/04_monitoring.sql`)."""))

cells.append(code("""ctl = con.execute('SELECT * FROM cvr_control').df()
display(ctl[['month_raw','n','cvr','z','lcl','ucl','flag']].rename(columns={'month_raw': 'month'}))
x = np.arange(len(ctl))
fig, ax = plt.subplots(figsize=(9, 3.4))
ax.fill_between(x, ctl.lcl.clip(lower=0), ctl.ucl, step='mid', color=BLUE, alpha=.10, lw=0,
                label='Laney 3σ limits')
ax.axhline(ctl.pbar.iloc[0], color=MUTED, lw=1, ls=':', label=f'Annual CVR {ctl.pbar.iloc[0]:.1%}')
ax.plot(x, ctl.cvr, color=BLUE, lw=2, marker='o', ms=6, label='Monthly CVR')
for i, r in ctl.iterrows():
    if r.flag != 'in_control':
        ax.plot(i, r.cvr, 'o', ms=11, mfc='none', mec=CRIT, mew=2)
        ax.annotate(f'{r.month_raw}: {r.cvr:.1%}  above limit', (i, r.cvr), xytext=(-150, 4),
                    textcoords='offset points', color=INK, fontsize=9)
ax.set_xticks(x, ctl.month_raw); ax.yaxis.set_major_formatter(lambda v, _: f'{v:.0%}')
ax.set(ylim=(0, .5), ylabel='Conversion rate', title='Only November breaks the control limits')
ax.legend(loc='upper left', fontsize=8, frameon=False); fig.tight_layout(); plt.show()
display(Markdown(f\"\"\"**Reading.** σ_z = {ctl.sigma_z.iloc[0]:.2f}: monthly variation is ~4× binomial noise.
November ({ctl.set_index('month_raw').loc['Nov','cvr']:.1%}) is the only month above its limit. December's
{ctl.set_index('month_raw').loc['Dec','cvr']:.1%} is inside the limits, so the Nov→Dec "drop" is mostly
a return from an unusual November, not a December failure. The useful questions are therefore
(a) *what made November high* and (b) *what changed from November to December*.\"\"\"))"""))

# ---------------------------------------------------------------- attribution
cells.append(md("""## 3. Why did CVR change? Mix vs rate attribution

For one dimension with segments *i*, CVR = Σ wᵢ rᵢ (wᵢ = session share, rᵢ = segment CVR). Between
periods 0 and 1 (`sql/03_shift_share.sql`):

$$\\Delta\\text{CVR}=\\underbrace{\\sum_i \\Delta w_i\\,(\\bar r_i-\\overline{\\text{CVR}})}_{\\text{mix: traffic moved between segments}}
+\\underbrace{\\sum_i \\Delta r_i\\,\\bar w_i}_{\\text{rate: segments converted differently}}$$

Bars are two-period averages. The split is exact (no residual). Centring the mix term at the average
CVR makes each segment's mix contribution readable: gaining share helps only if the segment converts
above average. Uncertainty: 2,000 bootstrap resamples of sessions within each month (95% percentile
intervals). Five dimensions are decomposed separately; each answers "how much would CVR have moved
from re-weighting this dimension alone?"

| Comparison | Why |
|---|---|
| Jul–Oct (pooled) → Nov | What made the flagged month high? Pooled baseline avoids Oct's small sample |
| Nov → Dec | What changed in the largest month-over-month fall? |"""))

cells.append(code("""COMPARISONS = [(['Jul', 'Aug', 'Sep', 'Oct'], 'Nov'), ('Nov', 'Dec')]
results = {}
for base, comp in COMPARISONS:
    point = A.summary_sql(con, base, comp).set_index('dimension')
    boot = A.bootstrap_summary(con, base, comp, reps=2000, seed=42).set_index('dimension')
    results[(A.label(base), comp)] = point.join(boot)

fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), sharey=True)
order = A.DIMENSIONS
for ax, ((b, c), res) in zip(axes, results.items()):
    res = res.loc[order]
    y = np.arange(len(order))
    for off, kind, color in [(-.19, 'mix', BLUE), (.19, 'rate', ORANGE)]:
        val = res[f'{kind}_effect'] * 100
        err = np.vstack([val - res[f'{kind}_lo'] * 100, res[f'{kind}_hi'] * 100 - val])
        ax.barh(y + off, val, height=.36, color=color, label=f'{kind} effect',
                edgecolor='#fcfcfb', linewidth=1)
        ax.errorbar(val, y + off, xerr=err, fmt='none', ecolor=INK, elinewidth=1, capsize=2.5)
    ax.axvline(0, color=MUTED, lw=.8)
    delta = res.delta_cvr.iloc[0] * 100
    ax.set_title(f'{b} → {c}: ΔCVR {delta:+.1f} pp'); ax.set_xlabel('Contribution to ΔCVR (pp)')
    ax.set_yticks(y, [d.replace('_', ' ') for d in order]); ax.invert_yaxis()
fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=8.5,
           loc='upper right', ncol=2)
fig.suptitle('Within every dimension, mix + rate = ΔCVR. Whiskers: 95% bootstrap intervals',
             x=.01, ha='left', fontsize=11)
fig.tight_layout(); plt.show()
for key, res in results.items():
    display(Markdown(f'**{key[0]} → {key[1]}**'))
    display(res[['mix_effect','mix_lo','mix_hi','rate_effect','rate_lo','rate_hi','mix_share_of_change']])"""))

cells.append(code("""def segment_plot(ax, base, comp, dim, title):
    seg = A.shift_share_sql(con, dim, base, comp)
    seg = seg[seg.n0 + seg.n1 >= 30].head(8)
    y = np.arange(len(seg))
    ax.barh(y - .19, seg.mix_effect * 100, height=.36, color=BLUE, label='mix', edgecolor='#fcfcfb')
    ax.barh(y + .19, seg.rate_effect * 100, height=.36, color=ORANGE, label='rate', edgecolor='#fcfcfb')
    ax.set_yticks(y, [f'{s}  ({w0:.0%}→{w1:.0%} share, {r0:.0%}→{r1:.0%} CVR)'
                      for s, w0, w1, r0, r1 in seg[['segment','w0','w1','r0','r1']].itertuples(index=False)])
    ax.invert_yaxis(); ax.axvline(0, color=MUTED, lw=.8)
    ax.set(title=title, xlabel='Contribution to ΔCVR (pp)')
    return seg

fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.4))
s1 = segment_plot(axes[0], ['Jul','Aug','Sep','Oct'], 'Nov', 'traffic_source',
                  'Jul–Oct → Nov, by traffic source: T08/T10 switch on (mix); T02 grows and converts better')
s2 = segment_plot(axes[1], 'Nov', 'Dec', 'intent',
                  'Nov → Dec, by funnel intent: fewer sessions reach value pages, both paths convert less')
axes[0].legend(frameon=False, fontsize=8, loc='lower right')
fig.tight_layout(); plt.show()"""))

cells.append(code("""for (b, c) in [(['Jul','Aug','Sep','Oct'], 'Nov'), ('Nov', 'Dec')]:
    display(Markdown('```\\n' + A.driver_summary(con, b, c) + '\\n```'))"""))

cells.append(md("""**Findings.**

* **What made November high (+6.9 pp vs Jul–Oct).** Traffic mix explains **+2.7 pp (~38%)**,
  95% CI about +1.3 to +4.0 pp. About half of that (+1.4 pp) comes from two sources, T08 and T10, that
  are **active only in Mar, Nov and Dec**. They were absent in Jul–Oct, made up 12.5% of November
  sessions and converted at 30–37%, which looks like campaign channels switching on. Source T02 also grew
  from 28% to 41% of sessions (+0.7 pp mix) and converted better (+2.0 pp rate). The remaining ~62% is a
  within-segment lift seen in nearly every visitor type, region, weekday/weekend and both funnel paths,
  which looks like a site-wide seasonal or promotional effect rather than one channel.
* **What changed Nov → Dec (−12.8 pp).** Traffic mix did *not* cause it: its contribution is slightly
  positive. The largest mix effect is in the funnel. Fewer sessions reached a value page (26% → 20%),
  worth **−3.3 pp (~26%)**, CI about −4.6 to −2.1 pp. The rest is lower conversion on both paths
  (CVR_hi 66% → 52%, CVR_lo 11% → 3%), with every major source and region falling.
* **Operational reading.** For November's spike, check the campaign calendar: what ran on T08/T10 and
  what drove the site-wide lift. December needs funnel diagnosis (product → value-page step), not a
  channel investigation.

**What this does not show.** These are accounting decompositions of observational data, not causal
effects. "Traffic mix explains 2.7 pp" means "re-weighting segments to November's shares moves CVR by
2.7 pp at the average segment rates", not "buying more T08/T10 traffic would raise CVR by that much";
marginal users from a scaled-up channel may convert differently. A segment absent in one period has
no rate there, so its whole contribution is booked as mix (convention documented in the SQL).
Decompositions also depend on the segmentation chosen, and dimensions overlap (they are not additive
across dimensions). The intent step is itself
a behavioural outcome, so its "mix" is a funnel shift, not an external traffic shift."""))

# ---------------------------------------------------------------- data quality
cells.append(md("""## 4. Can we trust the funnel definition? A consistency check

If `PageValues > 0` marks "reached a page preceding a transaction", purchases with no value page
should be rare *and stable over time*. They are not:"""))

cells.append(code("""dq = kpi[['month_raw', 'purchases', 'purchases_without_value_page']].copy()
dq['share_of_purchases'] = dq.purchases_without_value_page / dq.purchases
display(dq.rename(columns={'month_raw': 'month'}))
fig, ax = plt.subplots(figsize=(8.5, 2.9))
ax.bar(dq.month_raw, dq.share_of_purchases, color=BLUE, width=.6)
for i, r in dq.iterrows():
    ax.text(i, r.share_of_purchases + .008, f'{r.share_of_purchases:.0%}', ha='center', fontsize=8.5, color=INK)
ax.yaxis.set_major_formatter(lambda v, _: f'{v:.0%}')
ax.set(ylabel='Share of purchases', ylim=(0, .4),
       title='Purchases recorded without any value page: exactly 0% through May, 13–33% from June on')
fig.tight_layout(); plt.show()"""))

cells.append(md("""**Reading.** In Feb, Mar and May, *every* purchase has PageValues > 0. From June on, 13–33% do not.
A step change like this is more consistent with a change in tracking or attribution windows than with
user behaviour. Consequences:

* `s_hi`, `CVR_hi` and `CVR_lo` are **comparable within Jun–Dec** (both attribution comparisons above
  stay inside that regime) but should not be trended across May → June without a caveat.
* In a production setting this would be raised with the tracking owner before anyone builds a target on `s_hi`.
* It also adds to the leakage concern for PageValues in the course project's prediction task: the
  feature's meaning may not be fixed over time.

## 5. Limitations and next steps

| Limitation | Implication |
|---|---|
| No user IDs or dates within a month | No retention, cohort or weekly monitoring; months are the smallest period |
| No revenue amount | Purchases stand in for GMV; AOV cannot be decomposed |
| Anonymised traffic / region codes | Findings name codes (T02, T08) and cannot be tied to real channels |
| Observational data | Attribution ≠ causal effect; an experiment would be needed to size interventions |
| One year, 10 months | The control limits rest on 10 points and are indicative only |

**Next:** (B) an experiment-analysis module on a randomised e-commerce dataset (SRM, CUPED, delta
method, heterogeneous effects), and (C) an LLM layer that turns `driver_summary()` output into a
written explanation with automatic number checks."""))

nb.cells = cells
nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
out = Path(__file__).parent / 'notebooks' / 'metric_attribution.ipynb'
out.parent.mkdir(exist_ok=True)
nbf.write(nb, out)
print('wrote', out)
