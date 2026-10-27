-- 03_shift_share.sql
-- Symmetric shift-share (mix vs rate) decomposition of a ratio metric.
--
-- For one dimension with segments i, CVR = sum_i w_i * r_i, where w_i is the
-- segment's session share and r_i its conversion rate. Between a base period 0
-- and a comparison period 1:
--
--   dCVR = sum_i (w1_i - w0_i) * (rbar_i - CVRbar)     <- mix effect
--        + sum_i (r1_i - r0_i) * wbar_i                <- rate effect
--
-- with bars = simple averages of the two periods. Subtracting CVRbar inside
-- the mix term changes nothing in total (sum_i dw_i = 0) but makes each
-- segment's mix contribution interpretable: gaining share only helps the
-- total if the segment converts better than average. The decomposition is
-- exact (no residual term).
--
-- Segments absent in one period get w = 0 there; their rate is undefined, so
-- the observed rate of the other period is used (rate effect 0 for them).

CREATE OR REPLACE MACRO shift_share(dim, base_ms, comp_ms) AS TABLE
-- base_ms / comp_ms are lists of month labels, e.g. ['Nov'] or a pooled
-- baseline such as ['Jul', 'Aug', 'Sep', 'Oct'].
WITH cells AS (
    SELECT segment,
           coalesce(sum(sessions)  FILTER (WHERE list_contains(base_ms, month_raw)), 0) AS n0,
           coalesce(sum(purchases) FILTER (WHERE list_contains(base_ms, month_raw)), 0) AS k0,
           coalesce(sum(sessions)  FILTER (WHERE list_contains(comp_ms, month_raw)), 0) AS n1,
           coalesce(sum(purchases) FILTER (WHERE list_contains(comp_ms, month_raw)), 0) AS k1
    FROM segment_month
    WHERE dimension = dim
    GROUP BY segment
),
rates AS (
    SELECT *,
           n0 / sum(n0) OVER () AS w0,
           n1 / sum(n1) OVER () AS w1,
           coalesce(k0 / nullif(n0, 0), k1 / nullif(n1, 0), 0) AS r0,
           coalesce(k1 / nullif(n1, 0), k0 / nullif(n0, 0), 0) AS r1,
           sum(k0) OVER () / sum(n0) OVER () AS cvr0,
           sum(k1) OVER () / sum(n1) OVER () AS cvr1
    FROM cells
    WHERE n0 + n1 > 0
)
SELECT dim AS dimension, segment, n0, k0, n1, k1, w0, w1, r0, r1,
       (w1 - w0) * ((r0 + r1) / 2 - (cvr0 + cvr1) / 2) AS mix_effect,
       (r1 - r0) * (w0 + w1) / 2                       AS rate_effect,
       (w1 - w0) * ((r0 + r1) / 2 - (cvr0 + cvr1) / 2)
         + (r1 - r0) * (w0 + w1) / 2                   AS total_effect,
       cvr1 - cvr0                                     AS delta_cvr
FROM rates
ORDER BY abs(total_effect) DESC;

-- One summary row per dimension: how much of the change is mix vs rate?
CREATE OR REPLACE MACRO shift_share_summary(base_ms, comp_ms) AS TABLE
SELECT dimension,
       sum(mix_effect)                               AS mix_effect,
       sum(rate_effect)                              AS rate_effect,
       sum(total_effect)                             AS total_effect,
       any_value(delta_cvr)                          AS delta_cvr,
       sum(mix_effect) / any_value(delta_cvr)        AS mix_share_of_change
FROM (
    SELECT * FROM shift_share('visitor_type',   base_ms, comp_ms) UNION ALL
    SELECT * FROM shift_share('traffic_source', base_ms, comp_ms) UNION ALL
    SELECT * FROM shift_share('intent',         base_ms, comp_ms) UNION ALL
    SELECT * FROM shift_share('weekend',        base_ms, comp_ms) UNION ALL
    SELECT * FROM shift_share('region',         base_ms, comp_ms)
)
GROUP BY dimension
ORDER BY abs(sum(mix_effect)) DESC;
