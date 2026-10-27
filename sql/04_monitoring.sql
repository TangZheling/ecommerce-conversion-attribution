-- 04_monitoring.sql
-- Laney p'-chart for monthly conversion rate.
--
-- A classic p-chart uses sigma_i = sqrt(pbar (1 - pbar) / n_i). With
-- thousands of sessions per month, real month-to-month variation (seasonality,
-- campaigns) is much larger than binomial noise, so almost every month would
-- be flagged. Laney's p' chart rescales sigma_i by sigma_z, the spread of the
-- monthly z-scores (estimated from the average moving range / 1.128), so
-- that flags mean "unusual even given normal month-to-month variation".

CREATE OR REPLACE TABLE cvr_control AS
WITH base AS (
    SELECT month_raw, month_num, sessions AS n, purchases AS k, cvr,
           (SELECT sum(purchases) / sum(sessions) FROM kpi_month) AS pbar
    FROM kpi_month
),
z AS (
    SELECT *, sqrt(pbar * (1 - pbar) / n) AS sigma_binom,
           (cvr - pbar) / sqrt(pbar * (1 - pbar) / n) AS z
    FROM base
),
mr AS (
    SELECT *, abs(z - lag(z) OVER (ORDER BY month_num)) AS moving_range
    FROM z
),
sz AS (
    SELECT *, (SELECT avg(moving_range) FROM mr) / 1.128 AS sigma_z FROM mr
)
SELECT month_raw, month_num, n, k, cvr, pbar, z, sigma_z,
       pbar - 3 * sigma_binom * sigma_z AS lcl,
       pbar + 3 * sigma_binom * sigma_z AS ucl,
       CASE WHEN cvr > pbar + 3 * sigma_binom * sigma_z THEN 'above'
            WHEN cvr < pbar - 3 * sigma_binom * sigma_z THEN 'below'
            ELSE 'in_control' END AS flag
FROM sz
ORDER BY month_num;
