-- 02_metric_tree.sql
-- Monthly KPI table following the metric tree
--
--   Purchases = Sessions x CVR
--   CVR       = s_hi x CVR_hi + (1 - s_hi) x CVR_lo
--     s_hi    = share of sessions that reached a page with PageValues > 0
--               (Google Analytics page value: pages that preceded a transaction)
--     CVR_hi  = conversion among those sessions, CVR_lo = among the rest
--
-- Guardrail / health metrics: bounce-heavy share, new-visitor share.

CREATE OR REPLACE TABLE kpi_month AS
SELECT
    month_raw,
    month_num,
    count(*)                                                    AS sessions,
    sum(purchase)::INTEGER                                      AS purchases,
    avg(purchase)                                               AS cvr,
    avg((page_value > 0)::INTEGER)                              AS s_hi,
    avg(purchase) FILTER (WHERE page_value > 0)                 AS cvr_hi,
    avg(purchase) FILTER (WHERE page_value = 0)                 AS cvr_lo,
    (sum(purchase) FILTER (WHERE page_value = 0))::INTEGER      AS purchases_without_value_page,
    avg((bounce_rate >= 0.2)::INTEGER)                          AS bounce_heavy_share,
    avg((visitor_type = 'New_Visitor')::INTEGER)                AS new_visitor_share,
    avg(product_pages)                                          AS avg_product_pages
FROM sessions
GROUP BY ALL
ORDER BY month_num;

-- Identity check (should be ~0 for every month).
CREATE OR REPLACE VIEW kpi_identity_check AS
SELECT month_raw,
       cvr - (s_hi * cvr_hi + (1 - s_hi) * coalesce(cvr_lo, 0)) AS identity_gap
FROM kpi_month;
