-- 01_base.sql
-- Typed session table + analysis dimensions.
-- Parameter: the CSV path is substituted for {csv_path} by the Python runner.

CREATE OR REPLACE TABLE sessions AS
SELECT
    row_number() OVER () - 1                         AS session_id,   -- 0-based original row position
    Month                                            AS month_raw,
    CASE Month WHEN 'Feb' THEN 2  WHEN 'Mar' THEN 3  WHEN 'May' THEN 5
               WHEN 'June' THEN 6 WHEN 'Jul' THEN 7  WHEN 'Aug' THEN 8
               WHEN 'Sep' THEN 9  WHEN 'Oct' THEN 10 WHEN 'Nov' THEN 11
               WHEN 'Dec' THEN 12 END                AS month_num,
    VisitorType                                      AS visitor_type,
    TrafficType                                      AS traffic_type,
    Region                                           AS region,
    Weekend                                          AS weekend,
    ProductRelated                                   AS product_pages,
    BounceRates                                      AS bounce_rate,
    ExitRates                                        AS exit_rate,
    PageValues                                       AS page_value,
    SpecialDay                                       AS special_day,
    Revenue::INTEGER                                 AS purchase
FROM read_csv('{csv_path}', header = true,
     columns = {
        'Administrative': 'BIGINT', 'Administrative_Duration': 'DOUBLE',
        'Informational': 'BIGINT', 'Informational_Duration': 'DOUBLE',
        'ProductRelated': 'BIGINT', 'ProductRelated_Duration': 'DOUBLE',
        'BounceRates': 'DOUBLE', 'ExitRates': 'DOUBLE', 'PageValues': 'DOUBLE',
        'SpecialDay': 'DOUBLE', 'Month': 'VARCHAR', 'OperatingSystems': 'BIGINT',
        'Browser': 'BIGINT', 'Region': 'BIGINT', 'TrafficType': 'BIGINT',
        'VisitorType': 'VARCHAR', 'Weekend': 'BOOLEAN', 'Revenue': 'BOOLEAN'});

-- Traffic sources with < 2% of all sessions are pooled, so every segment has
-- enough sessions per month to give a readable rate. The list is fixed on the
-- full year (a reporting convention, not a model fit).
CREATE OR REPLACE TABLE traffic_keep AS
SELECT traffic_type
FROM sessions
GROUP BY traffic_type
HAVING count(*) >= 0.02 * (SELECT count(*) FROM sessions);

-- Long format: one row per (session, dimension). Every dimension partitions
-- all sessions exactly once, which the shift-share identity requires.
CREATE OR REPLACE TABLE session_dims AS
SELECT session_id, month_raw, month_num, purchase, 'visitor_type' AS dimension,
       visitor_type AS segment
FROM sessions
UNION ALL
SELECT session_id, month_raw, month_num, purchase, 'traffic_source',
       CASE WHEN traffic_type IN (SELECT traffic_type FROM traffic_keep)
            THEN 'T' || lpad(traffic_type::VARCHAR, 2, '0') ELSE 'T_other' END
FROM sessions
UNION ALL
SELECT session_id, month_raw, month_num, purchase, 'intent',
       CASE WHEN page_value > 0 THEN 'reached_value_page' ELSE 'no_value_page' END
FROM sessions
UNION ALL
SELECT session_id, month_raw, month_num, purchase, 'weekend',
       CASE WHEN weekend THEN 'weekend' ELSE 'weekday' END
FROM sessions
UNION ALL
SELECT session_id, month_raw, month_num, purchase, 'region',
       'R' || region::VARCHAR
FROM sessions;

-- Cell counts: the sufficient statistics for every rate below and for the
-- multinomial bootstrap.
CREATE OR REPLACE TABLE segment_month AS
SELECT dimension, segment, month_raw, month_num,
       count(*)      AS sessions,
       sum(purchase) AS purchases
FROM session_dims
GROUP BY ALL;
