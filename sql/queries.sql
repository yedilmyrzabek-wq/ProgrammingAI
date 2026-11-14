-- name: readings_per_well
SELECT w.well_id,
       w.well_type,
       COUNT(*)           AS readings,
       MIN(r.timestamp)   AS first_ts,
       MAX(r.timestamp)   AS last_ts
FROM readings r
JOIN wells w ON w.well_id = r.well_id
GROUP BY w.well_id, w.well_type
ORDER BY readings DESC;

-- name: parameter_stats_by_type
SELECT w.well_type,
       r.parameter,
       p.unit,
       COUNT(r.value)        AS n,
       ROUND(AVG(r.value), 2) AS mean_value,
       ROUND(MIN(r.value), 2) AS min_value,
       ROUND(MAX(r.value), 2) AS max_value
FROM readings r
JOIN wells w      ON w.well_id = r.well_id
JOIN parameters p ON p.parameter = r.parameter
GROUP BY w.well_type, r.parameter, p.unit
ORDER BY w.well_type, r.parameter;

-- name: time_gaps
WITH well_hours AS (
    SELECT DISTINCT well_id, timestamp FROM readings
),
lagged AS (
    SELECT well_id,
           LAG(timestamp) OVER (PARTITION BY well_id ORDER BY timestamp) AS gap_start,
           timestamp AS gap_end
    FROM well_hours
)
SELECT well_id,
       gap_start,
       gap_end,
       ROUND((julianday(gap_end) - julianday(gap_start)) * 24) - 1 AS missing_hours
FROM lagged
WHERE gap_start IS NOT NULL
  AND (julianday(gap_end) - julianday(gap_start)) * 24 > 1.5
ORDER BY missing_hours DESC;

-- name: missing_hours_per_well
WITH bounds AS (
    SELECT well_id,
           COUNT(DISTINCT timestamp) AS observed_hours,
           CAST(ROUND((julianday(MAX(timestamp)) - julianday(MIN(timestamp))) * 24) + 1 AS INTEGER) AS expected_hours
    FROM readings
    GROUP BY well_id
)
SELECT well_id,
       expected_hours,
       observed_hours,
       expected_hours - observed_hours AS missing_hours,
       ROUND(100.0 * (expected_hours - observed_hours) / expected_hours, 2) AS missing_pct
FROM bounds
ORDER BY missing_pct DESC;

-- name: flags_by_well
SELECT well_id,
       flag,
       COUNT(*) AS flagged_rows
FROM readings
WHERE flag <> ''
GROUP BY well_id, flag
HAVING COUNT(*) >= 10
ORDER BY flagged_rows DESC;

-- name: flag_rate_ranking
WITH rates AS (
    SELECT well_id,
           COUNT(*) AS total_rows,
           SUM(CASE WHEN flag <> '' THEN 1 ELSE 0 END) AS flagged_rows
    FROM readings
    GROUP BY well_id
)
SELECT well_id,
       total_rows,
       flagged_rows,
       ROUND(100.0 * flagged_rows / total_rows, 2) AS flag_pct,
       RANK() OVER (ORDER BY 1.0 * flagged_rows / total_rows DESC) AS flag_rank
FROM rates
ORDER BY flag_rank;

-- name: monthly_dynamometer_area
SELECT well_id,
       strftime('%Y-%m', timestamp) AS month,
       ROUND(AVG(value), 1)         AS mean_area,
       COUNT(*)                     AS n
FROM readings
WHERE parameter = 'dynamometer_area'
  AND flag = ''
GROUP BY well_id, month
ORDER BY well_id, month;

-- name: nk7_regime_change
SELECT parameter,
       CASE WHEN timestamp < '2026-06-01' THEN 'before_june' ELSE 'from_june' END AS period,
       ROUND(AVG(value), 2) AS mean_value,
       SUM(CASE WHEN value = 0 THEN 1 ELSE 0 END) AS zero_values,
       COUNT(*) AS n
FROM readings
WHERE well_id = 'NK-7'
GROUP BY parameter, period
ORDER BY parameter, period;

-- name: rolling_pump_fillage
SELECT well_id,
       timestamp,
       ROUND(value, 2) AS pump_fillage,
       ROUND(AVG(value) OVER (
           PARTITION BY well_id ORDER BY timestamp
           ROWS BETWEEN 23 PRECEDING AND CURRENT ROW
       ), 2) AS rolling_24h
FROM readings
WHERE parameter = 'pump_fillage'
ORDER BY well_id, timestamp;

-- name: extreme_vs_well_mean
SELECT r.well_id,
       r.parameter,
       r.timestamp,
       ROUND(r.value, 2) AS value,
       ROUND(s.mean_value, 2) AS well_mean
FROM readings r
JOIN (
    SELECT well_id, parameter, AVG(value) AS mean_value
    FROM readings
    WHERE flag = ''
    GROUP BY well_id, parameter
) s ON s.well_id = r.well_id AND s.parameter = r.parameter
WHERE r.value > 3 * s.mean_value
  AND s.mean_value > 0
ORDER BY r.value / s.mean_value DESC
LIMIT 200;

-- name: split_summary
SELECT split,
       COUNT(DISTINCT well_id) AS wells,
       COUNT(*)                AS readings,
       MIN(timestamp)          AS first_ts,
       MAX(timestamp)          AS last_ts
FROM readings
GROUP BY split
ORDER BY first_ts;
