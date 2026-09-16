SELECT
    CAST(user_id AS BIGINT) AS user_id,
    COUNT(*) AS event_count,
    COUNT(DISTINCT item_id) AS unique_item_count,
    SUM(CASE WHEN event_type = 'view' THEN 1 ELSE 0 END) AS view_count,
    SUM(CASE WHEN event_type = 'addtocart' THEN 1 ELSE 0 END) AS cart_count,
    SUM(CASE WHEN event_type = 'transaction' THEN 1 ELSE 0 END) AS transaction_count,
    SUM(event_weight) AS weighted_event_sum,
    MAX(timestamp) AS last_event_timestamp,
    DATEDIFF(
        FROM_UNIXTIME(CAST({{as_of_ms}} / 1000 AS BIGINT)),
        FROM_UNIXTIME(CAST(MAX(timestamp) / 1000 AS BIGINT))
    ) AS recency_days
FROM {{source_view}}
GROUP BY user_id
