SELECT
    CAST(item_id AS BIGINT) AS item_id,
    COUNT(*) AS event_count,
    COUNT(DISTINCT user_id) AS unique_user_count,
    SUM(CASE WHEN event_type = 'view' THEN 1 ELSE 0 END) AS view_count,
    SUM(CASE WHEN event_type = 'addtocart' THEN 1 ELSE 0 END) AS cart_count,
    SUM(CASE WHEN event_type = 'transaction' THEN 1 ELSE 0 END) AS transaction_count,
    SUM(event_weight) AS weighted_popularity,
    MAX(timestamp) AS last_event_timestamp,
    DATEDIFF(
        FROM_UNIXTIME(CAST({{as_of_ms}} / 1000 AS BIGINT)),
        FROM_UNIXTIME(CAST(MAX(timestamp) / 1000 AS BIGINT))
    ) AS recency_days
FROM {{source_view}}
GROUP BY item_id
