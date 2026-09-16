SELECT
    CAST(user_id AS BIGINT) AS user_id,
    CAST(item_id AS BIGINT) AS item_id,
    COUNT(*) AS interaction_count,
    SUM(event_weight) AS affinity_score,
    MAX(event_weight) AS strongest_event_weight,
    MAX(timestamp) AS last_interaction_timestamp,
    DATEDIFF(
        FROM_UNIXTIME(CAST({{as_of_ms}} / 1000 AS BIGINT)),
        FROM_UNIXTIME(CAST(MAX(timestamp) / 1000 AS BIGINT))
    ) AS recency_days
FROM {{source_view}}
GROUP BY user_id, item_id
