from __future__ import annotations

import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from . import __version__
from .cache import Cache, MemoryTTLCache, RedisCache
from .features import FeatureStore, RedisFeatureStore
from .monitoring import CACHE, LATENCY, RECOMMENDATION_COUNT, REQUESTS
from .monitoring import CHALLENGER_LOADED
from .ranker import BPRChallenger
from .recommender import HybridRecommender


def create_app(
    feature_store: FeatureStore | None = None,
    cache: Cache | None = None,
    artifact_dir: Path | None = None,
) -> FastAPI:
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    feature_store = feature_store or RedisFeatureStore(redis_url)
    cache = cache or RedisCache(redis_url)
    model_path = Path(os.getenv("MODEL_PATH", "artifacts/bpr_ranker.pt"))
    challenger = None
    if model_path.exists():
        try:
            challenger = BPRChallenger(model_path)
            CHALLENGER_LOADED.labels("success").inc()
        except Exception:
            CHALLENGER_LOADED.labels("failure").inc()
            if os.getenv("REQUIRE_CHALLENGER", "false").lower() == "true":
                raise
    recommender = HybridRecommender(
        feature_store,
        artifact_dir or Path(os.getenv("ARTIFACT_DIR", "artifacts")),
        challenger=challenger,
        ranking_mode=os.getenv("RANKING_MODE", "champion"),
    )
    ttl = int(os.getenv("RECOMMENDATION_CACHE_TTL_SECONDS", "300"))
    app = FastAPI(title="PersonalizeAI", version=__version__)

    @app.middleware("http")
    async def request_metrics(request: Request, call_next):
        started = time.perf_counter()
        status = "500"
        try:
            response = await call_next(request)
            status = str(response.status_code)
            return response
        finally:
            REQUESTS.labels(request.url.path, status).inc()
            if request.url.path != "/metrics":
                LATENCY.labels(request.url.path, "unknown").observe(time.perf_counter() - started)

    @app.get("/health")
    def health() -> dict:
        try:
            version = feature_store.active_version()
            return {"status": "ok", "feature_version": version, "service_version": __version__}
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/v1/recommendations/{user_id}")
    def recommendations(user_id: int, count: int = Query(10, ge=1, le=100)) -> dict:
        version = feature_store.active_version()
        key = f"recommendations:{version}:{recommender.cache_namespace}:{user_id}:{count}"
        started = time.perf_counter()
        cached = cache.get(key)
        if cached is not None:
            CACHE.labels("hit").inc()
            LATENCY.labels("recommend", "hit").observe(time.perf_counter() - started)
            return {**cached, "cache": "hit"}
        CACHE.labels("miss").inc()
        values, actual_version, ranking = recommender.recommend(user_id, count)
        payload = {
            "user_id": user_id,
            "feature_version": actual_version,
            "recommendations": values,
            **ranking,
        }
        cache.set(key, payload, ttl)
        RECOMMENDATION_COUNT.observe(len(values))
        LATENCY.labels("recommend", "miss").observe(time.perf_counter() - started)
        return {**payload, "cache": "miss"}

    @app.get("/v1/model")
    def model() -> dict:
        return {
            "champion": "hybrid_reciprocal_rank_fusion",
            "ranking_mode": recommender.ranking_mode,
            "challenger_loaded": recommender.challenger is not None,
            "challenger": (
                recommender.challenger.model_version if recommender.challenger else None
            ),
            "promotion_policy": "offline validation gate before challenger activation",
        }

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
