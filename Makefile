.PHONY: profile split baseline cooccurrence categories hybrid rolling-examples temporal-folds ranker listwise-ranker diagnose-ranker ingest features quality publish-features serve benchmark stack restart-platform test

profile:
	python3 scripts/profile_data.py --data-dir data/raw --output data/reports/data_profile.json

split:
	python3 scripts/split_events.py

baseline:
	python3 scripts/evaluate_popularity.py

cooccurrence:
	python3 scripts/evaluate_item_cooccurrence.py

categories:
	python3 scripts/build_item_categories.py

hybrid: categories
	python3 scripts/evaluate_hybrid_candidates.py

rolling-examples: categories
	python3 scripts/generate_rolling_examples.py

temporal-folds: categories
	python3 scripts/generate_temporal_folds.py

ranker:
	.venv/bin/python scripts/train_bpr_ranker.py

diagnose-ranker:
	.venv/bin/python scripts/diagnose_ranker.py

listwise-ranker:
	.venv/bin/python scripts/train_listwise_ranker.py

ingest:
	python3 spark_jobs/ingest_events.py

features:
	python3 spark_jobs/build_offline_features.py

quality:
	python3 spark_jobs/validate_features.py $(FEATURE_VERSION_PATH)

publish-features:
	python3 spark_jobs/publish_online_features.py $(FEATURE_VERSION_PATH) --redis-url $${REDIS_URL:-redis://localhost:6379/0}

serve:
	PYTHONPATH=src uvicorn personalize_ai.main:app --reload

benchmark:
	python3 scripts/benchmark_api.py --repetitions 5 --api-key $${API_KEY:-codespace-demo-key}

stack:
	docker compose up --build

restart-platform:
	./scripts/restart_platform.sh

test:
	PYTHONPATH=src:. .venv/bin/python -m unittest discover -s tests -v
