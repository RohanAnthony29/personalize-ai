.PHONY: profile split baseline cooccurrence categories hybrid rolling-examples temporal-folds ranker listwise-ranker diagnose-ranker test

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

test:
	.venv/bin/python -m unittest discover -s tests -v
