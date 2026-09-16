#!/usr/bin/env python3
"""Import existing evaluation reports into the configured MLflow backend."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def flatten(prefix: str, value, output: dict[str, float]) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            flatten(f"{prefix}.{key}" if prefix else key, nested, output)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        output[prefix.replace("@", "at_")] = float(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", default="data/reports")
    parser.add_argument("--tracking-uri", default="http://mlflow:5000")
    parser.add_argument("--experiment", default="personalize-ai")
    args = parser.parse_args()
    import mlflow

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    for path in sorted(Path(args.reports).glob("*.json")):
        report = json.loads(path.read_text())
        metrics: dict[str, float] = {}
        flatten("", report, metrics)
        with mlflow.start_run(run_name=path.stem):
            mlflow.log_param("report", path.name)
            mlflow.log_metrics(dict(list(metrics.items())[:1000]))
            mlflow.log_artifact(str(path), artifact_path="reports")
        print(f"logged {path}")


if __name__ == "__main__":
    main()
