#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

export API_KEY="${API_KEY:-codespace-demo-key}"
export RATE_LIMIT_REQUESTS_PER_MINUTE="${RATE_LIMIT_REQUESTS_PER_MINUTE:-2000}"
export JAVA_HOME="${JAVA_HOME:-/usr/lib/jvm/java-17-openjdk-amd64}"
export PATH="$JAVA_HOME/bin:$PATH"

python_bin="${PYTHON_BIN:-.venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  python_bin="python3"
fi

feature_version_path="${FEATURE_VERSION_PATH:-}"
if [[ -z "$feature_version_path" ]]; then
  feature_version_path="$($python_bin -c 'import json; from pathlib import Path; candidates=[]
for manifest_path in Path("data/features").glob("*/_manifest.json"):
    root=manifest_path.parent
    try:
        manifest=json.loads(manifest_path.read_text()); quality=json.loads((root/"_quality.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        continue
    if manifest.get("snapshot_type")=="full" and quality.get("passed"):
        candidates.append(root)
if not candidates:
    raise SystemExit("No validated full feature snapshot found under data/features")
print(max(candidates, key=lambda path: path.name))')"
fi

echo "Starting Redis and MLflow"
docker compose up -d redis mlflow
until docker compose exec -T redis redis-cli ping 2>/dev/null | grep -q PONG; do
  sleep 1
done

echo "Publishing $feature_version_path to Redis"
"$python_bin" spark_jobs/publish_online_features.py \
  "$feature_version_path" \
  --redis-url redis://localhost:6379/0 \
  --batch-size 5000

echo "Starting FastAPI and Prometheus"
docker compose up -d api prometheus
until curl --fail --silent http://localhost:8000/health >/dev/null; do
  sleep 1
done

curl --fail --silent \
  -H "X-API-Key: $API_KEY" \
  http://localhost:8000/v1/model
echo
echo "PersonalizeAI is ready at http://localhost:8000/docs"
