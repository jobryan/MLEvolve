#!/usr/bin/env bash
set -euo pipefail

cd /opt/mlevolve

mkdir -p "${MLEBENCH_DATASET_DIR:-/mnt/mle-bench}"

if [[ -n "${KAGGLE_USERNAME:-}" && -n "${KAGGLE_KEY:-}" ]]; then
  mkdir -p "${HOME}/.kaggle"
  python3 - <<'PY'
import json
import os
from pathlib import Path

path = Path.home() / ".kaggle" / "kaggle.json"
path.write_text(
    json.dumps({"username": os.environ["KAGGLE_USERNAME"], "key": os.environ["KAGGLE_KEY"]}) + "\n",
    encoding="utf-8",
)
path.chmod(0o600)
PY
fi

exec "$@"
