#!/usr/bin/env bash
# Tranche-3 Block D1 "native research track" AWS Batch worker body.
#
# Runs AI Scientist v2 in its NATIVE mode: research idea in -> experiments +
# paper writeup + LLM review out. No MLE-bench task, no Kaggle data, no
# mlebench grading. Shipped inside the D1 worker patch addendum tarball and
# invoked by the short containerOverrides bootstrap (which only exports
# parameters and downloads/extracts the patch), mirroring worker_run.sh.
#
# IMPORTANT: ABLATION_TASK_ID must NOT be set. That env var activates the
# MLE-bench contract audit inside the fork (_ablation_contract_enabled in
# ai_scientist/treesearch/parallel_agent.py); native runs must not have it.
#
# Required env: RUN_ID OUTPUT_DIR T3D1_IDEA_PATH T3D1_IDEA_IDX T3D1_ATTEMPT_ID
# Optional env: PHASE (default tranche3-d1) ABLATION_ARTIFACT_ROOT
#               AIS_DISABLED_STAGES (simplified arm: "3,4")
#               T3_AIS_BUDGET_PATCH T3_AIS_EXEC_TIMEOUT T3_AIS_STAGE1..4
#               T3_AIS_STEPS T3_AIS_NUM_DRAFTS T3_AIS_DEBUG_DEPTH
#               T3D1_WALL_SECONDS (default 9000)
#               T3D1_MODEL_WRITEUP T3D1_MODEL_WRITEUP_SMALL T3D1_MODEL_CITATION
#               T3D1_MODEL_REVIEW T3D1_MODEL_AGG_PLOTS T3D1_CITE_ROUNDS
set -euo pipefail

: "${RUN_ID:?RUN_ID is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"
: "${T3D1_IDEA_PATH:?T3D1_IDEA_PATH is required (idea JSON, relative to the fork dir)}"
: "${T3D1_IDEA_IDX:?T3D1_IDEA_IDX is required}"
: "${T3D1_ATTEMPT_ID:?T3D1_ATTEMPT_ID is required}"
PHASE="${PHASE:-tranche3-d1}"

if [ -n "${ABLATION_TASK_ID:-}" ]; then
  echo "ABLATION_TASK_ID must not be set for native D1 runs (it enables the MLE-bench contract audit)" >&2
  exit 2
fi

AIS_ROOT=".context/external/AI-Scientist-v2-ablation"
RUN_START_TS=$(date +%s)

# sync_artifacts SRC DEST MODE
# MODE=checkpoint excludes *.verbose.log (can grow to 100MB+); MODE=final
# includes everything. Unlike the MLE-bench worker there is no
# workspace/input/** exclusion: native runs have no copied competition data.
sync_artifacts() {
  python3 - "$1" "$2" "$3" <<'PYEOF'
import sys

import boto3
from pathlib import Path

src = Path(sys.argv[1])
dest = sys.argv[2].rstrip('/') + '/'
mode = sys.argv[3]
bucket, key = dest.removeprefix('s3://').split('/', 1)
s3 = boto3.client('s3')
for path in sorted(src.rglob('*')):
    if not path.is_file():
        continue
    rel = path.relative_to(src).as_posix()
    if mode == 'checkpoint' and rel.endswith('.verbose.log'):
        continue
    s3.upload_file(str(path), bucket, key + rel)
PYEOF
}

# copy_recent_dirs SRC DEST CUTOFF_EPOCH
# Copies subdirectories of SRC modified at or after CUTOFF_EPOCH-5s into DEST.
copy_recent_dirs() {
  python3 - "$1" "$2" "$3" <<'PYEOF'
import shutil
import sys
from pathlib import Path

src = Path(sys.argv[1])
dest = Path(sys.argv[2])
cutoff = float(sys.argv[3]) - 5
matched = []
if src.is_dir():
    for p in src.iterdir():
        if p.is_dir() and p.stat().st_mtime >= cutoff:
            target = dest / p.name
            shutil.copytree(p, target, dirs_exist_ok=True)
            matched.append(str(target))
print('\n'.join(matched))
PYEOF
}

CHECKPOINT_PID=""
ARTIFACT_DEST=""
WORKER_FINAL_SYNC_DONE=0

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
exec > >(tee -a "$OUTPUT_DIR/worker_stdout.log") 2> >(tee -a "$OUTPUT_DIR/worker_stderr.log" >&2)

worker_cleanup() {
  local status=$?
  if [ -n "${CHECKPOINT_PID:-}" ]; then
    kill "$CHECKPOINT_PID" >/dev/null 2>&1 || true
  fi
  if [ "${WORKER_FINAL_SYNC_DONE:-0}" != "1" ] && [ -n "${ARTIFACT_DEST:-}" ] && [ -d "$OUTPUT_DIR" ]; then
    sync_artifacts "$OUTPUT_DIR" "$ARTIFACT_DEST" final >/dev/null 2>&1 || true
  fi
  exit "$status"
}
trap worker_cleanup EXIT

echo "worker_native_d1 start: run_id=$RUN_ID idea=$T3D1_IDEA_PATH idx=$T3D1_IDEA_IDX attempt=$T3D1_ATTEMPT_ID disabled_stages=${AIS_DISABLED_STAGES:-none}"

# Compute the artifact destination up-front so periodic checkpoints can sync
# partial artifacts even if the job is killed at the wall timeout.
if [ -n "${ABLATION_ARTIFACT_ROOT:-}" ]; then
  ARTIFACT_DEST="${ABLATION_ARTIFACT_ROOT%/}/${PHASE}/${RUN_ID}/"
fi

# The worker image ships without a LaTeX toolchain; the native writeup needs
# pdflatex + bibtex (and uses chktex for style checks). Best-effort install:
# a failure here degrades to "no paper PDF" rather than killing the run.
if ! command -v pdflatex >/dev/null 2>&1; then
  echo "pdflatex not found; installing texlive (best effort)"
  (apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      texlive-latex-base texlive-latex-recommended texlive-latex-extra \
      texlive-fonts-recommended texlive-bibtex-extra chktex) \
    || echo "WARNING: texlive install failed; writeup will not produce a PDF" >&2
fi

# Tranche-3 bfts runtime/budget clamp (same mechanism as worker_run.sh).
if [ "${T3_AIS_BUDGET_PATCH:-0}" = "1" ]; then
  AIS_PATCH_ARGS=()
  if [ -n "${T3_AIS_EXEC_TIMEOUT:-}" ]; then AIS_PATCH_ARGS+=(--exec-timeout "$T3_AIS_EXEC_TIMEOUT"); fi
  if [ -n "${T3_AIS_STAGE1:-}" ]; then AIS_PATCH_ARGS+=(--stage1 "$T3_AIS_STAGE1"); fi
  if [ -n "${T3_AIS_STAGE2:-}" ]; then AIS_PATCH_ARGS+=(--stage2 "$T3_AIS_STAGE2"); fi
  if [ -n "${T3_AIS_STAGE3:-}" ]; then AIS_PATCH_ARGS+=(--stage3 "$T3_AIS_STAGE3"); fi
  if [ -n "${T3_AIS_STAGE4:-}" ]; then AIS_PATCH_ARGS+=(--stage4 "$T3_AIS_STAGE4"); fi
  if [ -n "${T3_AIS_STEPS:-}" ]; then AIS_PATCH_ARGS+=(--steps "$T3_AIS_STEPS"); fi
  if [ -n "${T3_AIS_NUM_DRAFTS:-}" ]; then AIS_PATCH_ARGS+=(--num-drafts "$T3_AIS_NUM_DRAFTS"); fi
  if [ -n "${T3_AIS_DEBUG_DEPTH:-}" ]; then AIS_PATCH_ARGS+=(--debug-depth "$T3_AIS_DEBUG_DEPTH"); fi
  python3 scripts/worker_ais_budget_patch.py ${AIS_PATCH_ARGS[@]+"${AIS_PATCH_ARGS[@]}"}
fi

set +e

# Background checkpoint loop: periodically sync partial artifacts so a
# wall-timeout kill cannot lose everything before the final sync.
if [ -n "$ARTIFACT_DEST" ]; then
  (
    while true; do
      sleep 300
      sync_artifacts "$OUTPUT_DIR" "$ARTIFACT_DEST" checkpoint >/dev/null 2>&1 || true
    done
  ) &
  CHECKPOINT_PID=$!
fi

# Native invocation: NO --skip_writeup / --skip_review (both arms produce a
# paper + LLM review). Model flags are pinned to currently-served OpenAI
# models; the upstream defaults (o1-preview-2024-09-12 etc.) are retired.
LAUNCH_ARGS=(
  --load_ideas "$T3D1_IDEA_PATH"
  --idea_idx "$T3D1_IDEA_IDX"
  --attempt_id "$T3D1_ATTEMPT_ID"
  --model_writeup "${T3D1_MODEL_WRITEUP:-gpt-4.1}"
  --model_writeup_small "${T3D1_MODEL_WRITEUP_SMALL:-gpt-4.1-mini}"
  --model_citation "${T3D1_MODEL_CITATION:-gpt-4.1-mini}"
  --model_review "${T3D1_MODEL_REVIEW:-gpt-4.1}"
  --model_agg_plots "${T3D1_MODEL_AGG_PLOTS:-gpt-4.1}"
  --num_cite_rounds "${T3D1_CITE_ROUNDS:-20}"
)
if [ -f "$AIS_ROOT/${T3D1_IDEA_PATH%.json}.py" ]; then
  # README-canonical native invocation seeds experimentation with the
  # companion code snippet when one ships next to the idea JSON.
  LAUNCH_ARGS+=(--load_code)
fi

(
  cd "$AIS_ROOT"
  timeout --kill-after=120 "${T3D1_WALL_SECONDS:-9000}" \
    python3 launch_scientist_bfts.py "${LAUNCH_ARGS[@]}"
)
RUN_STATUS=$?
echo "launch_scientist_bfts exit status: $RUN_STATUS"

# Copy the freshly created experiments/<timestamp>_* dir(s) — including any
# generated PDF/tex and the review outputs (review_text.txt,
# review_img_cap_ref.json, written by launch_scientist_bfts.py into the idea
# dir) — into OUTPUT_DIR for artifact sync.
copy_recent_dirs "$AIS_ROOT/experiments" "$OUTPUT_DIR/ai_scientist_experiments" "$RUN_START_TS"

# Machine-readable completion marker for scripts/collect_d1_reviews.py.
python3 - "$OUTPUT_DIR/run_status.json" "$RUN_STATUS" <<'PYEOF'
import json
import os
import sys
import time

path, status = sys.argv[1], int(sys.argv[2])
payload = {
    "run_id": os.environ["RUN_ID"],
    "phase": os.environ.get("PHASE", "tranche3-d1"),
    "arm": os.environ.get("ABLATION_VARIANT_ID"),
    "idea_path": os.environ["T3D1_IDEA_PATH"],
    "idea_idx": int(os.environ["T3D1_IDEA_IDX"]),
    "idea_slug": os.environ.get("T3D1_IDEA_SLUG"),
    "attempt_id": int(os.environ["T3D1_ATTEMPT_ID"]),
    "disabled_stages": os.environ.get("AIS_DISABLED_STAGES", ""),
    "exit_status": status,
    "completed": status == 0,
    "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
with open(path, "w") as fh:
    json.dump(payload, fh, indent=2)
PYEOF

if [ -n "$CHECKPOINT_PID" ]; then kill "$CHECKPOINT_PID" >/dev/null 2>&1 || true; fi

SYNC_STATUS=0
if [ -n "$ARTIFACT_DEST" ] && [ -d "$OUTPUT_DIR" ]; then
  sync_artifacts "$OUTPUT_DIR" "$ARTIFACT_DEST" final
  SYNC_STATUS=$?
fi
WORKER_FINAL_SYNC_DONE=1

if [ "$RUN_STATUS" -ne 0 ]; then exit "$RUN_STATUS"; fi
exit "$SYNC_STATUS"
