#!/usr/bin/env bash
# AWS Batch ablation worker body.
#
# Shipped inside the worker patch tarball and invoked by the short
# containerOverrides command emitted by scripts/build_aws_ablation_jobs.py,
# which only exports parameters and downloads/extracts the patch. Keeping the
# run logic here keeps the Batch containerOverrides JSON far below the
# 8192-character limit.
#
# Required env: SYSTEM TASK_ID MANIFEST_REF PHASE RUN_ID OUTPUT_DIR
#               MLEBENCH_DATASET_DIR
# Optional env: ABLATION_ARTIFACT_ROOT
#               T3_AIS_BUDGET_PATCH T3_AIS_EXEC_TIMEOUT T3_AIS_STAGE1..4
#               T3_AIS_STEPS T3_AIS_NUM_DRAFTS T3_AIS_DEBUG_DEPTH
set -euo pipefail

: "${SYSTEM:?SYSTEM is required}"
: "${TASK_ID:?TASK_ID is required}"
: "${MANIFEST_REF:?MANIFEST_REF is required}"
: "${PHASE:?PHASE is required}"
: "${RUN_ID:?RUN_ID is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"

RUN_START_TS=$(date +%s)

# sync_artifacts SRC DEST MODE
# MODE=checkpoint excludes workspace/input/** (copied competition data; the
# tranche-3 canary uploaded ~470MB of it per 5-minute cycle) and *.verbose.log
# (can grow to 100MB+). MODE=final keeps the workspace/input exclusion but
# includes logs.
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
    if '/workspace/input/' in '/' + rel:
        continue
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

# valid_grade GRADE_REPORT_JSON -> prints 1 if report.valid_submission else 0
valid_grade() {
  python3 - "$1" <<'PYEOF'
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
ok = False
if p.is_file():
    r = json.loads(p.read_text())
    report = r.get('report') or {}
    ok = bool(report.get('valid_submission'))
print('1' if ok else '0')
PYEOF
}

CHECKPOINT_PID=""
ARTIFACT_DEST=""
WORKER_FINAL_SYNC_DONE=0

mkdir -p "$OUTPUT_DIR"
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

echo "worker_run start: run_id=$RUN_ID system=$SYSTEM task=$TASK_ID manifest=$MANIFEST_REF"

# Compute the artifact destination up-front so periodic checkpoints can sync
# partial artifacts even if the job is killed at the wall timeout.
if [ -n "${ABLATION_ARTIFACT_ROOT:-}" ]; then
  ARTIFACT_DEST="${ABLATION_ARTIFACT_ROOT%/}/${PHASE}/${RUN_ID}/"
fi

test -n "${MLEBENCH_DATASET_DIR:-}" || { echo "MLEBENCH_DATASET_DIR is required" >&2; exit 2; }
DESCRIPTION_PATH="${MLEBENCH_DATASET_DIR}/${TASK_ID}/prepared/public/description.md"
if [ ! -f "$DESCRIPTION_PATH" ]; then
  mlebench prepare -c "$TASK_ID" --data-dir "$MLEBENCH_DATASET_DIR"
fi

# Optional AI Scientist v2 runtime/budget patch (tranche-3 bfts budget clamp).
if [ "$SYSTEM" = "ai_scientist_v2" ] && [ "${T3_AIS_BUDGET_PATCH:-0}" = "1" ]; then
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

python3 scripts/run_ablation_manifest.py --manifest "$MANIFEST_REF"
RUN_STATUS=$?

if [ "$SYSTEM" = "ai_scientist_v2" ]; then
  AI_EXPERIMENTS_DIR=".context/external/AI-Scientist-v2-ablation/experiments"
  AI_EXPERIMENTS_OUT="$OUTPUT_DIR/ai_scientist_experiments"
  copy_recent_dirs "$AI_EXPERIMENTS_DIR" "$AI_EXPERIMENTS_OUT" "$RUN_START_TS"
  AI_WORKSPACES_DIR=".context/external/AI-Scientist-v2-ablation/workspaces"
  AI_WORKSPACES_OUT="$OUTPUT_DIR/ai_scientist_workspaces"
  copy_recent_dirs "$AI_WORKSPACES_DIR" "$AI_WORKSPACES_OUT" "$RUN_START_TS"
  python3 scripts/recover_ai_scientist_submission.py --output-dir "$OUTPUT_DIR" \
    --task-id "$TASK_ID" --data-dir "$MLEBENCH_DATASET_DIR" --run-id "$RUN_ID" \
    --timeout-seconds 900 --allow-fallback || true
fi

if [ -d "$OUTPUT_DIR" ]; then
  python3 scripts/grade_ablation_submissions.py --output-dir "$OUTPUT_DIR" \
    --task-id "$TASK_ID" --system "$SYSTEM" --run-id "$RUN_ID" \
    --data-dir "$MLEBENCH_DATASET_DIR" || true
fi

if [ "$SYSTEM" = "ai_scientist_v2" ] && [ -f "$OUTPUT_DIR/grader/grade_report.json" ]; then
  VALID_GRADE=$(valid_grade "$OUTPUT_DIR/grader/grade_report.json")
  if [ "$VALID_GRADE" = "1" ]; then RUN_STATUS=0; fi
fi

if [ -n "$CHECKPOINT_PID" ]; then kill "$CHECKPOINT_PID" >/dev/null 2>&1 || true; fi

SYNC_STATUS=0
if [ -n "$ARTIFACT_DEST" ] && [ -d "$OUTPUT_DIR" ]; then
  sync_artifacts "$OUTPUT_DIR" "$ARTIFACT_DEST" final
  SYNC_STATUS=$?
fi
WORKER_FINAL_SYNC_DONE=1

if [ "$RUN_STATUS" -ne 0 ]; then exit "$RUN_STATUS"; fi
exit "$SYNC_STATUS"
