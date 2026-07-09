"""Render AWS Batch job specs from E2 run manifests.

Takes the JSONL produced by `python -m experience.build_e2_manifests` and emits
one Batch submit-job spec per line (same shape as the phoenix harness's job
files). The worker command mirrors the anchored MLEvolve command skeleton —
prepare dataset -> fetch worker patch -> run manifest -> grade -> sync — with
the AIS-specific recovery block removed and three E2 additions:

  1. The worker patch tarball's sha256 is verified against
     `runtime_controls.worker_patch_sha256` before extraction (anchor integrity).
  2. Arms B/C/D download the experience store tarball from
     `runtime_controls.store_uri` into `.context/e2_store` and verify its
     snapshot hash against `runtime_controls.store_snapshot` using the
     experience package that arrives with the worker patch. A placebo/real
     store mix-up therefore fails the job instead of silently polluting an arm.
  3. `timeout.attemptDurationSeconds` comes from the manifest
     (wall_time + 900s), so overruns cannot destroy artifacts.

Usage:
    python -m experience.render_e2_jobs \
        --manifests .context/e2/manifests_arm_a.jsonl \
        --manifest-s3-prefix s3://<bucket>/e2-study/manifests/e2e1-v1 \
        --artifact-root s3://<bucket>/e2-study/artifacts \
        --output .context/e2/jobs_arm_a.jsonl
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_QUEUE = "mlevolve-e2-study-queue-serial4"

SAFE_FETCH_PY = (
    "import boto3,hashlib,sys,tarfile,tempfile\n"
    "from pathlib import Path\n"
    "uri, dest_s, expected = sys.argv[1], sys.argv[2], sys.argv[3]\n"
    "dest = Path(dest_s).resolve()\n"
    "bucket, key = uri.removeprefix('s3://').split('/', 1)\n"
    "tmp = Path(tempfile.mkdtemp(prefix='e2-fetch-')) / 'archive.tar.gz'\n"
    "boto3.client('s3').download_file(bucket, key, str(tmp))\n"
    "h = hashlib.sha256(tmp.read_bytes()).hexdigest()\n"
    "if expected and expected != 'SKIP' and h != expected:\n"
    "    raise SystemExit(f'sha256 mismatch for {uri}: got {h}, expected {expected}')\n"
    "tar = tarfile.open(tmp, 'r:gz')\n"
    "members = tar.getmembers()\n"
    "for member in members:\n"
    "    target = (dest / member.name).resolve()\n"
    "    if dest not in target.parents and target != dest:\n"
    "        raise RuntimeError(f'unsafe path in archive: {member.name}')\n"
    "dest.mkdir(parents=True, exist_ok=True)\n"
    "tar.extractall(dest, members=members)\n"
    "tar.close()\n"
    "print(f'fetched {uri} -> {dest} (sha256 {h})')\n"
)

VERIFY_STORE_PY = (
    "import sys\n"
    "from experience.store import compute_snapshot_hash\n"
    "got = compute_snapshot_hash(sys.argv[1])\n"
    "expected = sys.argv[2]\n"
    "if expected and expected != 'SKIP' and got != expected:\n"
    "    raise SystemExit(f'store snapshot mismatch: got {got}, expected {expected}')\n"
    "print(f'store snapshot verified: {got}')\n"
)


def build_command(manifest: Dict[str, Any], manifest_uri: str, artifact_root: str) -> str:
    rc = manifest["runtime_controls"]
    run_id = manifest["run_id"]
    task_id = manifest["task"]["id"]
    phase = manifest["phase"]
    output_dir = manifest["artifacts"]["output_dir"]

    parts = [
        "set -euo pipefail",
        f"SYSTEM=mlevolve; TASK_ID={task_id}; PHASE={phase}; RUN_ID={run_id}",
        f"MANIFEST_REF={manifest_uri}",
        f"OUTPUT_DIR={output_dir}",
        'test -n "${MLEBENCH_DATASET_DIR:-}" || { echo "MLEBENCH_DATASET_DIR is required" >&2; exit 2; }',
        'DESCRIPTION_PATH="${MLEBENCH_DATASET_DIR}/${TASK_ID}/prepared/public/description.md"',
        'if [ ! -f "$DESCRIPTION_PATH" ]; then mlebench prepare -c "$TASK_ID" --data-dir "$MLEBENCH_DATASET_DIR"; fi',
        # worker patch: fetch + sha256-verify + safe-extract into CWD
        f"python3 -c {shell_quote(SAFE_FETCH_PY)} {rc['worker_patch_uri']} . {rc.get('worker_patch_sha256') or 'SKIP'}",
    ]

    if rc.get("store_uri"):
        parts += [
            f"python3 -c {shell_quote(SAFE_FETCH_PY)} {rc['store_uri']} .context/e2_store {rc.get('store_tarball_sha256') or 'SKIP'}",
            f"python3 -c {shell_quote(VERIFY_STORE_PY)} .context/e2_store {rc.get('store_snapshot') or 'SKIP'}",
        ]

    # In-container watchdog: kill the agent run 900s before the Batch attempt
    # timeout so grading + artifact sync always execute — a Batch-level kill
    # destroys artifacts (journals included), turning timeouts into data loss.
    run_timeout = manifest["runtime_controls"]["batch_attempt_timeout_seconds"] - 900
    parts += [
        "set +e",
        f'timeout -k 60 {run_timeout} python3 scripts/run_ablation_manifest.py --manifest "$MANIFEST_REF"; RUN_STATUS=$?',
        'if [ "$RUN_STATUS" -eq 124 ]; then echo "agent run hit in-container watchdog (${RUN_STATUS})"; fi',
        'if [ -d "$OUTPUT_DIR" ]; then python3 scripts/grade_ablation_submissions.py --output-dir "$OUTPUT_DIR" --task-id "$TASK_ID" --system "$SYSTEM" --run-id "$RUN_ID" --data-dir "$MLEBENCH_DATASET_DIR" || true; fi',
        "SYNC_STATUS=0",
        f'ARTIFACT_DEST="{artifact_root.rstrip("/")}/${{PHASE}}/${{RUN_ID}}/"',
        'if [ -d "$OUTPUT_DIR" ]; then python3 -c '
        + shell_quote(
            "import sys,boto3\n"
            "from pathlib import Path\n"
            "src = Path(sys.argv[1]); dest = sys.argv[2].rstrip('/') + '/'\n"
            "bucket, key = dest.removeprefix('s3://').split('/', 1)\n"
            "s3 = boto3.client('s3')\n"
            "for p in src.rglob('*'):\n"
            "    if p.is_file():\n"
            "        s3.upload_file(str(p), bucket, key + str(p.relative_to(src)))\n"
        )
        + ' "$OUTPUT_DIR" "$ARTIFACT_DEST"; SYNC_STATUS=$?; fi',
        'if [ "$RUN_STATUS" -ne 0 ]; then exit "$RUN_STATUS"; fi',
        'exit "$SYNC_STATUS"',
    ]
    return "; ".join(parts)


def shell_quote(text: str) -> str:
    return "'" + text.replace("'", "'\"'\"'") + "'"


def render_job(
    manifest: Dict[str, Any],
    manifest_s3_prefix: str,
    artifact_root: str,
    queue: str,
    vcpus: int,
    memory_mb: int,
) -> Dict[str, Any]:
    run_id = manifest["run_id"]
    manifest_uri = f"{manifest_s3_prefix.rstrip('/')}/{run_id}.json"
    job_definition = manifest["anchor"]["job_definition"]
    timeout_seconds = manifest["runtime_controls"]["batch_attempt_timeout_seconds"]

    environment = [
        {"name": "ABLATION_RUN_ID", "value": run_id},
        {"name": "ABLATION_SYSTEM", "value": "mlevolve"},
        {"name": "ABLATION_VARIANT_ID", "value": manifest["variant"]["variant_id"]},
        {"name": "ABLATION_TASK_ID", "value": manifest["task"]["id"]},
        {"name": "ABLATION_PHASE", "value": manifest["phase"]},
        {"name": "MLEVOLVE_NO_GPU", "value": "1"},
        {"name": "MLEVOLVE_OFFLINE", "value": "1"},
        {"name": "CUDA_VISIBLE_DEVICES", "value": "-1"},
    ]

    return {
        "jobName": run_id,
        "jobQueue": queue,
        "jobDefinition": job_definition,
        "parameters": {
            "phase": manifest["phase"],
            "run_id": run_id,
            "seed": str(manifest["seed"]),
            "system": "mlevolve",
            "task_id": manifest["task"]["id"],
            "variant_id": manifest["variant"]["variant_id"],
        },
        "timeout": {"attemptDurationSeconds": timeout_seconds},
        "containerOverrides": {
            "command": ["bash", "-lc", build_command(manifest, manifest_uri, artifact_root)],
            "environment": environment,
            "resourceRequirements": [
                {"type": "VCPU", "value": str(vcpus)},
                {"type": "MEMORY", "value": str(memory_mb)},
            ],
        },
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifests", required=True, type=Path)
    parser.add_argument("--manifest-s3-prefix", required=True,
                        help="S3 prefix where per-run manifest JSONs will be uploaded")
    parser.add_argument("--artifact-root", required=True, help="S3 prefix for run artifact sync")
    parser.add_argument("--queue", default=DEFAULT_QUEUE)
    parser.add_argument("--vcpus", type=int, default=4)
    parser.add_argument("--memory-mb", type=int, default=16384)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    manifests = [json.loads(l) for l in args.manifests.read_text().splitlines() if l.strip()]
    jobs = [
        render_job(m, args.manifest_s3_prefix, args.artifact_root, args.queue, args.vcpus, args.memory_mb)
        for m in manifests
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for job in jobs:
            f.write(json.dumps(job, ensure_ascii=False) + "\n")

    with_store = sum(1 for m in manifests if m["runtime_controls"].get("store_uri"))
    print(f"rendered {len(jobs)} job specs -> {args.output}")
    print(f"queue={args.queue} vcpus={args.vcpus} memory={args.memory_mb}MB "
          f"store-download jobs={with_store} timeout(s)={jobs[0]['timeout']['attemptDurationSeconds'] if jobs else 'n/a'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
