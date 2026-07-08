#!/usr/bin/env python3
"""Verify that Kaggle competition download endpoints work for ablation tasks."""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


DEFAULT_REGION = "us-east-1"
DEFAULT_USERNAME_SECRET = "mlevolve-ai-scientist-v2-ablation/kaggle/username"
DEFAULT_KEY_SECRET = "mlevolve-ai-scientist-v2-ablation/kaggle/key"
DOWNLOAD_URL_TEMPLATE = "https://www.kaggle.com/api/v1/competitions/data/download-all/{competition}"


@dataclass(frozen=True)
class DownloadCheck:
    competition: str
    ok: bool
    status: int | None = None
    reason: str = ""
    content_type: str = ""
    content_range: str = ""
    body: str = ""
    error: str = ""


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            records.append(value)
    return records


def task_id_from_record(record: dict[str, Any]) -> str | None:
    if record.get("task_id"):
        return str(record["task_id"])

    task = record.get("task")
    if isinstance(task, dict) and task.get("id"):
        return str(task["id"])

    parameters = record.get("parameters")
    if isinstance(parameters, dict):
        for key in ("task_id", "task"):
            if parameters.get(key):
                return str(parameters[key])

    tags = record.get("tags")
    if isinstance(tags, dict) and tags.get("task"):
        return str(tags["task"])

    container = record.get("containerOverrides")
    if isinstance(container, dict):
        for env in container.get("environment", []):
            if isinstance(env, dict) and env.get("name") == "ABLATION_TASK_ID" and env.get("value"):
                return str(env["value"])

    return None


def competition_ids_from_records(records: Iterable[dict[str, Any]]) -> list[str]:
    competitions = {task_id for record in records if (task_id := task_id_from_record(record))}
    return sorted(competitions)


def competition_ids_from_task_manifest(path: Path, smoke_only: bool) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tasks = payload.get("tasks", [])
    if not isinstance(tasks, list):
        raise ValueError(f"{path}: expected tasks list")
    competitions = []
    for task in tasks:
        if not isinstance(task, dict) or not task.get("id"):
            continue
        if smoke_only and not task.get("smoke", False):
            continue
        competitions.append(str(task["id"]))
    return sorted(set(competitions))


def aws_secret_value(secret_id: str, aws_cli: str, region: str) -> str:
    command = [
        aws_cli,
        "secretsmanager",
        "get-secret-value",
        "--region",
        region,
        "--secret-id",
        secret_id,
        "--query",
        "SecretString",
        "--output",
        "text",
    ]
    completed = subprocess.run(command, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or f"failed to read secret {secret_id}")
    return completed.stdout.strip()


def credentials_from_args(args: argparse.Namespace) -> tuple[str, str]:
    if args.credential_source == "env":
        username = args.kaggle_username or ""
        key = args.kaggle_key or ""
        if not username or not key:
            raise ValueError("KAGGLE_USERNAME and KAGGLE_KEY must be set for --credential-source env")
        return username, key

    username = aws_secret_value(args.username_secret, args.aws_cli, args.region)
    key = aws_secret_value(args.key_secret, args.aws_cli, args.region)
    if not username or not key:
        raise ValueError("Kaggle username/key secrets must be non-empty")
    return username, key


def check_download(
    competition: str,
    username: str,
    key: str,
    *,
    timeout: int,
    urlopen: Callable[..., Any] = urllib.request.urlopen,
) -> DownloadCheck:
    token = base64.b64encode(f"{username}:{key}".encode("utf-8")).decode("ascii")
    request = urllib.request.Request(
        DOWNLOAD_URL_TEMPLATE.format(competition=competition),
        method="GET",
        headers={
            "Authorization": f"Basic {token}",
            "Range": "bytes=0-0",
            "User-Agent": "mlevolve-ablation-preflight/1.0",
        },
    )

    try:
        response = urlopen(request, timeout=timeout)
        try:
            status = int(response.getcode())
            headers = response.headers
            ok = status in {200, 206, 302, 303, 307, 308}
            return DownloadCheck(
                competition=competition,
                ok=ok,
                status=status,
                content_type=headers.get("Content-Type", ""),
                content_range=headers.get("Content-Range", ""),
            )
        finally:
            response.close()
    except urllib.error.HTTPError as exc:
        body = exc.read(300).decode("utf-8", errors="replace").replace("\n", " ")
        return DownloadCheck(
            competition=competition,
            ok=False,
            status=exc.code,
            reason=exc.reason,
            body=body,
        )
    except Exception as exc:  # pragma: no cover - exercised by live network failures.
        return DownloadCheck(
            competition=competition,
            ok=False,
            error=type(exc).__name__,
            body=str(exc),
        )


def collect_competitions(args: argparse.Namespace) -> list[str]:
    competitions = set(args.competition)
    for path in args.jobs_jsonl + args.manifest_jsonl:
        competitions.update(competition_ids_from_records(iter_jsonl(path)))
    if args.task_manifest:
        competitions.update(competition_ids_from_task_manifest(args.task_manifest, args.smoke_only))
    return sorted(competitions)


def summarize(username: str, checks: list[DownloadCheck]) -> dict[str, Any]:
    failed = [check for check in checks if not check.ok]
    return {
        "worker_kaggle_username": username,
        "checked_count": len(checks),
        "ok_count": len(checks) - len(failed),
        "failed_count": len(failed),
        "all_ok": not failed and bool(checks),
        "checks": [asdict(check) for check in checks],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competition", action="append", default=[], help="Competition slug; repeatable.")
    parser.add_argument("--jobs-jsonl", action="append", type=Path, default=[], help="AWS Batch job specs JSONL.")
    parser.add_argument("--manifest-jsonl", action="append", type=Path, default=[], help="Ablation manifests JSONL.")
    parser.add_argument("--task-manifest", type=Path, help="Task manifest JSON with a top-level tasks list.")
    parser.add_argument("--smoke-only", action="store_true", help="Only check smoke tasks from --task-manifest.")
    parser.add_argument("--credential-source", choices=["aws-secrets", "env"], default="aws-secrets")
    parser.add_argument("--username-secret", default=DEFAULT_USERNAME_SECRET)
    parser.add_argument("--key-secret", default=DEFAULT_KEY_SECRET)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--aws-cli", default="aws")
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    competitions = collect_competitions(args)
    if not competitions:
        print("provide --competition, --jobs-jsonl, --manifest-jsonl, or --task-manifest", file=sys.stderr)
        return 1

    username, key = credentials_from_args(args)
    checks = [check_download(competition, username, key, timeout=args.timeout) for competition in competitions]
    summary = summarize(username, checks)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["all_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
