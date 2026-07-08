#!/usr/bin/env python3
"""Archive oversized MLEvolve log directories to S3."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_DESTINATION_URI = (
    "s3://autoresearch-experiments-058264252788-us-east-1/"
    "mlevolve-ai-scientist-v2-ablation"
)


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"destination must be an s3:// URI: {uri!r}")
    without_scheme = uri.removeprefix("s3://")
    bucket, _, prefix = without_scheme.partition("/")
    if not bucket:
        raise ValueError(f"destination must include an S3 bucket: {uri!r}")
    return bucket, prefix.strip("/")


def iter_log_groups(root: Path) -> dict[Path, list[Path]]:
    groups: dict[Path, list[Path]] = {}
    for path in root.rglob("MLEvolve*.log"):
        if path.is_file():
            groups.setdefault(path.parent, []).append(path)
    return groups


def log_group_size(paths: list[Path]) -> int:
    return sum(path.stat().st_size for path in paths if path.exists())


def s3_key(prefix: str, archive_id: str, root: Path, path: Path) -> str:
    relative_path = path.relative_to(root)
    parts = [prefix, "manual-log-archives", archive_id, str(relative_path)]
    return "/".join(part.strip("/") for part in parts if part)


def archive_group(
    *,
    client: Any,
    bucket: str,
    prefix: str,
    archive_id: str,
    root: Path,
    log_dir: Path,
    paths: list[Path],
    delete_local: bool,
    dry_run: bool,
) -> dict[str, Any]:
    total_bytes = log_group_size(paths)
    uploads = []
    for path in sorted(paths):
        key = s3_key(prefix, archive_id, root, path)
        uploads.append({"path": str(path), "s3_uri": f"s3://{bucket}/{key}", "bytes": path.stat().st_size})
        if not dry_run:
            client.upload_file(path, bucket, key)

    if delete_local and not dry_run:
        for path in sorted(paths):
            path.unlink()

    return {
        "log_dir": str(log_dir),
        "total_bytes": total_bytes,
        "delete_local": delete_local,
        "dry_run": dry_run,
        "uploads": uploads,
    }


class S3Uploader:
    def __init__(self) -> None:
        try:
            import boto3  # type: ignore[import-not-found]

            self._client = boto3.client("s3")
        except ModuleNotFoundError:
            self._client = None

    def upload_file(self, path: Path, bucket: str, key: str) -> None:
        if self._client is not None:
            self._client.upload_file(str(path), bucket, key)
            return

        subprocess.run(
            [
                "aws",
                "s3",
                "cp",
                str(path),
                f"s3://{bucket}/{key}",
                "--only-show-errors",
            ],
            check=True,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Directory tree to scan for MLEvolve*.log files.",
    )
    parser.add_argument(
        "--destination-uri",
        default=os.environ.get("MLEVOLVE_LOG_BACKUP_URI")
        or os.environ.get("ABLATION_ARTIFACT_ROOT")
        or DEFAULT_DESTINATION_URI,
        help="S3 root where log archives should be stored.",
    )
    parser.add_argument(
        "--threshold-bytes",
        type=int,
        default=1073741824,
        help="Archive log directories whose combined MLEvolve*.log size exceeds this value.",
    )
    parser.add_argument(
        "--delete-local",
        action="store_true",
        help="Delete local log files after every file in the group uploads successfully.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be uploaded without uploading or deleting files.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Optional JSON manifest path describing archived files.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = args.root.resolve()
    bucket, prefix = parse_s3_uri(args.destination_uri)
    archive_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    client = None if args.dry_run else S3Uploader()

    archived_groups = []
    for log_dir, paths in sorted(iter_log_groups(root).items()):
        if log_group_size(paths) < args.threshold_bytes:
            continue
        archived_groups.append(
            archive_group(
                client=client,
                bucket=bucket,
                prefix=prefix,
                archive_id=archive_id,
                root=root,
                log_dir=log_dir,
                paths=paths,
                delete_local=args.delete_local,
                dry_run=args.dry_run,
            )
        )

    manifest = {
        "archive_id": archive_id,
        "root": str(root),
        "destination_uri": args.destination_uri,
        "threshold_bytes": args.threshold_bytes,
        "groups": archived_groups,
    }
    payload = json.dumps(manifest, indent=2)
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
