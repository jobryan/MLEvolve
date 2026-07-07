"""Pack an experience store directory into a deterministic tarball for S3.

Deterministic (sorted entries, zeroed mtimes/owners), so the same store state
always produces the same tarball sha256. Prints both identifiers the manifests
need: the store snapshot hash (content identity, verified again on-worker) and
the tarball sha256 (transport integrity, verified before extraction).

Usage:
    python -m experience.pack_store --store-dir stores/e1_fold_a --output stores/e1_fold_a.tar.gz
"""

import argparse
import gzip
import hashlib
import io
import sys
import tarfile
from pathlib import Path
from typing import List, Optional

from .store import compute_snapshot_hash


def pack(store_dir: Path, output: Path) -> str:
    files = sorted(p for p in store_dir.rglob("*") if p.is_file())
    if not files:
        raise SystemExit(f"{store_dir} contains no files")

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for p in files:
            info = tar.gettarinfo(str(p), arcname=str(p.relative_to(store_dir)))
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            with open(p, "rb") as f:
                tar.addfile(info, f)

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "wb") as f:
        with gzip.GzipFile(filename="", fileobj=f, mode="wb", mtime=0) as gz:
            gz.write(buf.getvalue())
    return hashlib.sha256(output.read_bytes()).hexdigest()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    snapshot = compute_snapshot_hash(args.store_dir)
    tarball_sha = pack(args.store_dir, args.output)
    print(f"store snapshot:  {snapshot}   (--store-snapshot-fold-*)")
    print(f"tarball sha256:  {tarball_sha}   (--store-tarball-sha-fold-*)")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
