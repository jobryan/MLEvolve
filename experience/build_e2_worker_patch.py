"""Assemble the E2 worker patch: frozen v3 base tarball + this repo's code.

The E1 anchor commit (602291e, merged into this branch) covers the worker's
repo-side code but NOT the operational pieces that live only in the tarball
(phoenix's scripts/, .context/external/ mle-bench + fork snapshots). The E2
worker patch is therefore: the declared-frozen v3 tarball, with every path that
exists in this repo's HEAD replaced by our version (anchor + experience layer),
tarball-only paths preserved.

Steps: verify base tarball sha256 (refuses a silently-replaced base) ->
safe-extract -> overlay `git archive HEAD` of this repo -> repack
deterministically -> print the sha256 to pass to build_e2_manifests.

Usage:
    # base tarball fetched separately, e.g.:
    #   aws s3 cp s3://<bucket>/mlevolve-ai-scientist-v2-ablation/patches/tranche2_worker_patch_20260706_v3.tar.gz /tmp/v3.tar.gz
    python -m experience.build_e2_worker_patch --base-tarball /tmp/v3.tar.gz \
        --output /tmp/e2_worker_patch_$(date +%Y%m%d).tar.gz
"""

import argparse
import gzip
import hashlib
import io
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import List, Optional

V3_SHA256 = "7f01a9110b7fb2b1569ed62290892d221f42fba9b3b59cb4ebc56d2a166a3a3f"
REPO_ROOT = Path(__file__).resolve().parent.parent


def safe_extract(tarball: Path, dest: Path) -> None:
    with tarfile.open(tarball, "r:gz") as tar:
        members = tar.getmembers()
        for member in members:
            target = (dest / member.name).resolve()
            if dest.resolve() not in target.parents and target != dest.resolve():
                raise SystemExit(f"unsafe path in base tarball: {member.name}")
        tar.extractall(dest, members=members)


def overlay_repo(dest: Path, commit: str) -> int:
    """Extract `git archive <commit>` of this repo over dest; returns file count."""
    archive = subprocess.run(
        ["git", "archive", "--format=tar", commit],
        capture_output=True, cwd=REPO_ROOT, check=True,
    ).stdout
    count = 0
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r") as tar:
        for member in tar.getmembers():
            target = (dest / member.name).resolve()
            if dest.resolve() not in target.parents and target != dest.resolve():
                raise SystemExit(f"unsafe path in git archive: {member.name}")
            count += 1
        tar.extractall(dest)
    return count


def repack(src: Path, output: Path) -> str:
    files = sorted(p for p in src.rglob("*") if p.is_file() or p.is_symlink())
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for p in files:
            info = tar.gettarinfo(str(p), arcname=str(p.relative_to(src)))
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            if p.is_symlink():
                tar.addfile(info)
            else:
                with open(p, "rb") as f:
                    tar.addfile(info, f)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "wb") as f:
        with gzip.GzipFile(filename="", fileobj=f, mode="wb", mtime=0) as gz:
            gz.write(buf.getvalue())
    return hashlib.sha256(output.read_bytes()).hexdigest()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-tarball", required=True, type=Path)
    parser.add_argument("--expected-base-sha256", default=V3_SHA256)
    parser.add_argument("--commit", default="HEAD", help="Repo commit to overlay (default HEAD)")
    parser.add_argument("--overlay-dir", type=Path, action="append", default=[],
                        help="Extra directory tree(s) copied over the patch root after the repo overlay "
                             "(e.g. resolved mle-bench leaderboard CSVs that live outside the repo)")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    base_sha = hashlib.sha256(args.base_tarball.read_bytes()).hexdigest()
    if args.expected_base_sha256 and base_sha != args.expected_base_sha256:
        raise SystemExit(
            f"base tarball sha256 mismatch: got {base_sha}, expected {args.expected_base_sha256} — "
            "the frozen v3 anchor may have been replaced; stop and re-verify with phoenix."
        )

    commit_hash = subprocess.run(
        ["git", "rev-parse", args.commit], capture_output=True, text=True, cwd=REPO_ROOT, check=True,
    ).stdout.strip()

    with tempfile.TemporaryDirectory(prefix="e2-patch-") as tmp:
        work = Path(tmp) / "patch"
        work.mkdir()
        safe_extract(args.base_tarball, work)
        overlaid = overlay_repo(work, commit_hash)
        extra = 0
        for odir in args.overlay_dir:
            for p in sorted(odir.rglob("*")):
                if p.is_file():
                    target = work / p.relative_to(odir)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(p.read_bytes())
                    extra += 1
        if extra:
            print(f"extra overlay files: {extra}")
        sha = repack(work, args.output)

    print(f"base:    {args.base_tarball} (sha256 {base_sha[:16]}… verified)")
    print(f"overlay: {overlaid} entries from repo commit {commit_hash[:12]}")
    print(f"output:  {args.output}")
    print(f"sha256:  {sha}   (--worker-patch-sha256)")
    print(f"record e2_branch_commit={commit_hash[:12]} in the launch log")
    return 0


if __name__ == "__main__":
    sys.exit(main())
