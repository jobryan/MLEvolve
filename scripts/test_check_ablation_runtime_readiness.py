#!/usr/bin/env python3
"""Tests for check_ablation_runtime_readiness.py."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
READINESS = ROOT / "scripts/check_ablation_runtime_readiness.py"


def test_runtime_readiness_report() -> None:
    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        output_json = tmp / "readiness.json"
        output_md = tmp / "readiness.md"
        result = subprocess.run(
            [
                sys.executable,
                str(READINESS),
                "--output-json",
                str(output_json),
                "--output-md",
                str(output_md),
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert result.returncode in {0, 1}
        assert output_json.exists()
        assert output_md.exists()

        payload = json.loads(output_json.read_text(encoding="utf-8"))
        assert payload["overall_status"] in {"ready", "blocked"}
        assert "screening" in payload["groups"]

        markdown = output_md.read_text(encoding="utf-8")
        assert "This report only records whether credentials are present." in markdown
        for env_name in ("OPENAI_API_KEY", "KAGGLE_KEY"):
            value = os.environ.get(env_name)
            if value:
                assert value not in markdown


if __name__ == "__main__":
    test_runtime_readiness_report()
    print("check_ablation_runtime_readiness tests passed")
