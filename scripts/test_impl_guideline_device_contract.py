#!/usr/bin/env python3
"""Checks for the CPU-only and offline prompt contracts in the impl guideline."""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# The local harness environment does not ship the agent runtime deps.
if "humanize" not in sys.modules:
    fake_humanize = types.ModuleType("humanize")
    fake_humanize.naturaldelta = lambda seconds: f"{int(seconds)}s"
    sys.modules["humanize"] = fake_humanize

from agents.prompts.impl_guideline import get_impl_guideline


def _guideline_text(**env: str) -> str:
    saved = {key: os.environ.get(key) for key in ("MLEVOLVE_NO_GPU", "MLEVOLVE_OFFLINE")}
    try:
        for key in saved:
            os.environ.pop(key, None)
        os.environ.update(env)
        guideline = get_impl_guideline(
            tot_time_remaining=3600.0,
            steps_remaining=5,
            exec_timeout=600,
        )
        return "\n".join(guideline["Implementation guideline"])
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_no_gpu_contract_gated() -> None:
    default_text = _guideline_text()
    assert "CPU-only execution environment (HARD CONSTRAINT)" not in default_text

    gated_text = _guideline_text(MLEVOLVE_NO_GPU="1")
    assert "CPU-only execution environment (HARD CONSTRAINT)" in gated_text
    assert "torch.cuda.is_available()" in gated_text
    assert "unconditional `device='cuda'`" in gated_text


def test_offline_contract_gated() -> None:
    default_text = _guideline_text()
    assert "torch.hub.load(), HuggingFace, etc. available" in default_text
    assert "OFFLINE" not in default_text

    offline_text = _guideline_text(MLEVOLVE_OFFLINE="1")
    assert "This environment is OFFLINE" in offline_text
    assert "torch.hub.load(), HuggingFace, etc. available" not in offline_text


def test_ais_contract_source_gated() -> None:
    source = (
        ROOT
        / ".context/external/AI-Scientist-v2-ablation/ai_scientist/treesearch/parallel_agent.py"
    ).read_text(encoding="utf-8")
    assert 'os.getenv("MLEVOLVE_NO_GPU") == "1"' in source
    assert 'os.getenv("MLEVOLVE_OFFLINE") == "1"' in source
    assert "this environment has NO GPU" in source
    assert "this environment is OFFLINE" in source


def main() -> int:
    test_no_gpu_contract_gated()
    test_offline_contract_gated()
    test_ais_contract_source_gated()
    print("impl guideline device contract tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
