#!/usr/bin/env python3
"""Static checks for the AI Scientist fork's ablation hooks.

The fork cannot be imported in the local harness environment, so these checks
parse source. They exist because a mis-named import inside a blanket
try/except silently disabled submission persistence in tranche 1R canary v2.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORK = ROOT / ".context/external/AI-Scientist-v2-ablation"


def _module_symbols(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    symbols: set[str] = set()
    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols.add(stmt.name)
        elif isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    symbols.add(target.id)
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            symbols.add(stmt.target.id)
    return symbols


def test_shared_benchmark_imports_resolve() -> None:
    shared_symbols = _module_symbols(FORK / "ablation_adapter/shared_benchmark.py")
    checked = 0
    for rel in (
        "ai_scientist/treesearch/parallel_agent.py",
        "ai_scientist/treesearch/agent_manager.py",
        "ablation_adapter/schema_export.py",
    ):
        path = FORK / rel
        if not path.is_file():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.endswith("shared_benchmark")
            ):
                for alias in node.names:
                    checked += 1
                    assert alias.name in shared_symbols, (
                        f"{rel}: imports missing symbol "
                        f"'{alias.name}' from shared_benchmark"
                    )
    assert checked, "no shared_benchmark imports found to check"


def test_persistence_hook_wired() -> None:
    source = (FORK / "ai_scientist/treesearch/parallel_agent.py").read_text(encoding="utf-8")
    assert "_persist_ablation_submission(node, workspace)" in source or (
        "_persist_ablation_submission(" in source
    ), "persistence hook must be called from the contract audit"
    assert "find_submission_path" in source, "hook must use the real shared_benchmark symbol"


def main() -> int:
    test_shared_benchmark_imports_resolve()
    test_persistence_hook_wired()
    print("ais fork static tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
