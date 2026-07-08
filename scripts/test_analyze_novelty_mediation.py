#!/usr/bin/env python3
"""Tests for scripts/analyze_novelty_mediation.py with synthetic nodes.jsonl fixtures."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_novelty_mediation as anm  # noqa: E402

ANALYZER = ROOT / "scripts/analyze_novelty_mediation.py"


def make_node(
    node_id: str,
    run_id: str,
    validation_score: float | None,
    nn_distance: float | None,
    novelty_lambda: float,
    labels: dict[str, str],
    metric_direction: str = "minimize",
    embedding_backend: str = "hashing-token-v1",
) -> dict:
    return {
        "schema_version": "1.0",
        "node_id": node_id,
        "run_id": run_id,
        "system": "mlevolve",
        "task_id": "task-1",
        "variant_id": "novelty_sweep",
        "seed": 1,
        "operator": "Draft",
        "status": "success",
        "created_at": "2026-07-08T10:00:00Z",
        "metric_direction": metric_direction,
        "validation_score": validation_score,
        "diversity": {
            "strategy_labels": labels,
            "nearest_neighbor_distance": nn_distance,
            "embedding_backend": embedding_backend,
            "novelty_lambda_in_effect": novelty_lambda,
            "embedding_path": f"embeddings/{node_id}.json",
        },
        "artifacts": {},
    }


def write_fixture(path: Path) -> None:
    nodes = [
        # run-a: lambda 0.0, identical labels (entropy 0), low distances
        make_node("a1", "run-a", 0.50, None, 0.0, {"model_family": "tree_boosting"}),
        make_node("a2", "run-a", 0.45, 0.10, 0.0, {"model_family": "tree_boosting"}),
        make_node("a3", "run-a", 0.40, 0.20, 0.0, {"model_family": "tree_boosting"}),
        # run-b: lambda 0.5, mixed labels, medium distances
        make_node("b1", "run-b", 0.42, None, 0.5, {"model_family": "tree_boosting"}),
        make_node("b2", "run-b", 0.38, 0.40, 0.5, {"model_family": "neural_network"}),
        # run-c: lambda 1.0, all-distinct labels, high distances, semantic backend
        make_node(
            "c1", "run-c", 0.30, 0.60, 1.0,
            {"model_family": "linear"}, embedding_backend="semantic-bge-base-en-v1.5-v1",
        ),
        make_node(
            "c2", "run-c", 0.36, 0.80, 1.0,
            {"model_family": "neural_network"}, embedding_backend="semantic-bge-base-en-v1.5-v1",
        ),
    ]
    path.write_text(
        "".join(json.dumps(node, sort_keys=True) + "\n" for node in nodes),
        encoding="utf-8",
    )


def test_pearson() -> None:
    assert anm.pearson([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == 1.0
    assert anm.pearson([1.0, 2.0, 3.0], [6.0, 4.0, 2.0]) == -1.0
    assert anm.pearson([1.0, 2.0], [5.0, 5.0]) is None  # zero variance
    assert anm.pearson([1.0], [2.0]) is None  # n < 2


def test_entropy_and_signature() -> None:
    assert anm.shannon_entropy([]) is None
    assert anm.shannon_entropy(["x", "x", "x"]) == 0.0
    assert abs(anm.shannon_entropy(["x", "y"]) - 1.0) < 1e-9
    # dict and list strategy label shapes both collapse deterministically
    assert anm.strategy_signature({"b": "2", "a": "1"}) == "a=1|b=2"
    assert anm.strategy_signature(["linear", "ensemble"]) == "ensemble|linear"
    assert anm.strategy_signature(None) is None


def test_report_per_run_metrics() -> None:
    with tempfile.TemporaryDirectory() as raw_tmp:
        fixture = Path(raw_tmp) / "nodes.jsonl"
        write_fixture(fixture)
        report = anm.build_report([fixture])

        assert report["node_count"] == 7
        assert report["run_count"] == 3
        by_run = {summary["run_id"]: summary for summary in report["runs"]}

        run_a = by_run["run-a"]
        assert run_a["node_count"] == 3
        assert abs(run_a["mean_nn_distance"] - 0.15) < 1e-9  # None distances ignored
        assert run_a["strategy_label_entropy_bits"] == 0.0  # identical labels
        assert run_a["novelty_lambda"] == 0.0
        assert run_a["best_validation_score"] == 0.40  # minimize -> min

        run_b = by_run["run-b"]
        assert abs(run_b["strategy_label_entropy_bits"] - 1.0) < 1e-9
        assert run_b["best_validation_score"] == 0.38

        run_c = by_run["run-c"]
        assert abs(run_c["mean_nn_distance"] - 0.70) < 1e-9
        assert run_c["embedding_backends"] == ["semantic-bge-base-en-v1.5-v1"]
        assert run_c["best_validation_score"] == 0.30


def test_correlation_table() -> None:
    with tempfile.TemporaryDirectory() as raw_tmp:
        fixture = Path(raw_tmp) / "nodes.jsonl"
        write_fixture(fixture)
        report = anm.build_report([fixture])
        correlations = report["correlations"]

        for target in ("mean_nn_distance", "strategy_label_entropy_bits", "best_validation_score"):
            entry = correlations[f"novelty_lambda_vs_{target}"]
            assert entry["n_runs"] == 3
            assert entry["pearson_r"] is not None

        # Higher lambda -> larger NN distances in the fixture (positive r);
        # metric is minimize, so best score falls as lambda rises (negative r).
        assert correlations["novelty_lambda_vs_mean_nn_distance"]["pearson_r"] > 0.9
        assert correlations["novelty_lambda_vs_best_validation_score"]["pearson_r"] < -0.9


def test_cli_end_to_end() -> None:
    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        fixture = tmp / "nodes.jsonl"
        json_out = tmp / "report.json"
        write_fixture(fixture)
        result = subprocess.run(
            [sys.executable, str(ANALYZER), str(fixture), "--json-out", str(json_out)],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert result.returncode == 0, result.stderr
        assert "run-a" in result.stdout
        assert "pearson_r" in result.stdout
        saved = json.loads(json_out.read_text(encoding="utf-8"))
        assert saved["run_count"] == 3

        missing = subprocess.run(
            [sys.executable, str(ANALYZER), str(tmp / "does-not-exist.jsonl")],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert missing.returncode == 1


def main() -> int:
    test_pearson()
    test_entropy_and_signature()
    test_report_per_run_metrics()
    test_correlation_table()
    test_cli_end_to_end()
    print("analyze_novelty_mediation tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
