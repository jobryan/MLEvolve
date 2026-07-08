#!/usr/bin/env python3
"""Validate ablation JSONL records against the lightweight schema specs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAIRS = [
    (
        ROOT / "configs/ablations/schema/run.schema.json",
        ROOT / "configs/ablations/schema/sample_runs.jsonl",
    ),
    (
        ROOT / "configs/ablations/schema/node.schema.json",
        ROOT / "configs/ablations/schema/sample_nodes.jsonl",
    ),
]


TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "object": dict,
    "array": list,
    "null": type(None),
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    records: list[tuple[int, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: record must be a JSON object")
            records.append((line_no, value))
    return records


def allowed_types(spec: dict[str, Any]) -> tuple[type[Any], ...]:
    raw = spec.get("type")
    names = raw if isinstance(raw, list) else [raw]
    types: list[type[Any]] = []
    for name in names:
        if name not in TYPE_MAP:
            raise ValueError(f"unknown schema type {name!r}")
        mapped = TYPE_MAP[name]
        if isinstance(mapped, tuple):
            types.extend(mapped)
        else:
            types.append(mapped)
    return tuple(types)


def check_type(value: Any, types: tuple[type[Any], ...]) -> bool:
    if bool in types and isinstance(value, bool):
        return True
    if int in types and isinstance(value, bool):
        return False
    if float in types and isinstance(value, bool):
        return False
    return isinstance(value, types)


def validate_record(schema: dict[str, Any], record: dict[str, Any], source: str) -> list[str]:
    errors: list[str] = []
    required = schema.get("required", [])
    fields = schema.get("fields", {})

    for field in required:
        if field not in record:
            errors.append(f"{source}: missing required field {field!r}")

    for field, value in record.items():
        spec = fields.get(field)
        if spec is None:
            errors.append(f"{source}: unknown field {field!r}")
            continue

        types = allowed_types(spec)
        if not check_type(value, types):
            expected = spec.get("type")
            errors.append(f"{source}: field {field!r} expected {expected}, got {type(value).__name__}")
            continue

        if "enum" in spec and value not in spec["enum"]:
            errors.append(f"{source}: field {field!r} value {value!r} not in enum {spec['enum']!r}")

    return errors


def validate_pair(schema_path: Path, jsonl_path: Path) -> list[str]:
    schema = load_json(schema_path)
    errors: list[str] = []
    for line_no, record in iter_jsonl(jsonl_path):
        errors.extend(validate_record(schema, record, f"{jsonl_path}:{line_no}"))
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", type=Path, help="Schema JSON file.")
    parser.add_argument("--jsonl", type=Path, help="JSONL records to validate.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if bool(args.schema) != bool(args.jsonl):
        print("--schema and --jsonl must be provided together", file=sys.stderr)
        return 2

    pairs = [(args.schema, args.jsonl)] if args.schema else DEFAULT_PAIRS
    errors: list[str] = []

    for schema_path, jsonl_path in pairs:
        errors.extend(validate_pair(schema_path, jsonl_path))

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    for _, jsonl_path in pairs:
        print(f"validated {jsonl_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
