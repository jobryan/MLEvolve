# Ablation Run and Node Schema

This schema is the compatibility contract for the MLEvolve / AI Scientist v2 ablation project.

Every system-specific runner must emit:

- one run record per scheduled run
- one node record per generated candidate, stage attempt, or evaluated artifact

The schema is intentionally system-neutral. MLEvolve and AI Scientist v2 should preserve their own details in structured fields, but common analysis must be possible from the shared columns alone.

## Files

- Run schema: `configs/ablations/schema/run.schema.json`
- Node schema: `configs/ablations/schema/node.schema.json`
- Run samples: `configs/ablations/schema/sample_runs.jsonl`
- Node samples: `configs/ablations/schema/sample_nodes.jsonl`
- Validator: `scripts/validate_ablation_schema.py`

## Required Principles

1. Every result must be traceable to a run manifest, config, prompt, generated artifact, grader output, and audit result where applicable.
2. Failed, invalid, timed-out, and audit-failed outcomes are first-class records, not dropped rows.
3. Shared fields must be populated even when the system has a different internal vocabulary.
4. System-specific details belong in structured object fields such as `artifacts`, `memory_sources`, `selection`, `diversity`, and `error`.
5. Schema changes require a `schema_version` bump and a note in `.context/ablation/schema_changes.md`.

## Shared Operator Taxonomy

Use one of these operator values for every node:

- `Draft`
- `Debug`
- `Improve`
- `Evolution`
- `Fusion/Crossover`
- `Aggregation`
- `Ablation`
- `Review`
- `Memory`
- `Evaluation`

Mapping examples:

- MLEvolve initial code generation: `Draft`
- MLEvolve improve agent: `Improve`
- MLEvolve evolution agent: `Evolution`
- MLEvolve fusion agent: `Fusion/Crossover`
- MLEvolve aggregation agent: `Aggregation`
- AI Scientist v2 initial implementation stage: `Draft`
- AI Scientist v2 baseline tuning stage: `Improve`
- AI Scientist v2 creative research stage: `Evolution` or `Improve`
- AI Scientist v2 ablation studies stage: `Ablation`
- AI Scientist v2 writeup/review stage: `Review`

## Run Record

A run record describes one scheduled execution of one system, variant, task, and seed.

Key required fields:

- `schema_version`
- `run_id`
- `system`
- `benchmark_track`
- `task_id`
- `variant_id`
- `phase`
- `seed`
- `status`
- `budget`
- `grader`
- `artifacts`

Important interpretation fields:

- `metrics`: run-level summary metrics such as best validation score, final grader score, valid submission rate, and medal threshold status.
- `audit_summary`: top-candidate audit status and failure reasons.
- `actual`: actual runtime, token, dollar, and node counts.
- `error`: structured error information for failed or blocked runs.

## Node Record

A node record describes one generated or evaluated candidate inside a run.

Key required fields:

- `schema_version`
- `node_id`
- `run_id`
- `system`
- `task_id`
- `variant_id`
- `seed`
- `operator`
- `status`
- `created_at`
- `artifacts`

Important interpretation fields:

- `parent_node_ids`: parent or reference nodes used to create this node.
- `memory_sources`: child history, global retrieval, journal summary, archive records, or cold-start guidance used in the prompt.
- `selection`: selected-by policy, selection score, UCT components, novelty score, and selection rationale.
- `diversity`: strategy labels, embedding path, nearest-neighbor distance, and entropy bucket where available.
- `validation_score` and `final_score`: proxy/search metric and held-out grader metric, respectively.
- `error`: structured failure information for invalid, crashed, timed-out, or audit-failed nodes.

## Validation

Run:

```bash
python3 scripts/validate_ablation_schema.py
```

The default command validates the sample run and node records against their schemas.

Specific validation:

```bash
python3 scripts/validate_ablation_schema.py \
  --schema configs/ablations/schema/run.schema.json \
  --jsonl path/to/runs.jsonl
```

The validator is intentionally lightweight and uses only the Python standard library. It checks required fields, simple type constraints, and enums. Later tasks may add richer validation if needed.
