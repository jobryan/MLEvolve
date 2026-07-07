# Ablation Evaluator Hardening

This document defines the first evaluator-hardening layer for the MLEvolve / AI Scientist v2 ablation project.

The audit layer does not replace MLE-bench grading. It catches obvious invalid or suspicious outputs before a run can be promoted as evidence for a component decision.

## Blocking Checks

These checks block node promotion:

- `missing_submission`: expected `submission.csv` does not exist.
- `missing_sample_submission`: sample submission fixture is unavailable for structural validation.
- `column_mismatch`: submission columns do not exactly match sample submission columns.
- `row_count_mismatch`: submission row count differs from sample submission row count.
- `empty_submission`: submission has no data rows.
- `non_finite_numeric_value`: numeric prediction field contains NaN, infinity, or an unparsable numeric value where the sample expects numeric output.
- `runtime_over_limit`: actual runtime exceeds the supplied runtime limit.
- `forbidden_code_pattern`: generated code contains obvious leakage or benchmark-circumvention patterns.

## Report-Only Checks

These checks should be reported but do not automatically block promotion until we have stronger evidence:

- suspicious references to public leaderboard files
- use of external network calls in generated code
- repeated identical predictions
- extreme constant predictions
- imports or file reads that are allowed in some tasks but suspicious in others

## Forbidden Code Pattern Examples

The initial audit script blocks obvious references to:

- `prepared/private`
- `answers.csv`
- `gold_submission`
- `leaderboard.csv`
- `private/test.csv`
- `/home/data/private`
- `/home/submission/submission.csv` reads from generated code

The last item is intended to catch code reading a previously written submission as hidden state. Writing `/home/submission/submission.csv` is allowed; reading it should be reviewed.

## Promotion Rule

A top candidate cannot support a `keep` recommendation unless:

1. It has a valid MLE-bench grader result.
2. It passes the blocking audit checks.
3. It is confirmed by repeated seed or stricter grader/audit status when used for finalist claims.

A high-scoring audit-failed candidate must be reported as `audit_failed` or provisional until repaired and re-confirmed.

## Script

Run:

```bash
python3 scripts/audit_submission.py \
  --run-id <run_id> \
  --node-id <node_id> \
  --submission path/to/submission.csv \
  --sample-submission path/to/sample_submission.csv \
  --code path/to/generated_code.py
```

The script emits JSON with:

- `run_id`
- `node_id`
- `status`
- `blocking_failures`
- `report_only_findings`
- `checks`

Status is `pass` when there are no blocking failures and `fail` otherwise.
