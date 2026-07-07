# MLEvolve / AI Scientist v2 Ablation Summary

Status: scaffolded; empirical recommendations pending live experiments on target execution hosts
Date: 2026-06-23

## Objective

Determine how MLEvolve and AI Scientist v2 component choices affect research output and performance under comparable benchmark conditions.

The study must produce defensible recommendations for:

- memory
- search
- workflow decomposition
- model routing
- diversity and novelty
- operators
- evaluator hardening

Each final recommendation must be one of `keep`, `remove`, `modify`, or `needs more evidence`.

## Current Execution Status

Live empirical execution has not started. The local Conductor workspace is not dependency-complete for local smoke runs; the intended execution hosts, presumably AWS workers, still need to pass runtime readiness.

Primary blocker summary:

- MLEvolve runtime dependencies are not installed in the local workspace.
- AI Scientist v2 runtime dependencies are not installed in the local workspace.
- `mlebench` CLI/module is unavailable locally.
- Kaggle CLI is unavailable locally.
- Kaggle credentials exist in 1Password and need to be injected on the execution host.
- `GEMINI_API_KEY` is unavailable locally for the default MLEvolve model configuration.

See:

- `.context/ablation/runtime_readiness.md`
- `.context/ablation/screening_preflight.md`
- `.context/ablation/kaggle_credentials_1password.md`
- `.context/ablation/aws_execution_handoff.md`
- `.context/ablation/mlevolve_smoke.md`
- `.context/ablation/ais_v2_adapter_smoke.md`

## Completed Infrastructure

- Shared schema for run and node logs.
- MLE-bench Lite task manifest.
- Run-matrix launcher and budget policy.
- MLEvolve ablation configs and runtime controls.
- AI Scientist v2 MLE-bench adapter fixtures, schema export helpers, and ablation config matrix.
- Submission audit helper.
- Shared result aggregator.
- Statistical analysis script.
- Audit promotion gate.
- Persistent progress log.
- AWS worker setup guide, worker environment spec, manifest dispatcher, and AWS Batch job-spec renderer.
- Locked T22 cheap-screening matrix and 360 planned screening manifests/job specs.

## Current Evidence

Only fixture evidence is available. Fixture outputs validate the harness but do not support empirical component recommendations.

Available fixture artifacts:

- `.context/ablation/exports/sample/`
- `.context/ablation/analysis/sample/`
- `.context/ablation/audit/sample/`

## Recommendation Table

| Component Class | Current Recommendation | Evidence State | Notes |
| --- | --- | --- | --- |
| Memory | needs more evidence | Runtime controls and configs exist; no live runs | Compare none, child history, global retrieval, success/failure memory, journal memory. |
| Search | needs more evidence | MLEvolve and AI Scientist v2 search variants schedulable; no live runs | Compare linear, greedy, tree, MCTS, MCGS/BFTS controls. |
| Workflow decomposition | needs more evidence | Controls exist; no live runs | Compare single/multi-step ideation, debugging, improvement, ablation, writeup/review. |
| Model routing | needs more evidence | MLEvolve model-profile configs exist; no live runs | Requires provider credentials and cost tracking. |
| Diversity and novelty | needs more evidence | Diversity/novelty controls and metrics exist; no live runs | Analyze entropy, nearest-neighbor novelty, and score tradeoffs. |
| Operators | needs more evidence | Operator configs and export taxonomy exist; no live runs | Compare Draft, Debug, Improve, Evolution, Fusion/Crossover, Aggregation, Ablation, Review. |
| Evaluator hardening | needs more evidence | Audit gate exists; no live runs | High-score audit failures cannot be promoted. |

## Next Required Steps

1. Satisfy runtime readiness on the AWS execution host with `python3 scripts/check_ablation_runtime_readiness.py --environment-label aws-execution-host`.
2. Complete T21.5 using `docs/ablation_aws_worker_setup.md` and `configs/ablations/aws_worker_environment.json`.
3. Replace placeholder AWS queue/job-definition values in `.context/ablation/aws_jobs/screening_jobs.jsonl`.
4. Rerun MLEvolve smoke on AWS.
5. Rerun AI Scientist v2 adapter smoke on AWS.
6. Run T22 cheap screening from `configs/ablations/screening_matrix.json`.
7. Aggregate screening results and apply the audit gate.
8. Select variants for main ablation.
9. Run main and finalist confirmation phases.
10. Replace this scaffold with empirical recommendations and cited artifacts.

## Interpretation Rules

- Failed, invalid, timed-out, and audit-failed runs remain in denominators.
- Invalid submissions receive effective normalized score `0.0` in effective-score and pairwise summaries.
- Raw final scores are comparable only within the same task and metric.
- Cross-task claims must use normalized score, paired win rate, valid-submission rate, audit-pass rate, cost, and confidence intervals.
- No variant can be promoted unless top candidates pass audit and repeated confirmation.
