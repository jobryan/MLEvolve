# MLEvolve / AI Scientist v2 Ablation Execution Charter

## Goal Statement

Run the MLEvolve / AI Scientist v2 ablation project to completion by implementing the shared evaluation harness, MLEvolve ablations, AI Scientist v2 MLE-bench adapter, evaluator hardening, screening/main/finalist experiments, and final recommendation report.

The project is complete only when `reports/ablation_summary.md` contains defensible recommendations for each major component class:

- memory
- search
- workflow decomposition
- model routing
- diversity and novelty
- operators
- evaluator hardening

Each recommendation must be one of:

- `keep`
- `remove`
- `modify`
- `needs more evidence`

## Completion Standard

A recommendation is defensible only if it is backed by:

- matched shared-benchmark runs where applicable
- immutable run manifests
- prompt/config snapshots
- generated code artifacts
- node and run logs
- grader outputs
- validity and audit outcomes
- paired comparisons against the default or nearest comparable baseline
- cost, robustness, and diversity measurements where relevant
- explicit caveats for budget, seed count, task coverage, implementation deviations, and native-system mismatch

## Operating Constraints

### Comparability

- Use the same task manifest, seed policy, grader version, resource policy, and wall-clock budget for shared MLE-bench Lite comparisons.
- Any deviation from matched conditions must be logged and surfaced in the final report.
- Screening results must be labeled as screening only and must not be used as final ranking claims.

### Reproducibility

- Every run must have an immutable manifest before execution.
- Every run must record git commit, branch, config overrides, model routing, package/runtime metadata where available, task id, seed, budget, and output paths.
- Prompt templates, generated prompts, generated code, grader outputs, and audit outputs must be preserved.
- Failed, invalid, timed-out, and audit-failed runs stay in the denominator.

### Execution Environment

- Runtime dependency checks apply to the machines that actually execute experiments, not necessarily the local Conductor coordination workspace.
- If experiments run on AWS, the AWS image/instance must have the MLEvolve stack, AI Scientist v2 stack, MLE-bench, Kaggle tooling, model credentials, and benchmark data access.
- The local workspace only needs those dependencies when running local smoke tests or local experiments.
- Before starting T22 screening, T21.6 AWS provisioning must produce placeholder-free smoke and screening job specs using a real Batch queue/job definition.
- Before starting T22 screening, run `python3 scripts/check_ablation_runtime_readiness.py --environment-label <execution-host>` on the execution host and preserve the report.
- Kaggle credentials may be injected from 1Password or another secret manager, but resolved values must not be committed or written into progress reports.

### Evaluator Integrity

- Top candidates must pass validity and evaluator-hardening checks before promotion.
- Validation/proxy-only wins are not sufficient for `keep`.
- A high-scoring run that fails audit, leakage checks, runtime checks, or held-out grader confirmation must be marked failed or provisional.
- Reward-hack, data-leakage, hidden-state/caching, malformed-submission, timeout, and grader-error cases must be reported as first-class outcomes.

### Component Isolation

- Search-policy claims must be separated from operator-set claims.
- Memory claims must distinguish child history, global retrieval, journal summaries, and archive memory.
- Diversity/novelty claims must report both diversity metrics and downstream performance.
- AI Scientist v2 native paper-generation claims must be reported separately from shared MLE-bench performance claims.

### Statistical Discipline

- Main shared-benchmark claims require paired task/seed comparisons.
- Use at least 5 seeds for main comparisons.
- Use 10 seeds for final ranking claims where budget permits; otherwise mark ranking confidence as limited.
- Report confidence intervals or paired bootstraps, paired win rates, valid submission rate, validation-test gap, and audit-pass rate for promoted variants.

### Error Handling

- Recoverable errors should be fixed or categorized, then the run can continue.
- Irrecoverable errors must be logged with exact command, task, variant, seed, error message, and impact on project objectives.
- If a task cannot meet passing criteria, mark it blocked with the smallest clear reason and the next action needed.
- Do not silently skip failed experiments or remove them from summaries.

## Progress Reporting Requirements

Progress must be written in two places:

1. Conversation updates while actively working.
2. `.context/ablation/progress.md` as the persistent handoff log.

Every completed task must get a readable write-up with:

- task id and name
- status: completed, blocked, failed, or superseded
- what changed
- evidence produced
- passing criteria result
- implications for project objectives
- follow-up tasks or risks

Every completed experiment must get a readable write-up with:

- experiment id
- system
- variant
- task id
- seed or seed set
- budget and actual runtime
- result status
- primary score and metric direction
- valid submission status
- audit status
- cost/tokens if available
- immediate interpretation
- implication for component recommendation

## Progress Write-Up Templates

### Task Completion Template

```markdown
### Task Update: <task id> - <task name>

Status: <completed | blocked | failed | superseded>
Date:
Owner:

What changed:
- ...

Evidence produced:
- ...

Passing criteria:
- <criterion>: <pass/fail/partial>

Implication for project objectives:
- ...

Follow-ups / risks:
- ...
```

### Experiment Completion Template

```markdown
### Experiment Update: <experiment id>

Status: <completed | blocked | failed | invalid | audit-failed>
Date:
System:
Variant:
Task:
Seed(s):
Budget:
Actual runtime:

Result:
- Primary metric:
- Valid submission:
- Audit:
- Cost/tokens:

Interpretation:
- ...

Implication for component recommendation:
- ...

Artifacts:
- Manifest:
- Node logs:
- Generated code:
- Submission:
- Grader output:
- Audit output:
```

## Stop Conditions

Stop and ask for direction only when:

- credentials, data access, or required external compute are unavailable
- benchmark licensing or data terms block execution
- multiple agents produce incompatible schema changes that cannot be reconciled locally
- a repeated infrastructure failure prevents meaningful progress
- the user changes the project objective

Otherwise, continue through the task backlog, record progress, and surface errors with their impact.
