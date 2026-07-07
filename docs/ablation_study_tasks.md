# MLEvolve / AI Scientist v2 Ablation Task Backlog

This document turns `docs/ablation_study_plan.md` into an actionable task set. It is written so work can be split across multiple agents with minimal coordination overhead.

Execution charter:

- Project goal, constraints, completion standard, error handling, and progress templates live in `docs/ablation_project_execution.md`.
- Persistent task and experiment updates must be written to `.context/ablation/progress.md`.

## Project Objectives

Primary objective:

- Produce defensible, reproducible evidence about which components should be kept, removed, or modified in MLEvolve and AI Scientist v2 when optimizing for measurable AI research and ML engineering performance.

Shared-benchmark objective:

- Compare MLEvolve and an adapted AI Scientist v2 on MLE-bench Lite under matched tasks, seeds, budgets, grader versions, resource policies, logging, and audit rules.

Native-system objective:

- Evaluate system-specific behaviors that do not transfer cleanly to the shared benchmark, especially AI Scientist v2's paper-generation workflow and MLEvolve's graph-search/code-optimization workflow.

Decision objective:

- Produce a final component table with one of: `keep`, `remove`, `modify`, or `needs more evidence` for memory, search, workflow decomposition, model routing, diversity/novelty, operators, and evaluator hardening.

Non-goals:

- Do not claim one system is universally better across all scientific research tasks.
- Do not treat generated paper quality as equivalent to objective ML benchmark performance.
- Do not promote a component on validation/proxy improvement alone if held-out grader score, validity, audit status, or robustness worsens.
- Do not allow unrestricted self-modification or unlogged changes to prompts, policies, code, configs, or evaluators.

## Project-Level Acceptance Criteria

The overall project is complete only when all of the following are true:

1. Shared benchmark comparability
   - MLEvolve and the AI Scientist v2 adapter both run through the same MLE-bench Lite manifest.
   - Each run uses recorded task id, seed, budget, grader version, resource policy, model routing, and config overrides.
   - Any deviations from matched conditions are reported explicitly.

2. Reproducible evidence
   - Every reported result links back to immutable run manifests, prompt/config snapshots, generated code, node logs, grader outputs, and audit results.
   - Default variants are included as repeated anchors.
   - Failed, invalid, timed-out, and audit-failed runs are included in the denominator rather than dropped silently.

3. Component isolation
   - Each major component class has at least one direct ablation against the default or nearest comparable baseline.
   - Search-policy claims are separated from operator-set claims.
   - Memory claims distinguish child history, global retrieval, journal summaries, and archive memory.
   - Diversity claims report both diversity metrics and downstream performance.

4. Statistical sufficiency
   - Screening results are labeled as screening only.
   - Main shared-benchmark claims use paired task/seed comparisons.
   - Final ranking claims use 10 seeds where budget permits, or are explicitly marked lower confidence.
   - Confidence intervals, paired win rates, and validation-test gaps are reported for promoted variants.

5. Evaluator integrity
   - Top candidates pass validity checks and evaluator-hardening audits before promotion.
   - Validation-test overfitting, data leakage, reward-hack risk, runtime violations, and malformed submissions are tracked as first-class outcomes.
   - A high-scoring but audit-failed result cannot be used as evidence for `keep` without a repaired and re-confirmed run.

6. Final decision output
   - `reports/ablation_summary.md` contains a recommendation table for every major component class.
   - Each recommendation cites score, validity, cost, robustness, diversity, and audit evidence.
   - Each recommendation includes caveats for budget, seed count, task coverage, native-system mismatch, and implementation deviations.

## Task-Level Passing Criteria Standard

Each task's passing criteria should prove one of three things:

- Correctness: the implementation does what the task says on fixtures, smoke runs, or controlled examples.
- Comparability: outputs are compatible with the shared schemas, manifests, budgets, and grader/audit rules.
- Decision usefulness: the task produces evidence that can affect a final keep/remove/modify/needs-more-evidence decision.

A task is not considered done if it only produces code or docs but cannot be validated through its stated passing criteria.

## Quick Outline

Recommended workstreams:

1. Shared evaluation harness
   - Owns benchmark setup, run manifests, cost controls, common schemas, result aggregation, and statistics.
2. MLEvolve implementation
   - Owns MLEvolve instrumentation, config variants, search/memory/workflow/novelty ablations, and native MLEvolve smoke runs.
3. AI Scientist v2 adapter
   - Owns the MLE-bench adapter, AI Scientist v2 ablation switches, operator taxonomy, and shared-benchmark compatibility.
4. Evaluator hardening and audit
   - Owns leakage checks, validity checks, validation-test gap logging, reward-hack audits, and top-candidate confirmation.
5. Experiment execution and reporting
   - Owns cheap screening, main ablation runs, native finalist runs, and final report generation.

Recommended sequencing:

1. Complete T01-T05 first; they define schemas, manifests, and the shared benchmark contract.
2. Run T06-T12 and T13-T18 in parallel; these are the MLEvolve and AI Scientist v2 implementation tracks.
3. Run T19-T21 once both systems can produce compatible run outputs.
4. Run T22-T24 for screening, main ablation, and finalist reporting after AWS worker provisioning is complete.

## Parallelization Map

- Agent A, shared harness: T01, T02, T03, T04, T19, T20, T21
- Agent B, MLEvolve: T06, T07, T08, T09, T10, T11, T12
- Agent C, AI Scientist v2: T13, T14, T15, T16, T17, T18
- Agent D, evaluator hardening: T05, T19, T21
- Agent E, execution/reporting: T21.5, T21.6, T22, T23, T24

Agents should write intermediate notes and handoff files under `.context/ablation/` and avoid editing another agent's implementation files unless explicitly coordinated.

Agents must append a readable task update to `.context/ablation/progress.md` whenever a task is completed, blocked, failed, or superseded. Experiment runners must append an experiment update whenever a smoke, screening, main, or finalist experiment finishes.

## Task Backlog

### T01 - Define Shared Ablation Schemas

Owner: Shared harness agent

Dependencies: none

Summary: Define the canonical run, node, operator, metric, and artifact schemas that both MLEvolve and AI Scientist v2 must emit.

Detailed steps:

1. Create a schema document under `docs/` or `.context/ablation/` that defines one row per run and one row per node.
2. Include fields for system, variant, task id, seed, model role, operator, parent id, branch id, stage, prompt path, code path, submission path, metric names, validation score, final grader score, token count, wall time, cost, validity, audit status, and failure mode.
3. Define the shared operator taxonomy: Draft, Debug, Improve, Evolution, Fusion/Crossover, Aggregation, Ablation, Review, Memory, Evaluation.
4. Define accepted values for run status: `success`, `invalid_submission`, `timeout`, `runtime_error`, `grader_error`, `audit_failed`, `cancelled`.
5. Add a version field to every schema so future changes are traceable.

Passing criteria / tests:

- A checked-in schema document exists.
- A sample JSONL or CSV row for MLEvolve and AI Scientist v2 validates against the schema.
- The schema can represent at least one default MLEvolve node and one AI Scientist v2 stage/node without losing system-specific information.

### T02 - Build Benchmark Task Manifest

Owner: Shared harness agent

Dependencies: T01

Summary: Create the canonical MLE-bench Lite task manifest used by both systems.

Detailed steps:

1. Create `configs/ablations/tasks/mle_bench_lite.yaml` or equivalent.
2. Record task id, domain, difficulty if available, metric direction, expected submission filename, resource class, timeout, seed list, and grader command.
3. Add a small smoke subset of 2 to 3 tasks for fast local or low-cost validation.
4. Add fields for train/validation/test split handling and whether the system is allowed to see public leaderboard context.
5. Document which fields are mandatory before a task can be scheduled.

Passing criteria / tests:

- The manifest contains the 22 MLE-bench Lite tasks with no unresolved `TODO` fields in task id, grader command, metric direction, timeout, resource class, or submission contract.
- The smoke subset can be selected mechanically from the manifest.
- A manifest validator fails on missing task id, grader command, metric direction, or timeout.

### T03 - Implement Run Matrix Launcher

Owner: Shared harness agent

Dependencies: T01, T02

Summary: Build a launcher that expands system x task x seed x variant into reproducible run manifests.

Detailed steps:

1. Create `scripts/run_ablation_matrix.py`.
2. Support dry-run mode that prints planned runs without launching agents.
3. Support filters for system, variant, task, seed, phase, and max runs.
4. Write one immutable manifest per run before execution.
5. Add resume behavior that skips completed manifests unless `--force` is supplied.
6. Log environment metadata: git commit, branch, Python version, package lock info if available, model endpoint aliases, and hardware profile.

Passing criteria / tests:

- `python scripts/run_ablation_matrix.py --dry-run --phase smoke` emits expected run count from the smoke manifest.
- Dry-run output includes system, task, seed, variant, budget, and output directory for every planned run.
- Re-running dry-run with the same inputs produces stable manifest ids.

### T04 - Add Cost and Budget Governance

Owner: Shared harness agent

Dependencies: T01, T03

Summary: Add hard limits and early-stop controls so expensive ablations do not run uncontrolled.

Detailed steps:

1. Define per-run and per-variant limits for wall time, token count, API cost, debug attempts, and maximum generated nodes.
2. Add budget fields to run manifests.
3. Add a small budget monitor that can consume node/run logs and mark a run over budget.
4. Define catastrophic early-stop criteria, such as valid-node rate below a threshold after N attempts.
5. Ensure budget exits are recorded as structured statuses rather than silent failures.

Passing criteria / tests:

- A synthetic run log that exceeds token or wall-clock budget is marked `cancelled` or `budget_exceeded`.
- Budget limits appear in every generated run manifest.
- Early-stop reason is preserved in the run-level output.

### T05 - Define Evaluator Hardening Checks

Owner: Evaluator hardening agent

Dependencies: T01, T02

Summary: Specify and implement the first pass of validity, leakage, runtime, and reward-hack checks for generated solutions.

Detailed steps:

1. Define audit checks for valid `submission.csv`, correct columns, correct row count, finite predictions, no private/test labels usage, bounded runtime, and no hidden state/caching tricks.
2. Create a checklist for top-candidate manual review.
3. Add audit result fields to the shared schema.
4. Add smoke fixtures for valid and invalid submissions.
5. Decide which checks block node selection and which are report-only.

Passing criteria / tests:

- Valid fixture passes and invalid fixtures fail for expected reasons.
- Audit results can be joined to node/run tables by run id and node id.
- Top-candidate confirmation requires at least repeated seed or stricter grader status before a variant is promoted.

### T06 - Instrument MLEvolve Node and Run Exports

Owner: MLEvolve agent

Dependencies: T01

Summary: Make MLEvolve emit schema-compatible node and run logs.

Detailed steps:

1. Add export hooks around `engine/agent_search.py`, `engine/search_node.py`, and `agents/result_parse_agent.py`.
2. Capture node id, parent id, branch id, stage/operator, prompt paths, generated code paths, evaluation output, score, validity, and wall time.
3. Capture memory sources used for each node: child history, global retrieval, cold start, archive/top candidates.
4. Save prompts and summaries needed for diversity and novelty analysis.
5. Ensure exports are append-only and robust to crashes.

Passing criteria / tests:

- A short default MLEvolve smoke run emits run and node tables matching the shared schema.
- Crashing a generated script still produces a node record with failure details.
- Node parent-child relationships can reconstruct a search tree or graph.
- Exported artifacts include enough prompt, config, generated-code, and evaluation-output paths to reproduce or audit the reported node.

### T07 - Create MLEvolve Ablation Config Matrix

Owner: MLEvolve agent

Dependencies: T03, T06

Summary: Encode MLEvolve variants as explicit config overrides.

Detailed steps:

1. Create `configs/ablations/mlevolve/`.
2. Add configs for default, no memory, child-history only, global retrieval, linear chain, greedy tree, vanilla MCTS, default MCGS, no fusion/evolution, single-shot, stepwise plus diff, no cold start, diversity ablated, diversity high, novelty lambda variants, all strong, and strong-code-cheap-feedback.
3. Keep each override minimal and inherit from the default config.
4. Add a README table mapping each config to the research question it tests.
5. Add config validation to catch unsupported keys.

Passing criteria / tests:

- Every recommended initial variant has one config file.
- Config validation succeeds for every file.
- A dry-run matrix can select these configs and produce manifests.

### T08 - Implement MLEvolve Search Policy Ablations

Owner: MLEvolve agent

Dependencies: T06, T07

Summary: Add linear, greedy, vanilla MCTS, progressive MCTS, and default MCGS selectable search modes.

Detailed steps:

1. Modify `engine/node_selection.py` or surrounding selector plumbing to accept a search mode.
2. Implement linear chain selection: one active node improved repeatedly.
3. Implement greedy tree selection: select current best valid node without UCT exploration.
4. Implement vanilla MCTS: fixed UCT, no stagnation detection, no top-K switch, no graph fusion/aggregation.
5. Preserve default MCGS behavior under the existing default config.
6. Log the selected node and selection rationale for every expansion.

Passing criteria / tests:

- Unit or smoke tests show different selection modes pick expected nodes on a synthetic tree.
- Default mode produces behavior compatible with current MLEvolve defaults.
- Selection rationale appears in node exports.

### T09 - Implement MLEvolve Memory Ablations

Owner: MLEvolve agent

Dependencies: T06, T07

Summary: Make local child history, global retrieval, success-only memory, failure-only memory, and dissimilar guidance independently controllable.

Detailed steps:

1. Add switches to disable child-memory prompt injection.
2. Add retrieval filters for success-only and failure-only global records.
3. Add optional dissimilar guidance retrieval for diversity.
4. Log every memory item retrieved or injected into a prompt.
5. Confirm no-memory mode removes both child history and global retrieval.

Passing criteria / tests:

- Prompt snapshots show expected memory content for each memory variant.
- No-memory prompts contain no prior experiment summaries.
- Success-only and failure-only variants retrieve only matching records in a controlled fixture.

### T10 - Implement MLEvolve Workflow and Operator Ablations

Owner: MLEvolve agent

Dependencies: T06, T07

Summary: Make stepwise generation, diff mode, code review, leakage checking, cold start, evolution, fusion, and aggregation independently measurable.

Detailed steps:

1. Verify existing config switches work and add missing switches where needed.
2. Add an `aide_operator_set` mode that restricts operators to Draft, Debug, Improve, and summary memory.
3. Add configs for plus-evolution, plus-fusion, and plus-aggregation.
4. Ensure every node logs the operator that created it.
5. Ensure disabled workflow steps are not invoked indirectly.

Passing criteria / tests:

- Smoke runs show operator counts matching enabled steps.
- AIDE operator-set mode never emits Evolution, Fusion/Crossover, or Aggregation nodes.
- Default config still emits the expected operator mix.

### T11 - Add MLEvolve Diversity and Novelty Scoring

Owner: MLEvolve agent

Dependencies: T06, T07

Summary: Add diversity labels, embeddings, and novelty-aware selection/reward options.

Detailed steps:

1. Add plan/code summary export suitable for embedding.
2. Add deterministic strategy-label extraction fallback for model family, architecture, data strategy, feature strategy, training strategy, and ensembling.
3. Add embedding export for plan summaries.
4. Implement novelty score as nearest-neighbor embedding distance among previous plans.
5. Add novelty lambda configs for 0.05, 0.10, 0.20, and 0.40.
6. Support novelty in selection and optionally in reward/backprop.

Passing criteria / tests:

- A synthetic set of repeated plans gets lower novelty than distinct plans.
- Entropy and nearest-neighbor novelty can be computed from exported node tables.
- Novelty lambda 0.0 is equivalent to baseline selection.

### T12 - MLEvolve Smoke Run and Regression Check

Owner: MLEvolve agent

Dependencies: T06-T11

Summary: Run a small MLEvolve smoke matrix before integrating with shared analysis.

Detailed steps:

1. Select 2 to 3 smoke tasks from the shared task manifest.
2. Run default, no memory, linear, vanilla MCTS, no fusion/evolution, and novelty lambda 0.10.
3. Collect run/node exports and generated submissions.
4. Run evaluator audits on top nodes.
5. Record failures and implementation gaps in `.context/ablation/mlevolve_smoke.md`.

Passing criteria / tests:

- At least one valid submission is produced on each smoke task by the default variant.
- All smoke variants emit schema-compatible logs.
- Any failure is categorized as implementation bug, task setup issue, model/API issue, or expected agent failure.

### T13 - Prepare AI Scientist v2 Working Fork or Patch Layer

Owner: AI Scientist v2 agent

Dependencies: T01

Summary: Create a safe place to patch AI Scientist v2 without losing the original external clone.

Detailed steps:

1. Decide whether to patch `.context/external/AI-Scientist-v2` directly, copy it to `.context/external/AI-Scientist-v2-ablation`, or maintain a separate fork.
2. Record upstream commit hash.
3. Add a patch README explaining local changes and how to rebase.
4. Add a minimal smoke command that starts from a fixed idea and skips writeup/review.
5. Confirm the original clone remains available for comparison.

Passing criteria / tests:

- A working AI Scientist v2 patch location exists and is documented.
- Upstream commit hash is recorded.
- A no-op or minimal fixed-idea run can start without modifying the original reference unexpectedly.

### T14 - Build AI Scientist v2 MLE-bench Task Adapter

Owner: AI Scientist v2 agent

Dependencies: T02, T13

Summary: Convert MLE-bench tasks into AI Scientist v2 idea/task inputs with an explicit `submission.csv` output contract.

Detailed steps:

1. Create a converter that maps each task manifest entry to an AI Scientist v2 idea JSON.
2. Include task description, data file summary, metric, runtime budget, submission format, allowed tools, and explicit instruction to write `./submission.csv`.
3. Add templates for implementation-focused, tuning-focused, and ablation-focused prompts.
4. Disable or bypass paper-specific assumptions for shared benchmark mode.
5. Save generated idea JSONs under a deterministic path per task and seed.

Passing criteria / tests:

- Converter produces valid AI Scientist v2 idea JSON for every complete task in the shared MLE-bench Lite manifest.
- Generated idea JSON includes the submission contract, metric direction, runtime budget, and grader/audit expectations.
- AI Scientist v2 can launch from a converted smoke idea without entering writeup/review stages.

### T15 - Integrate MLE-bench Grading Into AI Scientist v2

Owner: AI Scientist v2 agent

Dependencies: T05, T14

Summary: Replace AI Scientist v2 paper-oriented scoring with MLE-bench grader feedback in shared benchmark mode.

Detailed steps:

1. Add shared-benchmark mode that runs generated code and then invokes the MLE-bench grader.
2. Parse validation and final grader metrics into the shared schema.
3. Treat missing or malformed `submission.csv` as invalid, not as an unstructured run failure.
4. Feed grader output back into AI Scientist v2's journal or stage summaries.
5. Add audit blocking rules before a node can be selected as best.

Passing criteria / tests:

- Valid fixture submission returns a parsed score.
- Invalid submission produces structured invalid status.
- AI Scientist v2 node selection can use the grader score in shared-benchmark mode.

### T16 - Instrument AI Scientist v2 Nodes, Stages, and Operators

Owner: AI Scientist v2 agent

Dependencies: T01, T13

Summary: Make AI Scientist v2 emit schema-compatible logs and map stages to the shared operator taxonomy.

Detailed steps:

1. Add logging around `agent_manager.py`, `parallel_agent.py`, `journal.py`, and stage transitions.
2. Map initial implementation to Draft, baseline tuning to Improve, creative research to Evolution/Improve, ablation studies to Ablation, debugging to Debug, and review/writeup to Review where enabled.
3. Capture parent node, selected node, stage, prompt, code, metric, memory summary, and selection rationale.
4. Export journal summaries used by each worker.
5. Preserve enough information to compare memory and stage ablations.

Passing criteria / tests:

- A smoke run produces run and node tables matching the shared schema.
- Stage/operator counts match the enabled AI Scientist v2 workflow.
- Empty-memory mode can be verified from exported prompt or journal fields.
- Exported artifacts include enough prompt, config, generated-code, journal-summary, and grader-output paths to reproduce or audit the reported node.

### T17 - Implement AI Scientist v2 Ablation Switches

Owner: AI Scientist v2 agent

Dependencies: T13, T16

Summary: Add configurable switches for memory, stage workflow, Semantic Scholar, debug behavior, VLM feedback, writeup/review, and node selection.

Detailed steps:

1. Add config switches for journal memory off, Semantic Scholar off, stage 2 off, stage 4 off, plot/VLM feedback off, writeup off, and review off.
2. Add fixed idea-set runner for paired ablations.
3. Add selection policies: default BFTS, linear stage, greedy, debug-off greedy, and optional UCT/MCTS port.
4. Add prompt controls for default diversity, ablated diversity, architecture entropy, and novelty filtering.
5. Ensure config overrides are logged in each run manifest.

Passing criteria / tests:

- Each switch can be toggled from config or CLI.
- Smoke runs demonstrate disabled stages are not invoked.
- Default behavior remains equivalent to upstream AI Scientist v2 where shared-benchmark mode is disabled.

### T18 - AI Scientist v2 Adapter Smoke Run

Owner: AI Scientist v2 agent

Dependencies: T14-T17

Summary: Run a small AI Scientist v2 shared-benchmark smoke matrix.

Detailed steps:

1. Use the same smoke tasks and seeds as MLEvolve.
2. Run default adapter, no memory, draft-debug-improve only, creative research disabled, ablation disabled, and hardened evaluator.
3. Collect run/node exports, generated code, submissions, and grader outputs.
4. Run evaluator audits on top candidates.
5. Record implementation gaps in `.context/ablation/ais_v2_adapter_smoke.md`.

Passing criteria / tests:

- AI Scientist v2 adapter attempts `submission.csv` on every smoke task and the default adapter produces at least one valid, audit-pass submission across the smoke suite.
- All smoke variants emit schema-compatible logs.
- Invalid outputs are categorized structurally rather than disappearing into raw logs.

### T19 - Build Result Aggregator

Owner: Shared harness agent

Dependencies: T06, T16

Summary: Combine MLEvolve and AI Scientist v2 outputs into common node/run tables.

Detailed steps:

1. Create `scripts/export_ablation_results.py`.
2. Read run manifests, node logs, audit logs, grader outputs, and cost logs.
3. Emit `runs.csv`, `nodes.csv`, `operator_stats.csv`, `variant_summary.csv`, and `audit_summary.csv`.
4. Join task metadata from the benchmark manifest.
5. Include data-quality checks for missing ids, duplicated nodes, missing scores, and invalid parent references.

Passing criteria / tests:

- Aggregator can process one MLEvolve smoke run and one AI Scientist v2 smoke run.
- Output tables have stable columns and no duplicate primary keys.
- Bad fixture logs fail with actionable errors.

### T20 - Build Statistical Analysis Notebook or Script

Owner: Shared harness agent

Dependencies: T19

Summary: Implement paired comparisons, confidence intervals, validation-test gap analysis, and diversity analysis.

Detailed steps:

1. Create analysis script or notebook under `reports/` or `scripts/`.
2. Compute medal rate, valid submission rate, normalized score, pairwise win rate, cost per valid node, and cost per score improvement.
3. Compute bootstrap confidence intervals over tasks.
4. Compute mixed-effects or paired regression model if dependencies are available; otherwise implement paired bootstrap and clear limitations.
5. Compute entropy, nearest-neighbor novelty, operator-level success rates, and validation-test gap.
6. Generate plots for performance profiles, cost-quality frontier, operator contribution, and diversity vs performance.

Passing criteria / tests:

- Analysis runs on smoke data without manual edits.
- Metrics match hand-calculated values on a tiny fixture.
- Outputs include both run-level and node-level summaries.
- The analysis output is sufficient to populate at least one draft keep/remove/modify/needs-more-evidence recommendation with cited evidence.

### T21 - Integrate Evaluator Audit With Aggregation

Owner: Evaluator hardening agent

Dependencies: T05, T19

Summary: Ensure audit status affects promotion decisions and final reports.

Detailed steps:

1. Join audit results into node and run summaries.
2. Add report fields for audit failure counts and failure reasons by variant.
3. Mark unaudited top candidates as provisional.
4. Add a promotion gate requiring top candidates to pass audit and repeated confirmation.
5. Create a small markdown audit report template.

Passing criteria / tests:

- A variant with high score but audit failure is not promoted automatically.
- Final summary distinguishes true invalid submissions from audit failures.
- Audit report can be generated from smoke data.

### T21.5 - Prepare AWS Worker Image and Execution Environment

Owner: Infrastructure/execution agent

Dependencies: T01-T21

Summary: Build or verify the AWS execution environment that will actually run smoke, screening, main, and finalist experiments.

Detailed steps:

1. Choose the AWS execution substrate: AWS Batch is the default recommendation; ECS, EC2, or SageMaker are acceptable if they preserve the same manifest/job contract.
2. Build or select worker images for MLEvolve and AI Scientist v2, or one combined image if dependency conflicts are manageable.
3. Install the MLEvolve dependency stack, AI Scientist v2 dependency stack, MLE-bench CLI/module, Kaggle CLI, Docker/runtime dependencies needed by MLE-bench, and GPU/CUDA libraries where required.
4. Copy or mount this repo, the AI Scientist v2 adapter patch, run manifests, and shared schemas into the worker image/environment.
5. Inject secrets on the worker:
   - `KAGGLE_USERNAME`
   - `KAGGLE_KEY`
   - `OPENAI_API_KEY` for both AI Scientist v2 and the default OpenAI-compatible MLEvolve routing
6. Configure durable artifact storage, preferably S3, for manifests, run logs, node logs, generated code, submissions, grader outputs, audit outputs, and analysis exports.
7. Run `python3 scripts/check_ablation_runtime_readiness.py --environment-label aws-execution-host` on the AWS worker and preserve the report.
8. Run a manifest-dispatch dry-run on at least one MLEvolve smoke manifest and one AI Scientist v2 smoke manifest.
9. Prepare the three shared smoke tasks in MLE-bench on the AWS worker or shared data volume.
10. Validate the worker setup artifacts with `python3 scripts/validate_aws_worker_setup.py`.

Passing criteria / tests:

- AWS worker readiness report marks `mlevolve_smoke`, `ais_v2_smoke`, `shared_mle_bench`, `shared_mle_bench_prepare`, and `screening` as ready, or any remaining missing check is documented as intentionally unused by the selected configs.
- Kaggle credentials are injected on the worker without writing resolved values to git, progress logs, or run artifacts.
- `mlebench prepare` succeeds for the three smoke tasks or the prepared data already exists and is readable.
- `scripts/run_ablation_manifest.py --dry-run --manifest <manifest>` succeeds for at least one MLEvolve manifest and one AI Scientist v2 manifest on the worker.
- Worker-generated run manifests and planned artifact paths match the shared launcher outputs.
- `docker/aws-ablation-worker/Dockerfile`, `docker/aws-ablation-worker/entrypoint.sh`, `scripts/build_push_aws_worker_image.sh`, and `configs/ablations/aws_batch_job_definition.template.json` exist and validate.

### T21.6 - Provision AWS Batch Resources and Secret Wiring

Owner: Infrastructure/execution agent

Dependencies: T21.5

Summary: Turn the AWS worker image/environment contract into concrete AWS resources, secrets, and placeholder-free job specs that can run smoke and screening experiments.

Detailed steps:

1. Create or identify the ECR repository for the ablation worker image.
2. Build and push the worker image with `scripts/build_push_aws_worker_image.sh --repository <ecr-repository> --push`.
3. Create or identify AWS Secrets Manager secrets for `KAGGLE_USERNAME`, `KAGGLE_KEY`, and `OPENAI_API_KEY`. Add `MLEVOLVE_CODE_API_KEY` / `MLEVOLVE_FEEDBACK_API_KEY` secrets only if MLEvolve roles intentionally use a different provider credential.
4. Register the AWS Batch job definition from `configs/ablations/aws_batch_job_definition.template.json` using the real image URI, execution role ARN, job role ARN, secret ARNs, log group, artifact root, and dataset root.
5. Create or identify a Batch compute environment and job queue that can satisfy the CPU, memory, and GPU resource classes in `configs/ablations/aws_worker_environment.json`.
6. Grant or verify the submitter and worker IAM permissions listed in `configs/ablations/aws_required_permissions_policy.template.json`.
7. Regenerate smoke and screening job specs with real `--job-queue` and `--job-definition` values.
8. Upload regenerated job specs to S3 and run `scripts/submit_aws_ablation_jobs.py --dry-run` without `--allow-placeholders`.
9. Submit one AWS worker readiness job and preserve the status summary plus log stream pointer.

Passing criteria / tests:

- ECR image URI exists and points to a pushed worker image built from the current repo state.
- Batch job definition is active and contains no placeholder image, role, secret, log, artifact, or dataset values.
- Runtime secrets are injected through AWS Secrets Manager or an approved equivalent without writing resolved values to git, progress logs, manifests, or job specs.
- Smoke and screening job specs contain zero placeholder values.
- `python3 scripts/submit_aws_ablation_jobs.py --jobs-jsonl .context/ablation/aws_jobs/smoke_jobs.jsonl --dry-run --region us-east-1` reports `ready_for_submit: true`.
- At least one AWS worker readiness job succeeds and produces a saved readiness report or log pointer.

### T22 - Execute Cheap Screening Phase

Owner: Execution/reporting agent

Dependencies: T12, T18, T19-T21.6

Summary: Run the reduced-cost screening matrix on the AWS worker environment to eliminate broken or clearly poor variants.

Detailed steps:

1. Confirm T21.6 AWS provisioning passed and T21.5 readiness passed on the AWS worker image/environment.
2. Use the locked reduced screening matrix in `configs/ablations/screening_matrix.json`: 6 representative MLE-bench Lite tasks, 3 seeds, and the initial 20 variants.
3. Generate immutable screening manifests using `scripts/run_ablation_matrix.py --phase screening --matrix configs/ablations/screening_matrix.json`.
4. Render AWS job specs from the manifests and submit them to the selected AWS execution substrate.
5. Run the recommended initial MLEvolve variants on AWS workers.
6. Run AI Scientist v2 adapter variants that are stable after smoke testing on AWS workers.
7. Enforce reduced 2 to 4 hour budgets where appropriate.
8. Sync artifacts from AWS/S3 to the aggregation location.
9. Aggregate results and produce a screening report.
10. Recommend variants to keep, drop, or repair before main ablation.

Passing criteria / tests:

- AWS worker readiness report is attached to the screening report.
- Smoke jobs have passed for at least one shared MLEvolve manifest and one shared AI Scientist v2 manifest on the same AWS worker image/job definition used for screening.
- Submitted screening job specs contain zero placeholder values and reference the approved Batch queue/job definition.
- Screening report includes valid submission rate, score, cost, audit failures, and diversity metrics.
- Every dropped variant has a documented reason.
- Kept variants are within 80% of default performance or show a clear diversity/cost benefit.

### T23 - Execute Main Ablation Phase

Owner: Execution/reporting agent

Dependencies: T22

Summary: Run the main MLE-bench Lite ablation with enough seeds for credible component comparisons.

Detailed steps:

1. Lock the kept variant list after screening.
2. Run 22 MLE-bench Lite tasks with 5 seeds minimum.
3. Use 10 seeds for finalists or any variant ranking claim if budget permits.
4. Keep default system as a repeated anchor.
5. Run aggregation, statistics, audit, and plot generation.
6. Produce an interim main-ablation report.

Passing criteria / tests:

- Main report contains paired comparisons against default for every kept variant.
- Confidence intervals or bootstrap intervals are reported.
- Validation-test gap is reported for each search strategy.
- Any claim of "best" is backed by audit-passed results.
- Any component recommendation not backed by the required seed count or task coverage is marked `needs more evidence` rather than `keep` or `remove`.

### T24 - Run Native Finalist Tracks and Final Report

Owner: Execution/reporting agent

Dependencies: T23

Summary: Run native-system finalist checks and write the final recommendation report.

Detailed steps:

1. Select finalists: default, best memory, best search, best workflow, best model routing, best diversity/novelty, and best combined.
2. Run MLEvolve native checks on MLE-bench full or held-out tasks if budget allows.
3. Run AI Scientist v2 native fixed-idea experiments with writeup/review enabled for finalists.
4. Optionally run MLGym, NanoChat Autoresearch, or SOL-ExecBench diagnostics if they directly answer remaining questions.
5. Prepare `reports/ablation_summary.md`.
6. Include final component table: keep, remove, modify, or needs more evidence.

Passing criteria / tests:

- Final report separates shared MLE-bench conclusions from native AI Scientist paper-generation conclusions.
- Every promoted component has evidence across score, validity, cost, robustness, or diversity.
- Every major caveat is explicit: budget limits, seed count, task coverage, evaluator limitations, and implementation deviations.

## Coordination Rules for Multiple Agents

1. Each agent owns its task files and should publish handoff notes under `.context/ablation/`.
2. Shared schema changes require updating T01 artifacts first and announcing the version bump in `.context/ablation/schema_changes.md`.
3. Agent-specific smoke runs should not block other agents unless they reveal a schema or manifest incompatibility.
4. All run outputs should be append-only and stored under a run id generated by the shared launcher.
5. Final experiment execution should only use configs and code that have passed smoke runs and schema validation.
