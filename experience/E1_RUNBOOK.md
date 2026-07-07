# E1 Runbook — fold-swap causal test (arms A/B/C/D)

Operational sequence for E1 (docs/self_learning_autoresearcher_plan.md §4). Split: `experience/mle_bench_lite_split.json` (v4, frozen — do not edit).

## Gates (all must hold before step 1)

- [x] **Anchor frozen and committed** (2026-07-06, from phoenix's reply memo + verified on origin):
  - `ANCHOR = 602291e` on `jobryan/e1-mlevolve-ablation` ("Ablation harness: wired variants, MLE-bench grading, model tiers, exports")
  - Worker patch: `tranche2_worker_patch_20260706_v3.tar.gz`, sha256 `7f01a9110b7fb2b1569ed62290892d221f42fba9b3b59cb4ebc56d2a166a3a3f` (S3 patches prefix, declared frozen ≥2 weeks; v3 required — v2 silently breaks gpt-5.x routing)
  - Worker image digest: `sha256:5c9f3e82ec427b325f5fff731c8b0def46f537528dd57a8a079d16c778625129`; job definition `mlevolve-ai-scientist-v2-ablation-worker:1`
  - Every E2 manifest records all four identifiers.
- [ ] **Grading in the run path**: proven inside phoenix's pipeline (graded medal rows on NOMAD, incl. gpt-5.5); still verify once on OUR queue via the smoke canary below.
- [ ] **Base-system confirmation**: phoenix's 149-job graded fleet (9 AIS-anchor vs 9 MLE-anchor rows, 3 tasks × 3 seeds) is running under v3; interim readout at phoenix `.context/ablation/aws_jobs/t2x_interim_readout.md` expected 04:00–08:00 PDT 2026-07-07.
- [x] Kaggle access 22/22 (verified 2026-07-02); budget check vs remaining ~$600 envelope.

## Operational notes from phoenix (2026-07-06 reply memo)

- **Queue**: E2 has its own dedicated On-Demand lane, usable immediately: queue `mlevolve-e2-study-queue-serial4`, CE `mlevolve-e2-study-ce-serial4` (config: phoenix `.context/ablation/aws_jobs/e2_study_aws_environment_serial4.json`). Never use the legacy shared queue; schedule any large Spot fleet after phoenix's t2x drain. Account On-Demand quota is 16 vCPUs shared across E1/E2/E3/E4 — avoid concurrent full-capacity runs.
- **1R mass-failure root cause (resolved)**: of 34 FAILED rows, 26 were the stop's own terminations, 7 organic Spooky wall-clock timeouts (DeBERTa-heavy first drafts at 10-node budgets), 1 infra. Not a substrate defect. Mitigations to inherit: keep the anchor's no-GPU/offline prompt contracts; prefer TF-IDF-first text baselines at micro budgets; set Batch attempt timeout ≥ agent wall budget + 900s so overruns can't destroy artifacts.
- **Manifest trap**: `build_ablation_subset.py --phase` defaults to `smoke` and silently rewrites cloned manifests — always pass `--phase` explicitly.
- Known CPU-worker caveat (arm-symmetric): denoising-dirty-documents is torch.hub-bound; expect low validity in all arms. Stays in fold_a per the frozen split; handled by flag/sensitivity analysis.

## Step 1 — Arm A (60 runs; doubles as experience corpus)

All 20 non-canary tasks × seeds {1,2,3}, micro budget, experience **off**:

```
agent.experience.enabled=false
```

Everything else is the frozen anchor. Full logging is default (global memory on ⇒ `workspace/global_memory/records.json`; journal always written).

## Step 2 — Build fold stores (offline, no LLM except reflect)

For each fold F ∈ {a, b}, over every arm-A run dir of F's tasks:

```bash
python -m experience.ingest  <run_dir>... --store-dir stores/e1_fold_<F> \
    --with-bugbook --with-solutions --top-n 3          # per-run: add --task-id/--domain/--metric-name from the task manifest
python -m experience.reflect <run_dir>... --store-dir stores/e1_fold_<F> \
    --model gpt-4.1                                     # ~1 cheap call per run; idempotent
```

Record snapshot hashes (printed by both CLIs): `STORE_A=____`, `STORE_B=____`.
Sanity: records > 0 for ≥ 8/10 tasks per fold; bugbook non-empty; ≥ 1 solution per scored run.

## Step 3 — Placebo stores (arm C)

```bash
python -m experience.placebo --source-store stores/e1_fold_a --output-store stores/e1_fold_a_placebo
python -m experience.placebo --source-store stores/e1_fold_b --output-store stores/e1_fold_b_placebo
```

Same sizes/tokens/structure, content deranged; `placebo_meta.json` records source snapshot + seed. Record: `PLACEBO_A=____`, `PLACEBO_B=____`.

## Step 4 — Canary pair (before the tranche)

One fold_b task (spooky) at micro AND one at 2–4h budget, arms A and B, seed 1: verify store loads on the worker (BM25 fallback ok), `experience_injections.jsonl` non-empty with correct `snapshot_hash` and `excluded_same_task ≥ 0`, grade report present, cost within plan.

## Step 5 — Arms B/C/D (150 runs)

Per-task overrides — a fold_a task uses the OPPOSITE fold's store (`Store_B`) and excludes its OWN fold:

```
# arm B, task t in fold_a, seeds {1,2,3}:
agent.experience.enabled=true
agent.experience.store_dir=stores/e1_fold_b
agent.experience.task_id=<t>
agent.experience.excluded_task_ids=[<all fold_a task ids>]

# arm C: same, store_dir=stores/e1_fold_b_placebo
# arm D (10-task stratified subset): same, store_dir=stores/e1_foreign  (build gated on B-store token counts)
```

Fold_b tasks mirror (store `e1_fold_a`, exclude fold_b). All four mechanism flags stay at default `true` for arm B/C; E4 later toggles `agent.experience.use_{episodic,lessons,bugbook,solutions}` one at a time.

## Step 6 — Analysis

- Grader `final_score` → leaderboard percentile, grader-valid rows only; paired bootstrap with task clustering (phoenix `analyze` tooling); confirmatory gates: B−A > 0 and B−C > 0 (Holm).
- Co-primaries at micro: valid-submission rate, time-to-first-valid.
- Moderator (free from logs): per-task same-domain retrieval fraction from `experience_injections.jsonl` vs per-task B−A delta.
- Ceiling screen: flag tasks with arm-A median percentile > 90; report with/without (plus flagged_tasks from the split file).

## Failure modes to watch

- Empty stores on worker (path or copy issues) → arm B silently equals arm A; the canary's injection-log check exists to catch exactly this.
- Same `snapshot_hash` across arms B and C would mean a wiring bug (placebo not actually different).
- Any run whose manifest anchor ≠ `ANCHOR` is excluded and the tranche investigated.
