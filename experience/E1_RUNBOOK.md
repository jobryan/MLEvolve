# E1 Runbook — fold-swap causal test (arms A/B/C/D)

Operational sequence for E1 (docs/self_learning_autoresearcher_plan.md §4). Split: `experience/mle_bench_lite_split.json` (v4, frozen — do not edit).

## Gates (all must hold before step 1)

- [ ] **Anchor frozen and committed**: one commit hash + model version + prompt set, recorded here: `ANCHOR=____`. The phoenix working tree is still mutating (Tranche 1R fixes are uncommitted); E1 may not start until the anchor commit exists and every E1 manifest pins it.
- [ ] **Grading in the run path**: worker writes `grader/grade_report.json` per run (phoenix `scripts/grade_ablation_submissions.py`, landed 2026-07-02 for Tranche 1R) — verify on the E1 canary before the tranche.
- [ ] **Base-system confirmation**: Tranche 1R results do not overturn MLEvolve as substrate.
- [ ] Kaggle access 22/22 (verified 2026-07-02); AWS Batch queue empty; budget check vs remaining ~$600 envelope.

Known CPU-worker caveat (arm-symmetric, from phoenix): denoising-dirty-documents is torch.hub-bound and was dropped from phoenix CPU tranches; expect low validity on it in all arms — it stays in fold_a per the frozen split, and the ceiling/flag sensitivity analysis handles it.

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
