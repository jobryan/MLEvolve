# E1 Runbook — fold-swap causal test (arms A/B/C/D)

Operational sequence for E1 (docs/self_learning_autoresearcher_plan.md §4). Split: `experience/mle_bench_lite_split.json` (v4, frozen — do not edit).

## Gates (all must hold before step 1)

- [x] **Anchor frozen and committed** (2026-07-06, from phoenix's reply memo + verified on origin):
  - `ANCHOR = 602291e` on `jobryan/e1-mlevolve-ablation` ("Ablation harness: wired variants, MLE-bench grading, model tiers, exports")
  - Worker patch: `tranche2_worker_patch_20260706_v3.tar.gz`, sha256 `7f01a9110b7fb2b1569ed62290892d221f42fba9b3b59cb4ebc56d2a166a3a3f` (S3 patches prefix, declared frozen ≥2 weeks; v3 required — v2 silently breaks gpt-5.x routing)
  - Worker image digest: `sha256:5c9f3e82ec427b325f5fff731c8b0def46f537528dd57a8a079d16c778625129`; job definition `mlevolve-ai-scientist-v2-ablation-worker:1`
  - Every E2 manifest records all four identifiers.
- [ ] **Grading in the run path**: proven inside phoenix's pipeline (graded medal rows on NOMAD, incl. gpt-5.5); still verify once on OUR queue via the smoke canary below.
- [x] **Base-system confirmation** (2026-07-07 readout, `t2x_interim_readout.md`): **qualified yes for MLEvolve** — on held-out grader scores it beats AIS-v2 wherever it completes (NOMAD RMSLE 0.0620 vs 0.0895; Aerial AUC 0.9996 vs 0.9963, margins larger than AIS variant spreads), but validity at 10-node/50-min CPU budgets was ~44% (0 valid Spooky rows) vs ~97% for AIS. Conditions adopted: (1) E2 runs use the reduced profile `max_nodes=4, wall_time_seconds=1800, max_debug_attempts=1` (now the generator default, budget policy v2); (2) wall-clock timeout is a first-class outcome in all denominators; (3) anchor prompt contracts (no-GPU/offline/TF-IDF-first) kept. Expect text tasks (esp. Spooky) to have low validity in ALL arms — arm-symmetric, and validity rate is a co-primary metric where experience could legitimately help.
- [x] Kaggle access 22/22 (verified 2026-07-02); budget check vs remaining ~$600 envelope.

## Operational notes from phoenix (2026-07-06 reply memo)

- **Queue**: E2 has its own dedicated On-Demand lane, usable immediately: queue `mlevolve-e2-study-queue-serial4`, CE `mlevolve-e2-study-ce-serial4` (config: phoenix `.context/ablation/aws_jobs/e2_study_aws_environment_serial4.json`). Never use the legacy shared queue; schedule any large Spot fleet after phoenix's t2x drain. Account On-Demand quota is 16 vCPUs shared across E1/E2/E3/E4 — avoid concurrent full-capacity runs.
- **1R mass-failure root cause (resolved)**: of 34 FAILED rows, 26 were the stop's own terminations, 7 organic Spooky wall-clock timeouts (DeBERTa-heavy first drafts at 10-node budgets), 1 infra. Not a substrate defect. Mitigations to inherit: keep the anchor's no-GPU/offline prompt contracts; prefer TF-IDF-first text baselines at micro budgets; set Batch attempt timeout ≥ agent wall budget + 900s so overruns can't destroy artifacts.
- **Manifest trap**: `build_ablation_subset.py --phase` defaults to `smoke` and silently rewrites cloned manifests — always pass `--phase` explicitly.
- Known CPU-worker caveat (arm-symmetric): denoising-dirty-documents is torch.hub-bound; expect low validity in all arms. Stays in fold_a per the frozen split; handled by flag/sensitivity analysis.

## Step 0 — Build the E2 worker patch (once, at launch)

E2 worker patch = the frozen v3 tarball with its repo-side code replaced by our merged
branch (which contains the E1 anchor 602291e + the experience layer); phoenix's
`scripts/` and `.context/external/` (mle-bench snapshot) come from the tarball —
they are not in the anchor commit.

```bash
aws s3 cp s3://autoresearch-experiments-058264252788-us-east-1/mlevolve-ai-scientist-v2-ablation/patches/tranche2_worker_patch_20260706_v3.tar.gz /tmp/v3.tar.gz
python -m experience.build_e2_worker_patch --base-tarball /tmp/v3.tar.gz --output /tmp/e2_worker_patch.tar.gz
# refuses to build if the base sha256 != the pinned v3 anchor; prints the E2 patch sha256
aws s3 cp /tmp/e2_worker_patch.tar.gz s3://<bucket>/e2-study/patches/
```

Then render job specs (after manifests exist):

```bash
python -m experience.render_e2_jobs --manifests .context/e2/manifests_arm_a.jsonl \
    --manifest-s3-prefix s3://<bucket>/e2-study/manifests/e2e1-v1 \
    --artifact-root s3://<bucket>/e2-study/artifacts \
    --output .context/e2/jobs_arm_a.jsonl
# per-run manifest JSONs must be uploaded to the same prefix before submission
```

The rendered worker command verifies the patch tarball sha256 before extraction, and
for arms B/C/D downloads the store tarball and re-verifies its snapshot hash on-worker
(a placebo/real mix-up fails the job instead of polluting the arm).

## Step 1 — Arm A (60 runs; doubles as experience corpus)

All 20 non-canary tasks × seeds {1,2,3}, micro budget, experience **off**. Manifests:

```bash
python -m experience.build_e2_manifests --arm A --phase screening \
    --run-id-prefix e2e1-v1 --seeds 1,2,3 \
    --worker-patch-uri <E2_PATCH_URI> --worker-patch-sha256 <E2_PATCH_SHA> \
    --output .context/e2/manifests_arm_a.jsonl
```

Everything else is the frozen anchor (journal always written; if the memory layer
disables on CPU workers, ingestion's journal fallback covers the corpus).

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

Note: on CPU/offline workers the anchor's global-memory layer may silently disable
(CUDA embedding config + no model download) — ingest then automatically mines
episodic records from `logs/journal.json` instead, so the corpus cannot come up
empty as long as journals sync. Verify the `[info] ... mined N episodic records`
line appears for such runs.

## Step 3 — Placebo stores (arm C)

```bash
python -m experience.placebo --source-store stores/e1_fold_a --output-store stores/e1_fold_a_placebo
python -m experience.placebo --source-store stores/e1_fold_b --output-store stores/e1_fold_b_placebo
```

Same sizes/tokens/structure, content deranged; `placebo_meta.json` records source snapshot + seed. Record: `PLACEBO_A=____`, `PLACEBO_B=____`.

Pack every store for S3 (deterministic tarballs; prints the snapshot + tarball hashes
the manifest generator takes as `--store-snapshot-fold-*` / `--store-tarball-sha-fold-*`):

```bash
python -m experience.pack_store --store-dir stores/e1_fold_a --output stores/e1_fold_a.tar.gz
# repeat for fold_b and both placebo stores; upload to s3://<bucket>/e2-study/stores/
```

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

`python -m experience.build_e2_manifests --arm B|C|D ...` emits all of this per task
(opposite-fold store URI + own-fold exclusions computed from the frozen split; stores
are uploaded to S3 as tarballs and downloaded to `.context/e2_store` by the job
command, per `runtime_controls.store_uri`). Validated invariants: fold-swap routing,
leakage exclusion lists, unique run ids, Batch timeout = wall + 900s, anchor block
in every manifest.

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
