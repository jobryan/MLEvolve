# MLEvolve / AI Scientist v2 Ablation Summary

Status: **final for screening scope** — all planned execution complete; recommendations below are grader-backed at screening confidence
Date: 2026-07-07
Anchor: commit `602291e` + worker patch `tranche2_worker_patch_20260706_v3.tar.gz` (sha256 `7f01a911…`) + image `sha256:5c9f3e82…`

## Evidence base

All scores are held-out MLE-bench grader results (`mlebench grade_csv`), never agent-side validation metrics. Two matched-budget regimes on CPU-only workers, gpt-4.1 anchor:

| Regime | Source | Graded-valid rows |
|---|---|---|
| 10-node / 50-min ("screening") | Tranche 2x fleet (149 jobs; 68 executed before the $50/day cost-gate stop) | 49 (31 AIS, 18 MLE) |
| 3-node / 30-min ("reduced") | Recovery v1 batches 1–5 (68 rows) | 48 |
| Backfill | Tranche 1 post-hoc grading | 11 (+1 grader-rejected) |

Paired comparisons are same task+seed vs the system default; bootstrap CIs (4,000 resamples) over normalized higher-is-better deltas: `.context/ablation/aws_jobs/final_cross_batch_stats_2026-07-07.json`. Rows cancelled unrun by the cost gate are excluded from validity denominators; wall-clock timeouts are counted as outcomes.

## Headline findings

1. **System comparison is regime-dependent, not absolute.** Where MLEvolve completes, it beats AI Scientist v2 on grader score (NOMAD 0.0620 vs 0.0895 RMSLE; Aerial 0.9996 vs 0.9963 AUC). But at the 10-node budget its validity was ~44% vs ~97% for AIS (0 valid MLE Spooky rows); at the reduced budget MLE validity recovers (NOMAD 10/12). AIS packs smaller, safer experiments; MLEvolve searches deeper and wins when it lands.
2. **AI Scientist v2's paper-pipeline stages do not earn their keep on benchmark tasks: `draft_debug_improve_only` significantly beats the full 4-stage default** (+0.022 mean normalized delta, 95% CI [+0.004, +0.043], n=6 paired cells, 9/9 valid). This is the single clearest component result of the study.
3. **MLEvolve's search machinery justifies itself:** `vanilla_mcts` is significantly worse than default MCGS (CI excludes 0), `linear_chain` nearly so (mean −0.0030, CI [−0.0057, +0.0001]), `greedy_tree` neutral-negative, and removing fusion/evolution trends negative (mean −0.013). Nothing simpler matched MCGS.
4. **Memory matters most under tight budgets:** `no_memory` lost every reduced-regime pairing (−0.025 mean, CI excludes 0) and had the worst 10-node validity (2/8). `global_retrieval` is the best-behaved memory variant (mildly positive both regimes; best validity 5/6). AIS journal memory is neutral within noise.
5. **Strong models don't pay for themselves here:** `all_strong` (gpt-5.5 code+feedback) is neutral-to-negative vs the gpt-4.1 anchor in both regimes despite ~4× token price; `strong_code_cheap_feedback` (gpt-5.5 code + gpt-4.1-mini feedback) is the only routing config with a positive trend (mean +0.0018, CI [−0.0002, +0.0033]).
6. **Diversity prompts flip sign by regime:** removing them looked mildly positive at 10 nodes (n=2) but lost 0–4 at the reduced budget (CI excludes 0). Under short budgets, diversity instructions help; selection-only novelty (λ=0.05) is indistinguishable from default.
7. **Evaluator hardening is vindicated by the study's own record:** one agent-"valid" submission was rejected by the grader; agent validation scores diverged from grader scores repeatedly (e.g., AIS Spooky 0.518 validation vs 0.508 graded; MLE tranche-1 denoising 0.06-class validation vs 0.30 graded). Proxy-only selection would have promoted wrong candidates.

## Recommendation table

| Component class | Recommendation | Basis |
|---|---|---|
| Memory (MLE child/global) | **keep** — prefer `global_retrieval` config; do not strip memory | Finding 4 |
| Memory (AIS journal) | **needs more evidence** — neutral within noise (n=4) | Finding 4 |
| Search (MLE MCGS vs simpler) | **keep** MCGS incl. fusion/evolution; reject linear/greedy/vanilla-MCTS | Finding 3 |
| Search (AIS BFTS vs linear) | **keep** BFTS default; `linear_stage` neutral (n=6) | stats file |
| Workflow decomposition (AIS) | **modify** — adopt `draft_debug_improve_only` for benchmark-style tasks; stages 3–4 add cost, not score | Finding 2 |
| Workflow decomposition (MLE) | **keep** stepwise+diff default; `no_stepwise` neutral on score, acceptable validity — optional simplification at reduced budgets | stats file |
| Model routing | **modify** — default to gpt-4.1 everywhere; offer `strong_code_cheap_feedback` as the only premium option; reject `all_strong` | Finding 5 |
| Diversity / novelty | **keep** diversity prompts (regime-dependent benefit); novelty search **needs more evidence** | Finding 6 |
| Operators (fusion/evolution/aggregation) | **keep** — removal trends negative | Finding 3 |
| Evaluator hardening / grading | **keep and extend** — grader-in-the-loop is mandatory; agent validity is an unreliable proxy | Finding 7 |

## Caveats

- Screening confidence: paired cells per variant are small (n=2–6 per regime); CIs marked significant can still be fragile. Confirmation at ≥5 seeds remains open per the charter for any promotion beyond "screening-supported."
- Regime-limited: CPU-only, 30–50-min budgets, 3 MLE-bench Lite tasks (NOMAD/Spooky/Aerial). GPU or 12-hour parity claims are out of scope.
- The 10-node MLE arm is right-censored by the cost-gate stop (73 rows unrun) and timeouts; the reduced-profile recovery deliberately re-covered those cells at a different budget rather than rerunning the expensive regime.
- Both systems ran with study-wide prompt hardening (device/offline/dtype contracts); results describe the hardened systems.

## Cost

Total study OpenAI spend ≈ **$100 list price** (July-7 UTC bucket $60.5 at the gate stop, of which the 10-node fleet was the bulk; recovery added $38.6 of its $250 allowance). AWS compute fully credit-offset throughout. Original $250 tranche cap respected; recovery budget 15% used.

## Key artifacts

- Cross-batch stats: `.context/ablation/aws_jobs/final_cross_batch_stats_2026-07-07.json`
- Fleet rollup: `t2x_terminal_grade_rollup_2026-07-07.json`; paired: `t2x_paired_comparisons_2026-07-07.json`
- Recovery: `recovery_v1_final_report_2026-07-07.md` + per-batch grade rollups
- E2-facing readout: `t2x_interim_readout.md`; stop reports: `t2x_openai_gate_stop_report_2026-07-07.md`, `tranche1r_seed1_stop_report_2026-07-04.md`
- Execution log: `.context/ablation/progress.md`
