# MLEvolve / AI Scientist v2 Ablation Summary — v2 (Conclusive)

Status: **final** — screening (Tranche 1R/2x/recovery, Jul 6–7) + conclusive follow-up (Tranche 3, Jul 8–13)
Anchors: screening `602291e`+v3 tarball; tranche-3 `077fc2e`→`4d0834e`+v4b tarball (sha256s in progress log)
Evidence: **~700 executed runs, ~490 grader-scored rows** across two studies; all scores from held-out `mlebench` grading; paired vs system anchors (same task+seed), 4k-resample bootstrap CIs. Stats: `.context/ablation/aws_jobs/t3_paired_stats.json`, `final_cross_batch_stats_2026-07-07.json`.

## Answers to the five study questions

**Q1 — Memory.** Keep it, and richer is better on the AIS side: `journal_summary_with_code` significantly beats the default summary (+0.075, CI [+0.003, +0.202], 18/18 valid). The retrieval implant — MLEvolve's memory mechanism transplanted into AIS — performs at parity with journal memory (±0.002, n=6): the *presence* of memory matters more than its mechanism. MLEvolve-side: stripping memory was significantly harmful at screening; tranche-3 reduced-profile deltas are noisy (wide CIs) but validity consistently favors memory-on. **Keep; AIS should adopt include_code.**

**Q2 — Search.** Architecture-specific in both directions, now proven by cross-implant: MLEvolve degrades under simpler search (vanilla MCTS significantly worse at screening), and AIS degrades under MCGS-style search (`mcgs_lite` −0.041, CI [−0.098, −0.004], 1W–5L). Linear policies are neutral-to-worse in both systems. **Keep each system's native search; do not transplant.**

**Q3 — Workflow decomposition (the study's most nuanced result).** AIS's `draft_debug_improve_only` significantly beats the 4-stage pipeline at 10-node benchmark budgets (+0.020, CI [+0.002, +0.040]; 4th independent confirmation) — but at 25-node long budgets the effect reverses to W3–L6 (mean −0.05, ns): **the full pipeline's stages are budget-contention victims, not dead weight.** Native paper-mode (D1) could not be compared: the full writeup pipeline is non-terminating on CPU workers (21/21 runs hung to wall at up to 6 h) — an infrastructure-conditional robustness finding in itself. MLEvolve-side: `no_stepwise_generation` is score-neutral at short budgets but **0/9 valid at long budgets** — stepwise generation is load-bearing for validity at scale. **Recommendation: simplification is a short-budget optimization; keep full decomposition when budget ≥ ~25 nodes; keep MLE stepwise always.**

**Q4 — Model strength.** With node-normalized budgets (latency confound removed, n=10): `all_strong` (gpt-5.5 everywhere) is precisely neutral vs gpt-4.1 (+0.003, CI [−0.002, +0.009]) at ~4× price. `strong_code_cheap_feedback` has the largest positive mean (+0.074) but a CI spanning zero (task-heterogeneous). Cheap-coder arms are catastrophic: `all_cheap` and `cheap_code_strong_feedback` both significantly negative AND validity-collapsed (3/14 and 4/15 valid). **Keep gpt-4.1 default; never route code generation to a cheap model; feedback role tolerates cheap models; premium code routing unproven on tasks of this difficulty.**

**Q5 — Diversity / novelty.** Diversity prompts: keep (removing them lost significantly at reduced budgets in screening). Novelty search: **works** once measured properly — at 6-node trees with semantic embeddings, λ=0.05: +0.044 (CI>0, 7W–1L); λ=0.20: +0.050 (CI>0, 6W–1L); λ=0.10 same magnitude, wider CI. Benefit saturates at small λ (flat dose-response). The earlier null was an artifact of 2-node trees and lexical hashing. **Adopt selection-side novelty (λ≈0.05) at screening-scale budgets and above.**

## Recommendation table (final)

| Component | Recommendation | Confidence |
|---|---|---|
| MLE memory | keep (global retrieval preferred) | confirmed (validity) / score-neutral at tiny budgets |
| AIS journal memory | keep; **adopt include_code** | confirmed |
| Memory mechanism (journal vs retrieval) | either — parity | confirmed (n=6) |
| MLE MCGS + fusion/evolution | keep | confirmed |
| AIS BFTS | keep; reject MCGS transplant | confirmed |
| AIS 4-stage workflow | **conditional**: simplify < ~10 nodes; keep ≥ 25 nodes | confirmed both regimes |
| MLE stepwise generation | keep (load-bearing at scale) | confirmed |
| Model routing | gpt-4.1 all roles; cheap feedback OK; cheap code forbidden | confirmed |
| Novelty search (selection-side, semantic) | **adopt, λ≈0.05** | confirmed at ≥6 nodes |
| Diversity prompts | keep | confirmed (screening) |
| Evaluator hardening / grader-in-loop | keep — mandatory | confirmed throughout |

## Caveats

CPU-only workers, ≤25-node budgets, 6 MLE-bench Lite tasks; native paper-quality comparison deferred (pipeline non-termination, suspect LaTeX/plot-agg hang — one instrumented canary would localize it). Cheap-arm and SCCF estimates rest on fewer valid rows due to their own validity collapse (which is itself the finding). Spot instance-family speed variance noted as covariate; Batch timeouts widened mid-tranche (agent budgets constant).

## Cost

Screening ≈ $100; Tranche 3 ≈ $540–575 attributed conservatively (shared-org buckets include sibling studies) of the $750 token cap; AWS credit-offset throughout (< $500 cap by a wide margin). Canary discipline caught 5 fleet-killing defects pre-spend across the program.
