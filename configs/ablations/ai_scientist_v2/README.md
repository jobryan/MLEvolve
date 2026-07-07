# AI Scientist v2 Ablation Config Matrix

Each file in this directory is a minimal JSON control record for the AI Scientist v2 MLE-bench adapter. These configs are consumed by the shared launcher and the adapter command builder under `.context/external/AI-Scientist-v2-ablation/ablation_adapter/`.

| Variant | Component | Research question |
| --- | --- | --- |
| `stage_as_operator_default` | Baseline | How does AI Scientist v2 behave when stages are mapped to shared operators under the MLE-bench adapter? |
| `no_journal_memory` | Memory | How much does journal memory contribute when the task/idea is fixed? |
| `semantic_scholar_off` | External tools | Does disabling Semantic Scholar/literature lookup matter for shared benchmark performance? |
| `stage2_off` | Workflow | Does baseline tuning add value beyond initial implementation? |
| `draft_debug_improve_only` | Workflow | How does an AIDE-like workflow compare to the full AI Scientist v2 stage plan? |
| `creative_research_disabled` | Workflow | Does the creative research stage improve benchmark performance or mainly add variance/cost? |
| `ablation_disabled` | Workflow | Does the stage-4 ablation pass improve final benchmark outputs? |
| `vlm_feedback_off` | Feedback | Does plot/VLM feedback matter for MLE-bench tasks? |
| `writeup_review_off` | Paper workflow | What is the shared-benchmark baseline with paper writeup/review disabled? |
| `linear_stage` | Search | What is lost when BFTS is reduced to a linear stage progression? |
| `greedy_selection` | Search | Does greedy best-node selection outperform default BFTS exploration? |
| `debug_off_greedy` | Search/debug | How much does debug behavior matter under greedy selection? |
| `diversity_ablated` | Diversity | What happens when diversity-specific prompting is removed? |
| `architecture_entropy` | Diversity | Does architecture-entropy prompting improve useful diversity? |
| `novelty_filtering` | Diversity | Does novelty filtering increase performance or only exploration cost? |
| `hardened_evaluator` | Evaluator | How does strict audit gating affect validity and promoted results? |

## Wiring status (updated 2026-07-02)

A config in this directory only changes runtime behavior if the adapter translates its overrides into something the fork reads. Current status:

**Wired (real behavioral variants):**

- `writeup_review_off` and every variant carrying `--skip_writeup --skip_review` `cli_flags` (upstream flags).
- `no_journal_memory` — `adapter.journal_memory=false` → `AIS_JOURNAL_MEMORY_DISABLED=1` → `Journal.generate_summary()` returns empty memory.
- `linear_stage` — `adapter.selection_policy=linear_stage` → `AIS_SELECTION_POLICY` → single-chain selection branch in `ParallelAgent._select_parallel_nodes()`.
- `draft_debug_improve_only` — `adapter.stage3_enabled/stage4_enabled=false` → `AIS_DISABLED_STAGES=3,4` → `StageManager._get_max_iterations()` returns 0 and the ablation micro path stops the run after stage 2.

**NOT wired (behavioral no-ops — do not include in paid comparisons):**

- `semantic_scholar_off`, `vlm_feedback_off` — env vars `AIS_S2_DISABLED` / `AIS_VLM_FEEDBACK_DISABLED` are emitted but nothing in the fork reads them.
- `greedy_selection`, `debug_off_greedy` — `selection_policy` values other than `linear_stage` have no selection branch; `agent.search.*` overrides are not applied to the bfts config.
- `stage2_off`, `creative_research_disabled`, `ablation_disabled` — stage-enable overrides other than stage3/stage4 are not translated.
- `diversity_ablated`, `architecture_entropy`, `novelty_filtering`, `hardened_evaluator` — no consumers.

Tranche-1 (2026-07-02) results for then-unwired variants measured run-to-run noise around `stage_as_operator_default` and were invalidated. Wire a switch (adapter env emission + a fork consumer + a `test_variant_command.py` case) before spending on its variant.

Validation:

```bash
python3 scripts/validate_ais_v2_ablation_configs.py
```
