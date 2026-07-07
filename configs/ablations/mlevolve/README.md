# MLEvolve Ablation Config Matrix

Each file in this directory is a minimal JSON override set for one MLEvolve ablation variant. The files inherit from `config/config.yaml`; the run launcher records the selected file and embeds its overrides into each immutable run manifest.

| Variant | Component | Research question |
| --- | --- | --- |
| `default_mcgs` | Baseline | How does the current MLEvolve configuration perform under the shared harness? |
| `no_memory` | Memory | What is the value of logged experiment memory when child history and global retrieval are removed? |
| `child_history_only` | Memory | Is local parent/child experiment history enough without global retrieval? |
| `global_retrieval` | Memory | Does global retrieval help when local child-history prompt injection is removed? |
| `success_memory_only` | Memory | Are successful prior experiments sufficient guidance, or do failure records prevent repeated mistakes? |
| `failure_memory_only` | Memory | Does explicit failure memory reduce repeated mistakes enough to improve performance or validity? |
| `dissimilar_guidance` | Memory | Does retrieval of dissimilar prior attempts increase useful exploration beyond similar-memory retrieval? |
| `linear_chain` | Search | What is lost when graph search is reduced to a single improvement chain? |
| `greedy_tree` | Search | Does greedy best-node expansion outperform or underperform exploration? |
| `vanilla_mcts` | Search | What does fixed UCT contribute without MCGS top-K switching, stagnation, fusion, or aggregation? |
| `progressive_mcts` | Search | Does progressive exploration decay help before adding MCGS top-K and graph-specific mechanisms? |
| `no_fusion_evolution` | Operators | How much do evolution and cross-branch fusion contribute beyond draft/debug/improve? |
| `aide_operator_set` | Operators | How much of MLEvolve's gain comes from the AIDE-style draft/debug/improve loop versus extra graph operators? |
| `plus_evolution` | Operators | What is the marginal value of intra-branch evolution alone? |
| `plus_fusion` | Operators | What is the marginal value of cross-branch fusion alone? |
| `plus_aggregation` | Operators | What is the marginal value of root-level aggregation alone? |
| `single_shot` | Workflow | What is the baseline value of one-shot generation without iterative improvement? |
| `stepwise_diff` | Workflow | What is the explicit default for stepwise generation plus diff editing? |
| `no_stepwise_generation` | Workflow | Does stepwise generation improve initial solution quality or mainly add latency? |
| `no_diff_mode` | Workflow | Does diff mode improve iteration reliability compared with full rewrites? |
| `no_code_review` | Workflow | Does pre-execution code review improve validity and score enough to justify cost? |
| `no_leakage_check` | Workflow | How much does the data-leakage check affect validity, audit failures, and apparent score? |
| `no_cold_start` | Workflow | Does cold-start task/model guidance improve outcomes relative to learned experiment memory alone? |
| `diversity_ablated` | Diversity | What happens when diversity-specific guidance and novelty reward are disabled? |
| `diversity_high` | Diversity | Does stronger dissimilar guidance increase useful exploration or only cost? |
| `novelty_lambda_005` | Novelty | Does a small novelty bonus improve exploration without hurting score? |
| `novelty_lambda_010` | Novelty | Does the planned default novelty bonus improve downstream score and validity? |
| `novelty_lambda_020` | Novelty | Where does novelty start to trade off against exploitation? |
| `novelty_lambda_040` | Novelty | Does high novelty produce diverse but lower-performing attempts? |
| `all_strong` | Model routing | What is the upper anchor when all major roles use the strong model profile? |
| `strong_code_cheap_feedback` | Model routing | Can strong code generation plus cheaper feedback preserve quality at lower cost? |

Validation:

```bash
python3 scripts/validate_mlevolve_ablation_configs.py
```

The validator checks required fields, registry consistency, override key support, basic value types, and the presence of every recommended initial variant.
