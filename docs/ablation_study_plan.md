# MLEvolve / AI Scientist v2 Ablation Study Plan

## Objective

Run a controlled ablation study to understand which agent components improve research output and task performance across MLEvolve and AI Scientist v2.

Primary project objective:

- Produce defensible, reproducible evidence about which agent components should be kept, removed, or modified in MLEvolve and AI Scientist v2 when optimizing for measurable AI research and ML engineering performance.

Secondary project objectives:

- Compare MLEvolve and an adapted AI Scientist v2 on a shared objective benchmark, primarily MLE-bench Lite.
- Preserve native-system evaluations where shared benchmarking would erase important behavior, especially AI Scientist v2's paper-generation workflow and MLEvolve's graph-search/code-optimization workflow.
- Quantify tradeoffs between performance, validity, cost, robustness, diversity, and research-output quality.
- Build a reusable evaluation harness that can support future ablations without rewriting the benchmark, logging, audit, and analysis layers.

Non-goals:

- Do not claim one system is universally better across all scientific research tasks.
- Do not treat generated paper quality as interchangeable with objective ML benchmark performance.
- Do not report a component as beneficial if it only improves validation/proxy metrics while worsening held-out grader performance or audit outcomes.
- Do not allow unrestricted self-modification or unlogged changes to prompts, policies, code, or evaluators.

The study should answer:

1. Which memory mechanisms improve performance, validity, diversity, and cost efficiency?
2. Which search strategy is most effective under the same compute budget?
3. Which workflow decomposition choices matter most?
4. Where do stronger models pay off, and where can weaker models substitute?
5. Does diversity of thought improve search outcomes, and can novelty search improve beyond prompt-only diversity?

The project is complete when the final report can make a supported recommendation for each major component class: memory, search, workflow decomposition, model routing, diversity/novelty, operators, and evaluator hardening.

Minimum evidence required for a recommendation:

- Shared benchmark results from the same task, seed, grader, budget, and resource policy.
- Schema-compatible run and node logs with prompts/configs sufficient to reproduce the run.
- Validity and audit outcomes for top candidates.
- Paired comparisons against the default variant.
- Cost and robustness measurements alongside score.
- Explicit caveats when seed count, task coverage, budget, or native-system mismatch limits the claim.

External references reviewed:

- AI Scientist v2 repository: https://github.com/SakanaAI/AI-Scientist-v2
- AI Scientist v2 paper: https://arxiv.org/abs/2504.08066
- AI Scientist v1: https://arxiv.org/abs/2408.06292
- Ideation diversity paper: https://arxiv.org/abs/2511.15593
- AIDE: https://arxiv.org/abs/2502.13138
- AIRA: https://arxiv.org/abs/2507.02554
- MLE-bench: https://arxiv.org/abs/2410.07095
- MLGym: https://arxiv.org/abs/2502.14499
- AutoMLGen: https://arxiv.org/abs/2510.08511
- AlphaEvolve: https://arxiv.org/abs/2506.13131
- Darwin Godel Machine: https://arxiv.org/abs/2505.22954
- Hyperagents: https://arxiv.org/abs/2603.19461
- Towards end-to-end automation of AI research: https://www.nature.com/search?q=%22Towards%20end-to-end%20automation%20of%20AI%20research%22
- InternAgent 1.5: https://arxiv.org/abs/2602.08990
- OpenEvolve: https://github.com/algorithmicsuperintelligence/openevolve
- Recursive first steps toward automated AI research blog: https://www.recursive.com/articles/first-steps-toward-automated-ai-research
- Recursive released artifacts repo: https://github.com/recursive-org/first-steps-toward-automated-ai-research

Additional reference implications:

- AI Scientist v1 reinforces the need for a native AI Scientist paper-generation track, because its claimed value includes idea generation, experiment execution, paper writing, and automated review rather than only metric optimization.
- AIDE supports the shared MLE-bench adapter direction. It frames ML engineering as code-space optimization with Draft, Debug, Improve, and summarized memory operators, which maps cleanly onto both MLEvolve and an adapted AI Scientist v2 experiment loop.
- AIRA is the most directly relevant ablation precedent. It decomposes research agents into fitness function, selection policy, operator set, operator policy, and termination rule; shows that operator design and search policy must be varied jointly; adds crossover as an explicit operator; and warns that low seed counts can invert rankings.
- MLE-bench remains the best first shared benchmark because it provides objective grading, human/Kaggle baselines, valid submission checks, medal thresholds, and an existing literature of agent comparisons.
- MLGym is better as a second-stage or native research benchmark. It supports more open-ended AI research tasks and flexible artifacts, but it is less directly compatible with MLEvolve's current `submission.csv` workflow.
- AutoMLGen validates the MLEvolve ablation dimensions: domain knowledge base, MCGS, intra-branch history, cross-branch references, aggregation, valid submission rate, medal rate, and beat ratio.
- OpenEvolve suggests adding quality-diversity archive and island/population-style controls, but its strongest fit is algorithmic code optimization rather than Kaggle-style ML engineering.
- Recursive's project validates the benchmark philosophy: use tight feedback loops, measurable objectives, long-horizon search, memory, branch recombination, and increasingly strict validation against reward hacks.

## Systems Reviewed

### MLEvolve

MLEvolve is a Kaggle-style ML engineering agent built around progressive Monte Carlo Graph Search. The main run loop is in `run.py`, and the default ablation switches are already exposed in `config/config.yaml`.

Relevant implementation hooks:

- Search coordinator: `engine/agent_search.py`
- UCT, top-K exploitation, and branch diversity: `engine/node_selection.py`
- Node state, child memory, and stage labels: `engine/search_node.py`
- Stagnation triggers: `engine/conditions.py`
- Reward/backprop and terminal logic: `engine/evaluation.py`
- Top candidates and best solution persistence: `engine/solution_manager.py`
- Memory layer: `agents/memory/global_memory.py`
- Memory save and code summaries: `agents/result_parse_agent.py`
- Draft/improve/debug/evolution/fusion/aggregation agents: `agents/*.py`
- Single-shot, stepwise, and diff codegen: `agents/coder/`
- Planner with retrieval memory: `agents/planner/planner_with_memory.py`
- Cold start model guidance: `engine/coldstart/`

Important built-in component switches:

- `agent.use_global_memory`
- `agent.use_diff_mode`
- `agent.use_stepwise_generation`
- `agent.use_evolution`
- `agent.use_fusion`
- `agent.use_aggregation`
- `agent.search.use_stagnation_detection`
- `coldstart.use_coldstart`
- `agent.search.parallel_search_num`
- `agent.search.num_drafts`, `num_improves`, `debug_prob`, top-K, fusion, and decay params

### AI Scientist v2

AI Scientist v2 is a template-free autonomous research system with ideation, agentic tree-search experimentation, plotting, writeup, citation, and review stages. The cloned reference repo is in `.context/external/AI-Scientist-v2`.

Relevant implementation hooks:

- End-to-end launcher: `.context/external/AI-Scientist-v2/launch_scientist_bfts.py`
- BFTS config: `.context/external/AI-Scientist-v2/bfts_config.yaml`
- Ideation with Semantic Scholar and reflections: `.context/external/AI-Scientist-v2/ai_scientist/perform_ideation_temp_free.py`
- Stage manager: `.context/external/AI-Scientist-v2/ai_scientist/treesearch/agent_manager.py`
- Parallel agent and node selection: `.context/external/AI-Scientist-v2/ai_scientist/treesearch/parallel_agent.py`
- Journal memory and best-node selection: `.context/external/AI-Scientist-v2/ai_scientist/treesearch/journal.py`
- Stage summaries and paper inputs: `.context/external/AI-Scientist-v2/ai_scientist/treesearch/log_summarization.py`

Important component surfaces:

- Ideation generation count and reflection count
- Semantic Scholar novelty/literature search
- Stage workflow: initial implementation, baseline tuning, creative research, ablation studies
- Journal summary memory
- Debug probability and max debug depth
- Best-first node expansion policy
- Multi-seed evaluation and plot/VLM feedback
- Model choices for code, feedback, VLM feedback, writeup, citation, and review

## Benchmark Recommendation

There is no good off-the-shelf benchmark that evaluates AI Scientist v2 and MLEvolve simultaneously in their native modes.

The core mismatch:

- MLEvolve is optimized for Kaggle-style ML engineering: produce runnable code and a graded `submission.csv`.
- AI Scientist v2 is optimized for autonomous research: generate ideas, run experiments, analyze plots, write a paper, and optionally pass review.

Recommended approach:

1. Use **MLE-bench Lite as the shared scaffold benchmark**.
   - This is the best choice if the goal is to understand how agent implementation choices affect objective ML performance.
   - Run MLEvolve natively.
   - Add an AI Scientist v2 adapter that converts each MLE-bench task into an idea JSON, forces generated code to save `submission.csv`, and replaces paper-oriented metrics with the MLE-bench grader.
   - Disable AI Scientist writeup/review for this shared benchmark except in a small auxiliary analysis.
   - Treat AI Scientist v2 stages as operators in the AIRA sense, so the adapter can still measure whether ideation, baseline tuning, creative research, and ablation stages help even when the output is a graded ML artifact.

2. Keep **native benchmark tracks** for components that do not transfer cleanly.
   - MLEvolve native track: MLE-bench Lite/full and AlphaEvolve mathematical optimization tasks.
   - AI Scientist v2 native track: fixed research ideas, end-to-end completion, generated plots, writeup quality, and review score.

3. Treat the shared benchmark as the primary answer to "which implementation performs better," and native benchmarks as the answer to "which system-specific components matter."

If we must choose only one ablation study, prioritize **MLEvolve ablations on MLE-bench Lite**. That study is more immediately actionable and statistically cleaner because MLEvolve already exposes most ablation switches, has objective grading, and is directly aligned with the user's performance question. AI Scientist v2 ablations are more interesting for paper-generation and open-ended research behavior, but they require noisier evaluation and more harness work.

## Alignment With Recursive's RSI Project

The AI Scientist v2 adapter recommendation is inline with Recursive's approach at the benchmark-design level.

Recursive's blog describes an automated research loop that proposes ideas, implements them, runs experiments, validates results, keeps useful context, combines promising branches, and hardens evaluation against reward hacks. Their released repo contains artifacts rather than the full agent: NanoGPT Speedrun solutions and reproducibility logs, NanoChat Autoresearch scripts and per-seed BPB results, and 10 illustrative SOL-ExecBench kernel implementations.

Key alignment points:

- Use objective, tight-feedback benchmarks where many experiments can be run and compared.
- Require generated code to produce measurable artifacts, not only explanations or papers.
- Preserve long-horizon memory and branch recombination as first-class components.
- Validate improvements across multiple seeds or stricter checks before treating them as real.
- Harden evaluators against reward hacking, data leakage, caching exploits, and variance.

Important difference:

- Recursive optimized training scripts and GPU kernels directly; the released repo is not a reusable agent harness.
- AI Scientist v2 natively optimizes a research narrative and paper pipeline, so an MLE-bench adapter is needed to make it behave like a measurable code-optimization researcher.

Implication for this study: keep MLE-bench Lite as the primary shared benchmark, but add RSI-style evaluator hardening and multi-seed confirmation. Do not use Recursive's released repo as the initial shared benchmark; use NanoChat Autoresearch or SOL-ExecBench later as optional diagnostics for fast-feedback training and systems optimization behavior.

## Evaluation Tasks

Use three tiers so the study is runnable before scaling to full cost.

### Tier 1: Shared MLE-bench Lite

Primary benchmark for implementation-level comparison across MLEvolve and an adapted AI Scientist v2.

- 22 tasks
- 3 seeds for smoke screening only, then 5 seeds minimum for main comparisons and 10 seeds for ranking claims when budget permits
- Fixed wall-clock budget per task, ideally 12 hours for parity with reported MLEvolve numbers
- Same task order, same resource pool, same grader version
- Same output contract: runnable Python, valid `submission.csv`, MLE-bench score
- Log both the validation metric used for search and the final private/test grader score so we can measure validation-test overfitting.

### Tier 2: Native System Benchmarks

Native MLEvolve track:

- MLE-bench full set if budget allows
- AlphaEvolve mathematical optimization tasks already listed in the MLEvolve README
- Optional algorithmic-code-search tasks where score is deterministic and fast
- Optional RSI-style fast-feedback tasks: NanoChat Autoresearch for fixed-budget training and SOL-ExecBench-style kernel tasks for systems optimization, after the main MLE-bench adapter is stable

Native AI Scientist v2 track:

- 6 to 10 fixed idea files
- Include at least 3 idea types: clear implementation task, open-ended method task, and high-risk scientific hypothesis
- Reuse the same pregenerated idea JSONs when ablating downstream experimentation
- Use `--skip_writeup` for low-cost experimentation ablations, then rerun finalists with writeup/review enabled
- Optional MLGym-Bench subset for a more research-like native task suite with flexible artifacts and AUP-style aggregate metrics

### Tier 3: End-to-End Research Demonstration

Use only for finalist variants because it is expensive and noisy.

- Run AI Scientist v2 in full paper-generation mode.
- Run MLEvolve as an algorithm/coding module inside a research-task wrapper where possible.
- Compare scientific usefulness by human rubric plus LLM review, not just generated paper acceptance.

## Core Metrics

### Performance

For MLE-bench:

- Medal rate
- Valid submission rate
- Best private/test score
- Normalized score against baseline and known best
- Human-score percentile where available
- Elo-style pairwise ranking across variants
- Validation-test gap between selected validation score and final grader score
- Best-attempt vs best-submission delta, to expose whether search finds strong nodes but selects poorly

For AI Scientist v2:

- Successful end-to-end completion rate
- Best metric selected by the research task
- Number of successful datasets tested
- Multi-seed mean and standard error
- LLM review score for generated paper, when writeup is enabled
- Human review rubric on a sampled subset

### Process Quality

- Valid node rate
- Bug rate and debug recovery rate
- Time to first valid solution
- Time to best solution
- Tokens and dollars per valid node
- Tokens and dollars per unit score improvement
- Number of unique branches, depth, and leaf count
- Percentage of nodes from draft/debug/improve/evolution/fusion/aggregation

### Diversity and Novelty

Adapt the ideation-diversity paper's method:

- Extract model family, specific architecture, data strategy, feature strategy, training strategy, and ensembling/postprocessing strategy from every draft and improvement plan.
- Compute Shannon entropy over initial draft model families and over broader strategy labels.
- Compute pairwise semantic distances between plan embeddings.
- Track nearest-neighbor novelty: `novelty(node) = distance(plan_embedding, nearest_previous_plan_embedding)`.
- Track branch diversity: entropy over strategy labels per branch and branch-level pairwise distance.

## Experiment Matrix

### A. Memory

Goal: separate local context memory, global retrieval memory, and summarized experiment memory.

MLEvolve variants:

1. `none_strict`: remove child memory from prompts and set `agent.use_global_memory=false`.
2. `child_history_only`: keep `fetch_child_memory()` / parent trajectory prompt memory, set `agent.use_global_memory=false`.
3. `global_retrieval`: default global memory with BM25 plus FAISS retrieval.
4. `global_success_only`: retrieve only successful records.
5. `global_failure_only`: retrieve only failed/debug records for avoidance.
6. `dissimilar_guidance`: prefer dissimilar records during draft/improve to encourage unexplored directions.

AI Scientist v2 variants:

1. `none`: pass an empty `memory_summary` into workers.
2. `journal_summary`: default `Journal.generate_summary(include_code=false)`.
3. `journal_summary_with_code`: include code for selected top and failed nodes only.
4. `stage_summary_only`: use stage-level summaries but not raw node summaries.

Primary comparisons:

- Memory improves performance if it increases valid score and reduces repeated failures without reducing diversity too much.
- Memory hurts if it causes mode collapse: lower entropy, lower novelty, more repeated strategy labels.

### B. Search Strategy

Goal: compare linear, tree, MCTS, MCGS, and novelty-aware search under equal node and time budgets.

MLEvolve variants:

1. `linear_chain`: `initial_drafts=1`, `num_drafts=1`, `num_improves=1`, `parallel_search_num=1`, always improve current best.
2. `greedy_tree`: multiple drafts, select current best good node, no UCT exploration, no top-K soft switch.
3. `vanilla_mcts`: UCT only, fixed exploration constant, no stagnation detection, no top-K switch, no fusion/evolution/aggregation.
4. `progressive_mcts`: UCT with decaying exploration and force-backprop, but no graph-level fusion/aggregation.
5. `default_mcgs`: current MLEvolve default.
6. `mcts_plus_novelty`: UCT reward includes novelty bonus.
7. `mcgs_plus_novelty`: default MCGS plus novelty-aware node selection/reward.

AI Scientist v2 variants:

1. `linear_stage`: one draft, improve same node through each stage.
2. `greedy_bfts`: current best-first selection.
3. `debug_off_greedy`: greedy with `debug_prob=0` to isolate debug repair.
4. `mcts_port`: optional port of UCT selection into `ParallelAgent._select_parallel_nodes`.
5. `novelty_bfts`: best-first score includes novelty among ideas/plans.

Suggested novelty formula:

```text
score(node) = normalized_performance(node) + lambda * normalized_novelty(node)
novelty(node) = min_distance(embedding(node.plan), embeddings(previous_plans))
```

Run `lambda` in `{0.0, 0.05, 0.10, 0.20, 0.40}`. Select by validation tasks, then confirm on held-out tasks.

### C. Workflow Decomposition

Goal: test whether splitting cognition into stages improves output, or just adds cost and failure points.

MLEvolve variants:

1. `single_shot`: `use_stepwise_generation=false`, `use_diff_mode=false`.
2. `stepwise_draft_only`: stepwise draft generation, full rewrite improves.
3. `diff_improve_only`: single-shot draft, structured planner plus diff improves.
4. `stepwise_plus_diff`: current default.
5. `no_code_review`: bypass `code_review_agent` to isolate review impact.
6. `no_data_leakage_check`: set `check_data_leakage=false` to measure cost vs validity protection.
7. `no_coldstart`: set `coldstart.use_coldstart=false`.
8. `coldstart_only_draft`: use cold start only in draft prompts, not improve prompts.

AI Scientist v2 variants:

1. `ideation_no_reflection`: `num_reflections=1`.
2. `ideation_default_reflection`: default `num_reflections=5`.
3. `no_semantic_scholar`: disable literature/novelty search.
4. `experiments_only`: fixed pregenerated ideas, `--skip_writeup --skip_review`.
5. `no_stage2_baseline_tuning`: jump from initial implementation to creative research.
6. `no_stage4_ablation`: skip explicit ablation stage.
7. `no_plot_vlm_feedback`: disable plot analysis feedback.
8. `full_pipeline`: default with writeup and review.

### D. Model Strength

Goal: identify where strong models are necessary and where cheaper models are adequate.

Use role-based model routing rather than only one global model.

MLEvolve roles:

- Draft/code generation
- Improve/evolution/fusion code generation
- Planner JSON
- Feedback/result parsing
- Code review
- Debug

AI Scientist v2 roles:

- Ideation
- Experiment code
- Feedback/result parsing
- VLM feedback
- Best-node selection
- Plot aggregation
- Writeup
- Citation
- Review

Variant families:

1. `all_strong`
2. `all_medium`
3. `all_cheap`
4. `strong_code_cheap_feedback`
5. `cheap_code_strong_feedback`
6. `strong_draft_medium_improve`
7. `medium_draft_strong_debug`
8. `strong_writeup_only` for AI Scientist v2

Report Pareto curves: score vs dollars, valid-node rate vs dollars, paper-review score vs dollars.

### E. Diversity of Thought

Goal: test diversity controls inspired by arXiv:2511.15593 and extend them with novelty search.

Prompt-level MLEvolve variants:

1. `diversity_default`: current draft prompt requires novelty vs memory.
2. `diversity_ablated`: remove novelty/diversity instructions from draft/improve/evolution/fusion prompts.
3. `diversity_high`: require each draft to use a distinct model family or strategy label.
4. `adaptive_complexity`: draft prompts cycle through simple, moderate, ambitious, and high-risk strategies.
5. `role_diverse_panel`: generate independent candidate plans from different personas, then implement one.

Search-level variants:

1. `novelty_reward`: add novelty bonus to reward/backprop.
2. `novelty_selection`: use novelty only in node selection.
3. `novelty_draft_allocation`: allocate more initial drafts to underrepresented strategy labels.
4. `quality_diversity_archive`: keep best node per strategy bucket and sample across buckets.

AI Scientist v2 variants:

1. `default_diverse_ideation`: current novelty/reflection prompt.
2. `ablated_diversity_prompt`: remove "new", "creative", "different from previous proposals" constraints.
3. `architecture_entropy_prompt`: require explicitly different model/experiment families.
4. `temperature_sweep`: use temperature as a secondary, not primary, diversity control.
5. `novelty_ideation_filter`: reject ideas too close to previous ideas by embedding distance.

Primary diversity metrics:

- Entropy over first 5 drafts' architecture labels.
- Entropy over all generated plans' strategy labels.
- Mean nearest-neighbor plan distance.
- Fraction of nodes whose nearest-neighbor distance exceeds a threshold.
- Performance conditional on diversity quantile.

### F. Archive and Meta-Search

Goal: incorporate lessons from AlphaEvolve, Darwin Godel Machine, Hyperagents, and InternAgent 1.5.

MLEvolve variants:

1. `archive_default`: current top-candidate and branch tracking.
2. `quality_diversity_archive`: maintain best node per strategy bucket, not just global top score.
3. `stepping_stone_archive`: sample from diverse historical nodes even when they are not current top performers.
4. `self_reflection_policy`: periodically ask a meta-agent to update search/prompt policy, but do not allow arbitrary codebase self-modification.
5. `multi_evaluator`: score nodes with task metric, validity, runtime, novelty, and robustness rather than task metric alone.

AI Scientist v2 variants:

1. `fixed_manager`: current stage manager/search policy.
2. `archive_manager`: keep an archive of diverse successful methods and sample from it during stage transitions.
3. `meta_manager`: allow the manager to revise future stage goals and search parameters based on run history.
4. `transfer_memory`: seed future idea runs with successful meta-level lessons from previous runs.

Do not allow unrestricted self-modification in this ablation. Measure meta-search as explicit, logged policy/config changes so results remain interpretable and safe.

### G. Operator and Evaluator Controls

Goal: incorporate AIDE, AIRA, AutoMLGen, and Recursive's lesson that operator design and evaluator reliability can dominate the apparent search algorithm.

MLEvolve variants:

1. `aide_operator_set`: restrict to Draft, Debug, Improve, and summary memory.
2. `plus_evolution`: add intra-branch evolution on top of the AIDE-style operator set.
3. `plus_fusion`: add cross-branch fusion/crossover.
4. `plus_aggregation`: add multi-branch aggregation.
5. `selection_by_validation_only`: select nodes only by the local validation score.
6. `selection_by_hardened_score`: select nodes by validation score plus validity, runtime, leakage checks, and multi-seed robustness when affordable.

AI Scientist v2 adapter variants:

1. `stage_as_operator_default`: map current stages to operators and keep default stage order.
2. `draft_debug_improve_only`: bypass paper-specific research stages and use AIDE-style coding operators.
3. `creative_research_disabled`: remove creative research stage while keeping implementation and tuning.
4. `ablation_stage_disabled`: remove final ablation stage.
5. `hardened_evaluator`: require valid submission, no obvious leakage, bounded runtime, and repeated-score confirmation before a node can be selected as best.

Report operator-level statistics separately from search-level statistics: valid-node rate, mean improvement per operator, cost per successful node, and downstream selection rate.

## Study Design

### Phase 0: Instrumentation

Add a small analysis layer before running expensive experiments.

1. Save every prompt, plan, stage, parent id, branch id, model role, token count, cost, wall time, metric, validity status, and submission path.
2. Save the operator that produced each node, using a shared taxonomy: Draft, Debug, Improve, Evolution, Fusion/Crossover, Aggregation, Ablation, Review.
3. Add strategy-label extraction for every node. Use a fixed extractor prompt plus deterministic fallback regexes for common model families.
4. Add embedding export for plans and code summaries.
5. Add validation-test gap logging whenever both proxy and final scores exist.
6. Add reward-hack and leakage audit fields for top candidates.
7. Add run manifests that record all config overrides.
8. Add a post-run aggregator that emits one row per node and one row per run.

### Phase 1: Cheap Screening

Purpose: identify obviously bad variants before full runs.

- MLEvolve: 6 representative MLE-bench lite tasks, 3 seeds, 2 to 4 hour budget.
- AI Scientist v2: 3 fixed ideas, experiments only, no writeup/review.
- Run all memory, workflow, and search variants at reduced scale.
- Keep variants that are within 80% of default performance or clearly improve diversity/cost.

### Phase 2: Main Ablation

Purpose: estimate component effects.

- MLEvolve: 22 MLE-bench lite tasks, 5 seeds minimum.
- Use 10 seeds for any final ranking claim if budget permits; AIRA's variance analysis suggests fewer seeds can produce unstable rankings.
- AI Scientist v2: 6 to 10 ideas, 3 seeds minimum.
- Use factorial blocks rather than all combinations.
- Always include the default system as a repeated anchor.

Recommended MLEvolve blocks:

1. Memory block: default search/workflow/model, vary memory.
2. Search block: child-history memory/default model/default workflow, vary search.
3. Workflow block: default search/default model, vary workflow.
4. Model block: default search/workflow/memory, vary model routing.
5. Diversity block: best search/workflow from previous blocks, vary diversity/novelty.

### Phase 3: Confirmation

Purpose: avoid overfitting to screening tasks.

- MLEvolve: full 75-task MLE-bench if budget allows, otherwise held-out 25 tasks.
- AI Scientist v2: fresh idea set with writeup/review enabled.
- Compare only: default, best memory, best search, best workflow, best model-routing, best diversity/novelty, and best combined.

## Statistical Analysis

Use paired comparisons by task and seed.

For MLE-bench:

- Fit mixed-effects model: `score ~ variant + task_difficulty + domain + (1|task) + (1|seed)`.
- Report bootstrap confidence intervals over tasks.
- Use paired win rate vs default: variant beats default on same task/seed.
- Correct for multiple comparisons within each block.
- Report validation-test gap and best-attempt vs selected-submission gap for every search strategy.
- Report performance profiles or AUP-style curves when comparing variants across tasks with different metric scales.

For AI Scientist v2:

- Use completion as binary outcome.
- Use normalized best metric and review score as continuous outcomes.
- Pair variants by idea and seed.
- Include human review on a stratified sample of generated papers.

## Recommended Initial Run Set

Start with the variants most likely to answer the user's questions directly:

1. `default_mcgs`
2. `child_history_only`
3. `global_retrieval`
4. `linear_chain`
5. `vanilla_mcts`
6. `default_mcgs_no_fusion_no_evolution`
7. `single_shot`
8. `stepwise_plus_diff`
9. `no_coldstart`
10. `diversity_ablated`
11. `diversity_high`
12. `mcgs_plus_novelty_lambda_0.10`
13. `all_strong`
14. `strong_code_cheap_feedback`
15. `all_medium`

This gives an interpretable first pass across memory, search, workflow, diversity, novelty, and model strength without exploding the run count.

## Expected Code Changes Before Running

MLEvolve needs only small changes for most ablations because many switches already exist. Add:

1. A config override directory, e.g. `configs/ablations/*.yaml`.
2. A run matrix launcher that expands task x seed x variant and writes manifests.
3. A node export hook for diversity labels and embeddings.
4. A linear/greedy/vanilla-MCTS selection mode in `engine/node_selection.py`.
5. Optional novelty scoring in selection and/or reward.
6. Prompt toggles for diversity instructions and child-memory removal.
7. Operator labels and evaluator-audit fields in node exports.

AI Scientist v2 needs patches in the external repo or a fork:

1. Config switches to disable journal memory, Semantic Scholar, stage 2, stage 4, VLM feedback, writeup, and review.
2. A selectable node policy in `ParallelAgent._select_parallel_nodes`.
3. Diversity/novelty extraction for ideas and generated plans.
4. Fixed idea-set runner for paired ablations.

Shared MLE-bench adapter for AI Scientist v2:

1. Convert each MLE-bench task statement into an AI Scientist idea JSON with explicit "produce submission.csv" requirements.
2. Bypass paper-specific stages for shared benchmarking.
3. Replace AI Scientist metric parsing with the MLE-bench grader result.
4. Preserve AI Scientist node/stage logs so memory/search/workflow ablations remain comparable.
5. Normalize wall-clock, GPU, and retry policies to match MLEvolve.
6. Expose an AIRA-style operator taxonomy over AI Scientist stages.
7. Add hardened evaluator checks before selecting a best node.
8. Log validation-test gaps and best-node selection decisions.

## Plan Improvements To Make Before Execution

1. Add a shared abstraction table that maps each component across systems:
   - memory: child history, retrieval memory, journal summary, archive memory
   - search: linear, greedy, UCT/MCTS, MCGS, BFTS, archive search
   - operators: draft, debug, improve, evolution, fusion, aggregation, ablation
   - verification: execution parse, grader, code review, data leakage, plots/VLM, multi-seed

2. Separate "performance benchmark" from "research-output benchmark."
   - Performance benchmark: MLE-bench Lite shared adapter.
   - Research-output benchmark: AI Scientist v2 native paper pipeline.

3. Make the first run set smaller and more diagnostic.
   - Start with default, no memory, no diversity, linear, vanilla MCTS/BFTS, no stepwise, no fusion/evolution, and novelty search.
   - Add model-strength sweeps only after the core scaffold results are stable.

4. Add cost governance.
   - Hard per-variant token and dollar budgets.
   - Stop variants early if valid-node rate is catastrophically low.
   - Require all variants to log model-role usage so model-routing results are interpretable.

5. Add human review only at the finalist stage.
   - Use LLM review for broad screening.
   - Use human review for final AI Scientist papers and any surprising results.

6. Add safety and reproducibility constraints.
   - Sandbox all LLM-written code.
   - Pin datasets, package versions, model endpoints, grader version, and random seeds.
   - Save every config override and prompt template used for a run.

7. Add evaluator hardening before large-scale runs.
   - Check for data leakage, invalid submissions, hidden state/caching hacks, and runtime overages.
   - Confirm top candidates with repeated seeds or stricter graders before reporting wins.
   - Track evaluator failures as first-class outcomes, not miscellaneous run errors.

## Decision Criteria

Promote a component if it meets at least two of:

- Statistically significant improvement in primary score or paper-review score.
- At least 10% relative improvement in valid submission/completion rate.
- At least 10% lower cost at statistically indistinguishable performance.
- Higher diversity without lower performance.
- Better robustness: lower variance or fewer catastrophic failures.

Reject or keep disabled if:

- It improves validation but lowers private/test score.
- It increases cost without measurable quality gain.
- It lowers diversity and creates repeated failed ideas.
- It improves only one narrow domain and hurts others.

## Main Risks

- Full factorial design is too expensive. Use blocked factorials and staged screening.
- Diversity can be confounded with implementation quality. Prefer prompt controls and novelty search over temperature-only controls.
- AI Scientist v2 paper scores are noisy. Pair by idea/seed and include human review for finalists.
- MLE-bench medal rate is coarse. Always report valid submission rate, normalized score, percentile, and pairwise Elo-style ranking.
- Memory can create leakage-like reuse or mode collapse. Inspect repeated strategy labels and nearest-neighbor distances.

## Deliverables

1. `configs/ablations/` with one YAML per variant.
2. `scripts/run_ablation_matrix.py` to launch variants.
3. `scripts/export_ablation_results.py` to produce node/run tables.
4. `reports/ablation_summary.md` with plots and tables.
5. A final recommendation table: keep, remove, or modify each component.
