# Self-Learning Autoresearcher: Analysis & Research Plan

**Status:** draft for review · 2026-07-02
**Goal:** Build an autoresearcher that accumulates experience from its own past runs and demonstrably performs better because of it — addressing the open-ended, cumulative-discovery goal that The AI Scientist papers leave as future work.

---

## 1. The claim we are trying to prove

"Meaningfully learns from its past work" must be operationalized before we design anything, because it is easy to build a memory feature and hard to prove it matters. We pre-register the claim as a conjunction:

> **C1 (Causal):** The system with access to experience accumulated from its own prior runs outperforms the *identical* system with that experience ablated, on **held-out tasks it has never seen**, with paired statistics.
>
> **C2 (Content-specific):** The gain disappears (or shrinks significantly) when the experience store is replaced by a same-size **placebo** (shuffled/irrelevant records) — i.e., the benefit comes from the content of its past work, not from extra prompt tokens or scaffolding changes.
>
> **C3 (Cumulative):** Performance improves with the *amount* of accumulated experience (dose-response), or across a task sequence with order randomized (learning curve).
>
> **C4 (Behavioral mechanism):** At least one mechanism-level signature moves in the right direction: repeated-error rate drops, time/cost to first valid submission drops, or retrieved experience is demonstrably used in plans that win.

C1 + C2 are mandatory; C3 and C4 make the result convincing rather than merely significant. Failure on C2 is itself a publishable finding (placebo-context effects in agentic systems), and failure everywhere is an honest negative result in the spirit of the ICBINB workshop that accepted AIS-v2's paper.

Three distinct flavors of "better" all count, and we measure all three:
- **Outcome:** higher graded score on new tasks.
- **Efficiency:** same outcome for less time/tokens/dollars.
- **Reliability:** fewer repeated mistakes; smaller validation→test gap.

---

## 2. Background and current state

### 2.1 What AI Scientist v1/v2 actually leave open

- AIS-v1 (arXiv:2408.06292) states the vision directly: *"In principle, this process can be repeated to iteratively develop ideas in an open-ended fashion, acting like the human scientific community."* Nothing in v1 or v2 implements it — every run starts amnesiac.
- AIS-v2 (arXiv:2504.08066) §5 concedes the system does not "consistently reach workshop level," and that *"formulating genuinely novel, high-impact hypotheses"* and *"rigorously justifying design choices with deep domain expertise"* remain out of reach. Both are exactly the capabilities that accumulated experience should feed.
- AIS-v2's evaluation (3 papers submitted, 1 accepted, human-selected best-of-seeds) is not a repeatable grading signal. This matters for us: **you cannot demonstrate learning without a cheap, objective, repeated measure of research quality.** This is the single biggest design constraint on our experiments.

### 2.2 What exists in this repo (MLEvolve)

MLEvolve (fork of AutoMLGen, arXiv:2510.08511) is an MLE agent solving Kaggle-style tasks via Monte-Carlo Graph Search; currently #1 on MLE-bench (65.3% medal rate, 12h budget). Relevant assets:

- **Within-run memory already wired:** `agents/memory/global_memory.py` stores every executed node (plan, code summary, stage, success/fail label, metrics, error logs) in `workspace_dir/global_memory/records.json`, retrieved via hybrid BM25 + FAISS (`use_global_memory: True` in `config/config.yaml`). `agents/planner/planner_with_memory.py` does two-stage planning: dissimilar records to force novelty, similar success/fail records to refine. **This is the seed of the learning layer — but it is scoped to a single task run and reset afterwards.**
- **Search operators** (draft/debug/improve/evolution/fusion/aggregation in `engine/agent_search.py`) with per-node journals (`logs/journal.json`) — rich raw material for post-run reflection.
- **Top-K solution persistence** (`engine/solution_manager.py`, `top_solution/top{1..20}/`) — raw material for a solution library.
- **Multi-model support** (Gemini + OpenAI-compatible: GPT, Claude, Qwen, DeepSeek, Kimi).

### 2.3 What exists in the phoenix workspace (ablation study)

The MLEvolve-vs-AIS-v2 ablation project (branch `jobryan/mlevolve-ablation`, authoritative log `.context/ablation/progress.md`) has a complete, locally tested harness: schemas, 22-task MLE-bench Lite manifest, matrix launcher, budget governance, paired-bootstrap analysis, promotion gate, AWS Batch worker stack (account 058264252788, us-east-1, CPU-only, measured micro-run cost $0.25–1 all-in). ~40 paid micro runs executed; ~$3–8 spent of the $200 cap. A revised Tranche-1 (72 jobs, $250 cap) was launch-ready on 2026-07-01 and — crucially — already includes *within-run memory ablations* (`no_memory`, `child_history_only`, `global_retrieval` for MLEvolve; `no_journal_memory` for AIS-v2).

The 2026-07-01 review (`.context/ablation/plan_review_2026-07-01.md`) found validity problems that this plan **inherits as prerequisites**:
1. ~13/16 AIS-v2 variants are behavioral no-ops — all per-variant AIS results are noise.
2. **No MLE-bench grading in the run path** — every recorded score is the agent's own validation metric.
3. Dead MLEvolve config keys (`diversity_mode`, `novelty_in_reward`, `model_profile`).
4. Default anchor mutated mid-study (prompt hardening + gemini→gpt-4.1); nothing committed to git.
5. Run→grade→audit→export pipeline never exercised on real runs.

We cannot claim "learning improved outcomes" on agent-side validation metrics — an agent with memory of past overfitting tricks could *look* better while being worse. **Real grading is non-negotiable for this project.**

### 2.4 Related work and where the novelty is

| System | Cross-run mechanism | Evidence offered |
|---|---|---|
| Reflexion / ExpeL | within-episode reflection / experience pool over episodes | task success on text games, HotpotQA-style tasks |
| Voyager | skill (code) library, grows over a Minecraft lifetime | tech-tree progress |
| Agent-KB (arXiv:2507.06229) | shared cross-domain experience KB | GAIA/SWE-bench gains vs no-KB |
| A-Mem, memory-agent line | generic agentic memory stores | task-level QA improvements |
| ML-Agent (arXiv:2505.23723) | RL fine-tuning of the LLM on MLE trajectories | weight-space learning, not memory |
| AutoResearchClaw (arXiv:2605.20025) | "cross-run evolution: mistakes → safeguards" | +54.7% vs AIS-v2 on ARC-Bench; mechanism thinly evaluated |
| **This project** | episodic + distilled + procedural self-experience in a top MLE-bench system | **causal ablation + placebo control + dose-response + transfer, on objectively graded held-out tasks, with mechanism analysis** |

Nobody in the autoresearch line has shown the *causal, placebo-controlled, dose-dependent* result. AutoResearchClaw asserts cross-run evolution but does not isolate it. That isolation — plus the self-vs-foreign-experience contrast in §4 (E1 arm D) — is our contribution. A negative or mixed result is still a contribution: it would bound how much current-generation retrieval memory helps a state-of-the-art research agent.

### 2.5 Base-system decision: MLEvolve vs AIS-v2

The user intends to pick "whichever works better at research." Recommendation, with reasoning:

**Use MLEvolve as the substrate for the causal learning claim; use the AIS-v2-style open-ended loop as a final, exploratory extension.**

- The learning claim needs hundreds of graded runs. MLE-bench gives a real leaderboard percentile per run for ~$0.25–4. AIS-v2's endpoint (paper quality) costs human/LLM review per sample and has huge variance — you would drown the effect in noise long before drowning it in budget.
- MLEvolve is the stronger researcher on the only shared measurable ground (MLE-bench #1; the AIS-v2 adapter in phoenix exists precisely because AIS-v2 needed an adapter to even run there). The in-flight Tranche-1, once grading is fixed, either confirms this or surprises us — cheap to let it decide (E0 below).
- MLEvolve already has the memory substrate (§2.2); AIS-v2's fork in phoenix has no functioning ablation switches to build on.
- The AIS-v2 goal ("open-ended discovery building on prior findings") is still honored: Phase 3 ports the learning layer to an ideation→experiment→findings loop and evaluates it with rubric judges — but only after the mechanism is proven where grading is objective.

---

## 3. Approach: the experience layer

One new subsystem, `experience/`, behind a single feature flag, with three mechanisms of increasing abstraction. Each is independently switchable (needed for E4).

### M1 — Episodic memory: persistent cross-run retrieval
Extend `agents/memory/global_memory.py` from per-task to a persistent store:
- **Store:** append each run's node records to a global corpus keyed by `(task_id, run_id, node_id)` with task metadata (domain, modality, metric type, dataset size) and outcome labels. Reuse the existing record schema plus `code_summary` (already added in commit `c03f3c4`).
- **Retrieve:** at draft/improve/debug time, query with (current task description + current plan) using the existing BM25+FAISS hybrid, **gated by task-similarity and filtered by a leakage guard** (hard-exclude records from the same competition — see §5 controls).
- **Inject:** through the existing two-stage planner (`planner_with_memory.py`) — minimal new prompt surface, so the ablation is clean.

### M2 — Semantic memory: post-run reflection into lessons
After each run, a reflection agent reads `logs/journal.json` (full search tree with per-node outcomes) and distills:
- **Lessons:** structured claims — "on small tabular datasets, gradient-boosting drafts beat NN drafts; NN branches wasted 40% of budget" — each with provenance (node ids), confidence, and scope (task-type tags).
- **Bug book:** error-signature → root-cause → fix triples mined from debug chains. This targets the most mechanically measurable behavior change: not repeating past errors (C4).
Lessons are retrieved by task-type tags and injected into the draft/improve prompts; the bug book is injected into the debug agent's prompt when an error signature matches. This is the ExpeL/Reflexion/AutoResearchClaw-safeguard idea, but with provenance and scoping so we can audit retrieval precision.

### M3 — Procedural memory: solution library
Index the already-persisted `top_solution/` artifacts across runs: (task metadata, solution skeleton, achieved percentile). At draft time, the top-matching skeletons for *similar-but-not-identical* tasks are offered as starting points (Voyager-style skills, at the granularity of pipelines rather than functions).

### Deferred (stretch, not required for the claim)
- **M4 — Policy priors:** learn operator-selection priors (which of draft/improve/evolution/fusion pays off, per task type) from past trees — a small bandit over the existing soft-switch in `engine/node_selection.py`.
- **M5 — Idea archive:** the AIS-style archive of past findings feeding ideation. This is Phase 3's open-ended loop, not part of the causal proof.

Design invariants: every injected artifact carries provenance; every injection is logged (so we can count counterfactual use); stores are content-addressable and diffable; a store snapshot is an explicit, pinned input to a run (a run manifest cites the store hash) — this is what makes dose-response and placebo arms possible.

---

## 4. Experiments

Notation: **A** = experience off (anchor). **B** = self-experience on. **C** = placebo store (same size, records shuffled across tasks / sourced from irrelevant domains, same injection prompts). **D** = foreign experience (store built by a *different* base model, or generic human-written MLE tips of matched token count).

Task substrate: MLE-bench Lite (22 tasks; all 22 verified downloadable for account `jonobryan`, API preflight 2026-07-02) in a **fold-swap (cross-over) design** — the 20 non-canary tasks split into two stratified folds of 10 (`experience/mle_bench_lite_split.json`, v4, frozen 2026-07-02). Arm-A runs on each fold double as the experience corpus for evaluating the *other* fold, so **every non-canary task is an eval task (n=20 paired comparisons)** and no separate experience-building runs are needed. Power at n=20: paired test detects d≈0.63 at 80% power; sign test needs 15/20 — versus d≈1.1 and a required 8/8 sweep under the earlier fixed 12/8 split.

Saturation note: MLEvolve's published 80.3% medal rate on this Lite split is at 12h/Gemini/GPU. Our discovery tier runs ~30-min micro budgets on CPU with gpt-4.1, where Tranche-1b showed large headroom (frequent invalid runs, unscored tasks) — the regime where experience should matter most. The claim framing is "**experience substitutes for compute**" (does B@30min approach A@2–4h?), with the 2–4h confirmation tranche answering the "only helps a crippled system" critique and a medium-complexity extension (E5) as ceiling insurance.

Primary metric: **leaderboard percentile from `mlebench grade`** (continuous; medal rate is too coarse at our n and near ceiling on Lite at high budgets). Co-primary at micro budget: valid-submission rate and time/cost to first valid submission (far from ceiling, mechanistically tied to memory). Secondary: any-medal, buggy-node fraction, validation→test gap, error-recurrence rate.

### E0 — Prerequisites + base-system decision (mostly already funded)
1. Fix the phoenix validity items on the critical path: wire `mlebench grade` into the worker (plus post-hoc `grade-sample` backfill over the ~18 existing artifact dirs), finalize run records, freeze and commit the anchor (code + model + prompts), pin the mle-bench commit and worker image digest. Cut or fix the no-op variants; for this project only `no_memory`/`global_retrieval`/`no_journal_memory` matter.
2. Run the already-planned Tranche-1 (72 jobs ≤ $250 cap, per the 2026-07-01 launch-readiness pass). Two outputs feed us: (a) MLEvolve-vs-AIS-v2 on graded tasks → confirms or overturns the base-system choice in §2.5; (b) the within-run memory ablation (`no_memory` vs `global_retrieval`) → establishes whether memory helps *within* a run before we ask whether it helps *across* runs. If within-run memory is worthless, that is a loud early warning for the whole program.
3. Accept Kaggle rules for the Lite competitions. **Done 2026-07-02: all 22/22 verified downloadable via the download-all API preflight (detecting-insults and whale-redux initially appeared closed to new joins; user found the accept path and access was re-verified).**

**Gate G0:** anchor frozen and committed; grading proven end-to-end on ≥5 real runs; base system chosen.

### E1 — Headline causal test (C1, C2), fold-swap design
1. **Arm A / experience corpus (same runs):** arm-A (experience off, full logging) on all 20 tasks × 3 seeds = 60 runs. Ingest fold-a's A-runs → `Store_A`; fold-b's → `Store_B`; snapshots hash-pinned. A task's own records are never in its store by construction, and the leakage guard enforces this again at index build time.
2. **Evaluate:** arm B on fold-b tasks with `Store_A` and fold-a tasks with `Store_B` (60 runs); arm C placebo, same fold structure (60 runs); arm D foreign-experience on a stratified 10-task subset (30 runs). All paired per (task, seed) at micro budget (~30 min).
3. **Analysis:** existing paired-bootstrap tooling with task-level clustering; primary contrast B−A on percentile and valid-submission rate; gate on B−C for content-specificity; B−D asks whether *self*-generated experience beats generic wisdom. Injection logs give a free moderator analysis: does the gain scale with how much *same-domain* experience was actually retrieved? (This partially replaces a dedicated transfer experiment.)

**Pre-registered success:** B−A > 0 with 95% bootstrap CI excluding 0 across the 20 tasks, and B−C > 0 same test; ceiling-screened sensitivity per §5. Report per-task win/loss regardless.

### E2 — Learning curve (C3, sequential form)
Run the system through a fixed stratified 12-task subset of the 20 *sequentially* with online ingestion after each task, in 3 randomized orders × 2 arms (persist vs wipe-after-each-task) × 1 seed = 72 runs. Metric: percentile vs position-in-sequence, order-randomized so task difficulty can't masquerade as learning. Test: positive interaction term (arm × position) / Page trend test. This is the experiment that produces the paper's figure-1 learning curve.

### E3 — Dose-response (C3, dosage form)
Freeze store snapshots at 0/33/66/100% of `Store_A` (by run-arrival order); evaluate on 4 fold-b tasks × 2 seeds × 4 doses = 32 runs. Monotone trend (Spearman/Jonckheere) supports cumulativity; saturation is fine and informative.

### E4 — Mechanism ablation
B with exactly one of M1/M2/M3 enabled × 6 tasks (stratified across both folds, using each task's opposite-fold store) × 2 seeds = 36 runs. Which memory type carries the effect? Also read out C4 signatures here: bug-book arm → error-recurrence rate (fraction of failed nodes whose error signature already existed in the store); M3 arm → time-to-first-valid-submission.

### E5 — Medium-complexity extension (headroom insurance + far transfer); contingent on E1 signal
4–6 MLE-bench *medium* competitions (selected for CPU feasibility and Kaggle-rules availability, frozen before launch), arms A vs B × 3 seeds = 24–36 runs, with stores built from the Lite folds. Medium sits at 64% medal even at 12h/Gemini, so it has headroom at any budget; it is also less pretraining-famous than Lite, testing whether self-experience adds more where the base model's prior is weaker, and it is the far-transfer test (Lite experience → harder unseen competitions). Near-vs-far *within* Lite comes free from E1's injection-log moderator analysis (near-transfer probe pairs are pre-placed across folds: text-norm en/ru, birds/whale).

### E6 — Open-ended research loop (exploratory; the AIS-v2 future-work goal itself)
Port the experience layer to a research-question loop (base system per E0 — expected MLEvolve-as-experiment-engine with an ideation wrapper, or the AIS-v2 adapter if E0 surprises): pick one research theme (e.g., "what regularizers help compositional generalization in small transformers" — deliberately AIS-adjacent), run N=8 sequential mini-studies where each study's findings and failures enter the archive (M5) and feed the next ideation round. Evaluate early-vs-late outputs with blinded position-randomized LLM-judge rubrics (novelty w.r.t. the archive, soundness, informativeness) + human spot-check of 25%. This is explicitly exploratory: n is small, the metric is soft; its job is to show the mechanism transplants to the paper-writing regime, not to carry the causal claim.

### Confirmation tranche
Whatever survives E1–E4 gets a finalist confirmation at realistic budget: A vs best-B, 6 tasks × 5 seeds × 2–4h runs = 60 runs (the phoenix charter's 5-seed minimum; AIRA-style warnings about seed-count rank inversion apply at exactly this step).

---

## 5. Validity controls (the section reviewers will attack first)

- **Leakage:** the fatal confound. Same-competition experience→eval leakage is cheating; hard-excluded twice — by the fold-swap construction (a task's store is built only from the opposite fold) and by the store's same-task guard at index build time, with `excluded_task_ids` set to the task's own fold as a third layer.
- **Ceiling effects:** pre-registered screen — any task where arm A's seed-median percentile exceeds 90 is flagged (aerial-cactus is already a canary for this reason); primary analysis reported with and without flagged tasks. Flagged-task list in the split file (`nomad2018`, `denoising` pending the E0 anchor freeze).
- **Pretraining familiarity as moderator, not confound:** Lite competitions are old and famous; the base model may already know the playbook, which shrinks what self-experience can add on those tasks (it affects both arms equally, so it biases toward null, not toward a false positive). Log competition age/popularity and report effect heterogeneity; E5's less-famous medium tasks probe the other end.
- **Placebo/context-length:** arm C matches token counts and injection prompts; without it, "more relevant-sounding text in context" explains everything.
- **Anchor drift:** the phoenix study already demonstrated the failure mode (mid-study prompt hardening + model swap). Anchor = commit hash + model version + prompt set, pinned in every manifest; any change restarts the affected arm. Model API drift mitigated by pinned model versions and by running paired arms concurrently.
- **Selection effects:** no per-arm cherry-picking; failed/timed-out/invalid runs stay in the denominator (phoenix charter rule, kept).
- **Grader integrity:** scores only from `mlebench grade`; agent-side metrics are diagnostics. Validation→test gap reported per arm — if memory teaches the agent to overfit leaderboards, this is where it shows.
- **Multiple comparisons:** E1's two gates (B−A, B−C) are the only confirmatory tests; everything else is labeled exploratory. Holm correction within confirmatory family.
- **Stats power honesty:** the unit of inference is the task (seeds are within-task replicates). At n=20 fold-swap tasks, the paired test detects d≈0.63 at 80% power; a plausible real effect (~5 percentile points mean, 8–12 SD across tasks after seed-averaging) sits near the detection edge. If E1 shows a small-but-consistent positive that misses the gate, the pre-registered escalation is more seeds (cheap, sharpens per-task deltas) plus the convergent C3 trend tests from E2/E3, which are more powerful for the cumulative claim than any single pairwise gate.

---

## 6. Budget and schedule

Measured micro-run cost $0.25–1 all-in; 2–4h runs ~$1.5–4 (phoenix actuals).

| Item | Runs | Est. cost |
|---|---|---|
| E0 Tranche-1 (already planned/capped) | 72 | ≤$250 (separate cap) |
| E1 fold-swap: A (doubles as corpus) + B + C + D-subset | 60+60+60+30 | $105–210 |
| E2 learning curves | 72 | $36–72 |
| E3 dose-response | 32 | $16–32 |
| E4 mechanism | 36 | $18–36 |
| E5 medium extension (contingent on E1 signal) | 24–36 | $30–80 |
| E6 open-ended pilot | ~8 long runs + judging | $40–80 |
| Confirmation tranche (2–4h, 5 seeds) | 60 | $90–240 |
| **Total (excl. E0)** | ~445 | **~$335–750 worst case** |

Budget approved 2026-07-02 at ~$600. The worst-case top end exceeds it, but the program is gated: E5 and the confirmation tranche launch only on E1 signal, and the confirmation tranche is sized at gate time to the remaining budget (e.g. 2h instead of 4h runs, or 4 tasks instead of 6). Gated expectation is ~$400–550.

Schedule (part-time, sequential where dependent): Week 1 = E0 fixes + Tranche-1 launch + Kaggle rules. Weeks 2–3 = experience layer implementation (M1 is mostly re-scoping existing code; M2/M3 are new agents/indexes) + unit tests + 2 canary runs. Week 3 = E1 experience build; Weeks 3–4 = E1 eval + E2. Week 5 = E3–E5. Week 6 = confirmation + E6 + write-up. Negative-result off-ramp after E1: if B≤A and B≤C decisively, skip E3/E5/confirmation, run E4 as diagnosis, write the negative-result paper.

## 7. Risks

- **Micro runs too short for memory to matter** (memory pays off most when budget allows acting on retrieved strategy). Mitigation: E1 runs one 2–4h canary pair per arm before committing the tranche; confirmation tranche is at realistic budget by design.
- **Stale/wrong lessons hurt** (negative transfer). That's a finding, and E4 localizes it; the bug book is the lowest-risk mechanism and would survive alone.
- **Effect is real but smaller than seed noise.** Paired design + continuous percentile + seed escalation; worst case, report CI as a bound.
- **Kaggle rules acceptance stalls the split** (6/22 today). Do it in week 1; it's clicking buttons.
- **Phoenix fixes balloon.** Scope discipline: only grading, run-record finalization, anchor freeze, and the 3 memory-relevant variants are on our critical path — the other 13 AIS variants are not our problem.
- **AutoResearchClaw or others publish the causal result first.** Our differentiators (placebo, dose-response, self-vs-foreign, MLE-bench-graded) hold even so; speed matters, hence micro-first sequencing.

## 8. Immediate next steps

1. ~~User: budget decision and substrate confirmation.~~ **Done 2026-07-02: budget raised to ~$600; MLEvolve confirmed as proof vehicle.**
2. ~~Accept remaining Kaggle competition rules.~~ **Done 2026-07-02: all 22/22 verified downloadable (API preflight, account `jonobryan`).**
3. Execute E0 critical-path fixes in phoenix (grading loop, anchor freeze/commit). Status 2026-07-02: Tranche 1b complete (final report in `.context/ablation/aws_jobs/tranche1b_final_report_2026-07-02.md`), Tranche 1c repair gate running; base-system decision still open — E0-dependent steps pause until it lands.
4. ~~Freeze the task split.~~ **Done 2026-07-02 (v4): fold-swap design frozen in `experience/mle_bench_lite_split.json` — two stratified folds of 10 + 2 canaries covering all 22 Lite tasks; arm-A runs double as the opposite fold's experience corpus, making all 20 non-canary tasks eval tasks (n=20). Version history in-file (v1 12/8/2 → v2 10/10 → v3 9/9 during a temporary Kaggle-access block → v4 = v2 restored after access verified); no version was ever consumed by a run. Must be committed before E1 and never edited after.**
5. ~~Implement `experience/` M1.~~ **Done 2026-07-02: `experience/` package — persistent JSONL store with same-task leakage guard applied at index build time, snapshot hashing, per-injection provenance log (`logs/experience_injections.jsonl`), offline ingestion CLI (`python -m experience.ingest`), BM25-only fallback when no embedding model is configured; injected into draft/improve Memory sections; config-gated by `agent.experience.enabled` (default off). Tests: `python -m experience.test_store`.**
6. Next: M2 (post-run reflection → lessons + bug book) and M3 (solution library), then E1 experience-building runs once E0's anchor freeze lands.
