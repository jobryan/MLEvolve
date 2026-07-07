#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prompt templates for code review in search pipeline.
"""

from typing import Dict, Any
from utils.response import wrap_code

# ============================================================================
# Code Review Prompts
# ============================================================================
def get_code_review_prompt(task_desc: str, code: str) -> Dict[str, Any]:
    """Build full code review prompt dict from task description and code."""
    introduction = (
        "You are a Senior Data Science Code Reviewer. Your goal is to ensure the submission is legally valid and logically sound.\n\n"
        "⚠️ **CRITICAL INSTRUCTION**:\n"
        "You must strictly follow the [Code Review Guidelines] provided below.\n"
        "Do NOT rely on your general knowledge if it conflicts with the Environment Facts listed in the guidelines.\n"
        "Your output must be a structured review focusing ONLY on Data Leakage and Critical Integrity."
        "**STRICTLY FORBIDDEN**: Do NOT replace the user's model architecture with other backbones (e.g., ResNet, VGG) just to make code executable. Do not question or change the user's model choice.\n"
    )
    prompt = {
        "Introduction": introduction,
        "Task description": task_desc,
        "Code to review": wrap_code(code),
        "Instructions": {},
    }
    prompt["Instructions"]["Code review guidelines"] = get_code_review_guidelines()
    prompt["Instructions"]["Response format"] = get_code_review_response_format()
    return prompt

def get_code_review_guidelines() -> list:
    """Code review guidelines."""
    guidelines = [
        "# 📜 Code Review Guidelines\n",
        "",
        "## ✅ Environment Facts (TRUTH - Do NOT Flag)\n",
        "**Trust these facts absolutely. Overwrite your internal knowledge cutoff:**",
        "  • **Paths**: `./input/`, `./working/`, `./submission/` ALL EXIST. **Don't question the path.**",
        "  • **Submission File Location**: Must save the submission to `./submission/submission.csv`.",
        "  • **Offline/Local Environment**: Prefer packages and checkpoints already available in the worker. Do not assume arbitrary external downloads, dynamic `pip install`, or remote pretrained models will work.",
        "  • **Model Availability**: Named models/checkpoints are available only if the code uses a local path or a package/model that is already present in the environment.",
        "  • **Targeted Model Substitution**: If code cannot run because it depends on an unavailable remote model, checkpoint, or heavyweight backbone, replace only that unavailable dependency with a CPU-safe local baseline while preserving the task, metric, and submission format.\n",
        "  • **Unknown Models Need Evidence**: If a model/checkpoint name is unknown and no local path is provided, treat it as unavailable in the offline worker.",
        "  • Execution time: 9 hours available\n\n",
        "---\n",
               "## 🚫 STRICTLY FORBIDDEN (Zero Tolerance)\n",
        "**You will be penalized if you violate these:**",
        "  • **NO Unnecessary Model Downgrades**: Do not change a working local model/backbone/checkpoint merely because a simpler option exists.",
        "  • **NO Compatibility Speculation**: Do not flag issues based on vague concerns. Flag concrete unavailable resources, removed API keywords, data leakage, invalid metric logic, or submission-format failures.",
        "  • **Model Variables Are Editable Only For Concrete Runtime Risk**: Variables defining `model_name`, `backbone`, or `checkpoint` may be changed only when they reference unavailable external resources or cause a concrete runtime failure.",
        "  • **Preserve the Solution Intent**: Only substitute the minimum needed model/dependency to make the code valid under the local worker constraints. Do not rewrite a correct solution for style or ambition.",
        "  **Don't question the path.**",
        "",
        "---\n",
        "## 🔴 P0 - Data Leakage (HIGHEST PRIORITY)\n",
        "",
        "### P0.1 Data Leakage - Process Order 🚨\n",
        "",
        "**Check if preprocessing is done BEFORE split** (validation data leaks into training):",
        "",
        "❌ **MUST FIX**:",
        "  • Scaler/PCA fitted on full data then split",
        "  • Feature engineering (Target Encoding, etc.) using full data",
        "  • Upsampling (SMOTE) applied before split",
        "",
        "✅ **Correct**: Split first → fit on train only → transform separately",
        "",
        "### P0.2 Data Leakage - Split Strategy 🚨\n",
        "**Core Logic: Check for I.I.D. Violation**",
        "❌ **Flag ONLY IF**: The chosen split method mathematically violates the data's dependency structure.",
        "",
        "## 🟡 P1 - Critical Correctness\n",
        "",
        "### P1.1 Metric & Logic Correctness",
        "  • Task requires F1 but code uses accuracy?",
        "  • Task requires RMSE but code uses MSE?",
        "",
        "### P1.2 Inference Integrity",
        "  • Test predictions: np.zeros(), np.ones(), train_mean(), np.random()?",
        "  • Val predictions: not from actual model.predict()?",
        "",
        "### P1.3 Best Model Usage",
        "  • Code uses best checkpoint (not last epoch) for test predictions?",
        "",
        "### P1.4 API Compatibility",
        "**Common API Issues to Fix:**",
        "  • LightGBM: Use `callbacks=[lgb.early_stopping(...)]` not `early_stopping_rounds=...` in fit()",
        "  • LightGBM native `lgb.train`: use callbacks such as `lgb.early_stopping(...)`, `lgb.log_evaluation(...)`, and `lgb.record_evaluation(evals_result)`; do not pass `early_stopping_rounds=`, `evals_result=`, or `verbose_eval=` directly.",
        "  • XGBoost: Use `XGBClassifier(early_stopping_rounds=...)` (correct) not `fit(early_stopping_rounds=...)`",
        "  • XGBoost sklearn API: put `eval_metric` and `early_stopping_rounds` in the `XGBClassifier`/`XGBRegressor` constructor, not in `.fit(...)`. With `MultiOutputRegressor`, avoid early-stopping fit kwargs unless fitting one target-specific model at a time.",
        "  • PyTorch `ReduceLROnPlateau`: do not pass `verbose=`.",
        "  • scikit-image rank filters: use `footprint=...`, not the removed `selem=` keyword.",
        "  • NumPy arrays with negative strides from flips/rotations/slicing must be copied or made contiguous before converting to torch tensors.",
        "  • AdamW: Use `from torch.optim import AdamW` (not from transformers)",
        "  • NO tqdm, NO verbose=1 in training",
        "",
        "### P1.5 Tabular dtype safety",
        "  • Flag raw object/string/category columns sent into StandardScaler, median/mean imputation, torch float tensors, or numeric model matrices before encoding.",
        "  • Flag numeric feature lists inferred only from column names or numeric-looking ranges when they include object/string columns.",
        "  • Low-cardinality categoricals may be one-hot encoded, but high-cardinality strings should use bounded encodings such as ordinal/factorize, frequency/count, hashing, or fixed-position character features.",
        "  • In a short-budget first tabular draft, flag custom TabNet-like/attention/embedding neural networks if the code does not explicitly bound categorical indices, embedding dimensions, and concatenated tensor widths. Prefer a simple validated baseline before complex neural tabular architectures.",
        "",
        "### P1.6 Cross-domain short-budget reliability",
        "  • For image tasks, flag `torch.hub.load`, remote checkpoint downloads, or named pretrained backbones when no exact local checkpoint path is present. Under short budgets, prefer small local CNNs or sklearn image-feature baselines that can finish and export a submission.",
        "  • For `aerial-cactus-identification`, flag DINO/SigLIP/timm/transformer/pretrained-backbone usage in the first draft unless a local checkpoint is explicitly present. A flattened-pixel sklearn model or tiny local CNN is sufficient for the first valid baseline.",
        "  • For binary PyTorch classifiers, flag `squeeze(1)` on model outputs unless dimension 1 is guaranteed. Prefer `logits = model(x).view(-1)` with `targets = y.float().view(-1)` for BCE-style losses.",
        "  • For text tasks, flag dependencies on NLTK resources such as `punkt`/`punkt_tab`, spaCy model downloads, or heavyweight transformer fine-tuning before a valid baseline exists. Prefer TF-IDF/character n-grams with sklearn classifiers for the first CPU-safe baseline.",
        "  • For custom sklearn transformers used in Pipeline/FeatureUnion, flag classes that do not inherit `BaseEstimator`/`TransformerMixin` or otherwise implement `fit`, `transform`, and `fit_transform` as required by the surrounding sklearn API. Prefer built-in vectorizers or `FunctionTransformer` under short budgets.",
        "  • For materials/scientific tabular tasks, flag geometry parsers that assume the first `geometry.xyz` line is always an integer atom count. Robust parsers should skip comment/header lines and parse only `element x y z` rows; CSV-only baselines are acceptable first attempts.",
        "  • For `nomad2018-predict-transparent-conductors`, flag code that requires geometry parsing before a valid baseline exists or accesses `row['lv1']`, `row['lv2']`, `row['lv3']`, `row['alpha']`, `row['beta']`, or `row['gamma']` without first confirming those columns exist. A numeric `train.csv`/`test.csv` baseline with one regressor per target is acceptable under short budgets.",
        "  • For NOMAD, flag single-output regression code that flattens or drops one of the two target columns. A valid solution must preserve both targets via a multi-output regressor, one model per target, or an estimator that explicitly accepts 2D targets.",
        "  • For denoising/image-restoration tasks, flag code that assumes `train.csv` exists without first inspecting `./input`; these tasks commonly use image folders and a sample submission.",
        "  • Flag `OneHotEncoder(sparse=...)` without compatibility handling; recent sklearn uses `sparse_output=`.",
        "  • Flag code that writes only `./working/submission.csv` and never copies or writes the required `./submission/submission.csv`.",
        "",
        "---\n",
        "## 📋 Decision Rule\n",
        "",
        "**needs_revision=True** ONLY IF:",
        "  • P0 data leakage found (MUST FIX)",
        "  • OR P1 critical bug found",
        "",
        "**needs_revision=False** IF:",
        "  • No P0/P1 bugs found",
        "",
        "**Default**: Approve unless concrete logic bugs found"
    ]
    return guidelines


def get_code_review_response_format() -> list:
    """Code review response format."""
    return [
        "🚨 **CRITICAL: OUTPUT REQUIREMENT**",
        "",
        "**Required Fields:**",
        "- `needs_revision` (boolean): true if code has issues that must be fixed, false if code is correct",
        "- `reasoning` (string): EXACTLY 2-4 sentences explaining your decision (NO MORE)",
        "",
        "**Conditional Field:**",
        "- `revised_code` (string): ONLY if needs_revision=true, provide targeted fixes using SEARCH/REPLACE format",
        "",
        "🚫 **If needs_revision=false (code is correct):**",
        "- DO NOT provide revised_code (must be null/omitted)",
        "- Original code will be used as-is",
        "- This prevents accidental modifications to working code",
        "",
        "✅ **If needs_revision=true (code has issues):**",
        "- MUST provide revised_code using SEARCH/REPLACE diff format",
        "- Use <<<<<<< SEARCH / ======= / >>>>>>> REPLACE blocks for each fix",
        "- SEARCH block must match original code EXACTLY (character-by-character, same indentation)",
        "- Only include the specific buggy lines that need fixing",
        "- Can provide multiple SEARCH/REPLACE blocks for different issues",
        "- Preserve the solution approach and model architecture",
        "- Fix only the specific issues identified (metric mismatch, data leakage, API errors)",
        "- DO NOT change model architecture, data split method, or metric calculation (unless they are buggy)",
        "",
        "**Reasoning Field Guidelines:**",
        "⚠️ STRICT LENGTH LIMIT: Write EXACTLY 2-4 sentences. Be concise.",
        "Cover: (1) what issues found, (2) why they matter, (3) what will be fixed.",
        "DO NOT write detailed analysis, step-by-step checks, or comprehensive explanations.",
        "",
        "**Why this format matters:**",
        "The JSON schema format ensures that code is ONLY modified when necessary.",
        "When needs_revision=false, it's impossible to accidentally change working code.",
        "⚠️ reasoning MUST be 2-4 sentences only. Do NOT write long analysis or enumerate checks."
    ]
