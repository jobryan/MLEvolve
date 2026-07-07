"""Implementation guideline."""

import os
import time

import humanize


def get_impl_guideline_from_agent(agent):
    """Build implementation guideline from agent config."""
    tot_time_remaining = agent.acfg.time_limit - (time.time() - agent.start_time)
    exec_timeout = int(min(agent.cfg.exec.timeout, tot_time_remaining))
    return get_impl_guideline(
        tot_time_remaining=tot_time_remaining,
        steps_remaining=agent.acfg.steps - agent.current_step,
        exec_timeout=exec_timeout,
        expose_prediction=getattr(agent.acfg, "expose_prediction", False),
        k_fold_validation=getattr(agent.acfg, "k_fold_validation", 0),
        pretrain_model_dir=getattr(agent.cfg, "pretrain_model_dir", ""),
    )


def _format_time(time_in_sec):
    """Format seconds for display."""
    return f"{int(time_in_sec) // 3600}h {(int(time_in_sec) % 3600) // 60}m {int(time_in_sec) % 60}s"


def get_impl_guideline(
    tot_time_remaining: float,
    steps_remaining: int,
    exec_timeout: int,
    expose_prediction: bool = False,
    k_fold_validation: int = 0,
    pretrain_model_dir: str = "",
) -> dict:
    """Build implementation guideline from time and config."""
    impl_guideline = [
        f"**Resource Budget**: Time left ≈ {_format_time(tot_time_remaining)} | Steps left = {steps_remaining} | Max execution time per run = {humanize.naturaldelta(exec_timeout)}",
        "",
        "**Note:** Code execution MUST complete within 9 hours (hard limit) — any solution exceeding this will be invalid. Within this constraint, prioritize performance and optimization.",
        "🎯 **CRITICAL REQUIREMENTS** (Non-Negotiable):",
        "",
        "**1. Model Inference for ALL Predictions**",
        "• EVERY prediction (validation & test) MUST come from trained model's forward pass",
        "• Process: Load data → Preprocess → model.predict()/model.forward() → Save predictions",
        "• ❌ FORBIDDEN: Constants, placeholders, dummy values, empty arrays, statistics, random numbers",
        "• ❌ FORBIDDEN: Fake/mock metric functions (must use real sklearn.metrics or correct manual implementation)",
        "• Why: Shortcuts create fake high validation scores but fail on test (CRITICAL SYSTEM FAILURE)",
        "",
        "**2. Generate submission.csv**",
        "• Path: `./submission/submission.csv` (NOT ./working/submission.csv)",
        "• Content: Model predictions on ALL test samples",
        "• Format: Follow task description exactly",
        "",
        "**3. Print Validation Metric**",
        "• MUST print: `print(f'Final Validation Score: {score}')`",
        "• Score MUST be computed on hold-out validation set using proper metric formula",
        "• CRITICAL CONSISTENCY REQUIREMENT: Ensure that validation and test inference use IDENTICAL processing logic. Any differences in how validation and test data are handled (such as post-processing, reconstruction, or formatting) can cause large performance gaps between validation and test sets. Maintain consistency across all data processing steps for both validation and test phases.",
        "",
        "**4. Preserve Task Semantics in Local Validation**",
        "• Your local validation must evaluate the SAME core task as the final submission, not an easier proxy sub-problem.",
        "• ❌ FORBIDDEN: Silently redefining the task into a simpler objective (for example: multi-target -> single-target, detection -> presence-only classification, ranking -> plain classification, structured prediction -> one-field prediction).",
        "• If the submission predicts structured outputs, your validation metric must cover the key predicted structure rather than only a weak sub-component.",
        "• The validation target, prediction format, and post-processing logic must stay semantically aligned with the required submission format.",
        "",
        "**5. Make Validation Auditable**",
        "• In code and logs, make the validation setup easy to audit: split method, metric formula, predicted target, and any threshold/post-processing used.",
        "• The reported `Final Validation Score` must be computed with the official metric definition, or a task-faithful local implementation of that same metric.",
        "• ❌ FORBIDDEN: Using a proxy metric as the main validation score for model comparison, search ranking, or best-solution selection.",
        "• Do not report a validation score from a metric that ignores critical task dimensions required by the leaderboard.",
        "",
        "**6. Tabular dtype safety**",
        "• For CSV/tabular data, inspect pandas dtypes before preprocessing.",
        "• Build numeric feature lists from actual numeric dtypes after dropping id/target columns; do not infer dtype from column names.",
        "• Encode or exclude object/string/category columns before StandardScaler, median/mean imputation, torch float tensors, or numeric model matrices.",
        "• For high-cardinality strings, avoid one-hot/get_dummies and use bounded encodings such as factorize/ordinal, frequency/count, hashing, string length, or fixed-position character features.",
        "• Under short search budgets (about 1-2 solution nodes or <30 minutes), first build a robust simple tabular baseline: LightGBM, XGBoost, ExtraTrees/RandomForest, HistGradientBoosting, or logistic/linear models on a validated numeric feature matrix.",
        "• Avoid custom TabNet-like/attention/embedding neural architectures as the first tabular draft unless the code proves all categorical indices, embedding dimensions, and tensor widths are bounded and consistent. Complex tabular neural nets are follow-up experiments after a valid submission exists.",
        "",
        "**7. Cross-domain short-budget reliability**",
        "• For small image classification tasks, first use a local, fast baseline: downsample images, normalize pixels, and train a small CNN or sklearn model on flattened/color-summary features. Do NOT call `torch.hub.load`, download checkpoints, or depend on DINO/SigLIP/transformer backbones unless the exact local checkpoint path is visible in the data preview.",
        "• For `aerial-cactus-identification`, the first valid baseline must use local train/test images plus the CSV labels and sample submission. Do not use DINO, torch.hub, timm, transformers, pretrained checkpoints, or external downloads in the first draft. A reliable baseline is: load 32x32 images with PIL/skimage/cv2, scale pixels to [0, 1], train LogisticRegression/SGDClassifier/RandomForest on flattened pixels or a tiny CNN, print validation AUC/log loss, then write probabilities to `./submission/submission.csv` in sample-submission order.",
        "• For binary PyTorch classifiers, make output and target shapes explicit: use `logits = model(x).view(-1)` and `targets = y.float().view(-1)` for BCE/BCEWithLogits, and generate probabilities with `torch.sigmoid(logits)`. Do not call `squeeze(1)` unless dimension 1 is known to exist.",
        "• PyTorch schedulers must use APIs available in current torch. For `torch.optim.lr_scheduler.ReduceLROnPlateau`, do not pass `verbose=`; log learning-rate changes manually if needed.",
        "• Image flips, rotations, and array slicing can produce NumPy arrays with negative strides. Call `.copy()` or `np.ascontiguousarray(...)` before `torch.tensor(...)` or `torch.from_numpy(...)`.",
        "• For denoising/image-restoration tasks, inspect `./input` first. Do not assume `train.csv` exists; many benchmarks provide image folders such as train/test/train_cleaned plus a sample submission.",
        "• For `skimage.filters.rank`, use `footprint=...`; do not use the removed `selem=` keyword.",
        "• For text classification tasks, first use `sklearn` TF-IDF/character n-grams plus LogisticRegression, LinearSVC+calibration, SGDClassifier, or Naive Bayes. Do NOT require NLTK resources such as `punkt`/`punkt_tab`, spaCy models, or transformer fine-tuning for the first valid baseline under a short/no-GPU budget.",
        "• If defining custom sklearn transformers, inherit `BaseEstimator` and `TransformerMixin` or implement `fit`, `transform`, and `fit_transform` correctly before using them in a Pipeline/FeatureUnion. Prefer built-in `TfidfVectorizer`, `CountVectorizer`, and `FunctionTransformer` over custom transformer classes under short budgets.",
        "• For materials/scientific tabular tasks such as NOMAD, start from provided CSV columns and robust numeric/categorical encodings before optional geometry parsing. For `nomad2018-predict-transparent-conductors`, the first valid baseline should use the actual numeric columns present in `train.csv`/`test.csv` and a simple multi-output regressor or one regressor per target; do NOT require `geometry.xyz` parsing for the first draft.",
        "• NOMAD has two regression targets. Preserve both targets throughout validation and submission with `MultiOutputRegressor`, one fitted model per target, or an estimator that explicitly accepts 2D targets. Do not flatten the targets to one column, and do not pass a 2D target into a single-output estimator unless that estimator supports multi-output regression.",
        "• If parsing NOMAD `geometry.xyz`, skip comment/header lines and only parse rows that look like `element x y z`; do NOT assume the first line is always an integer atom count. Do not access `row['lv1']`, `row['lv2']`, `row['lv3']`, `row['alpha']`, `row['beta']`, or `row['gamma']` unless those names are confirmed in `row.index`; lattice data may not be CSV columns in the prepared benchmark.",
        "• Use sklearn APIs compatible with recent versions: `OneHotEncoder(sparse_output=False, handle_unknown='ignore')`, or a try/except fallback to `sparse=False` only if needed.",
        "• If GPU is unavailable, prefer CPU-safe baselines over large neural architectures. A valid sample-shaped submission with an honest validation metric is better than an ambitious model that times out or fails to export.",
        "",
        *(
            [
                "**8. CPU-only execution environment (HARD CONSTRAINT)**",
                "• This environment has NO GPU: `torch.cuda.is_available()` is False.",
                "• Every device selection MUST be gated: `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`. ❌ FORBIDDEN: unconditional `device='cuda'`, `.cuda()`, `.to('cuda')`, or `device_type='cuda'`.",
                "• Do not enable GPU-only library options: no LightGBM `device='gpu'`, no XGBoost `tree_method='gpu_hist'`, no CUDA AMP.",
                "• Size models and batch counts for CPU: small architectures, few epochs, subsampled data if needed to finish within the execution timeout.",
                "",
            ]
            if os.environ.get("MLEVOLVE_NO_GPU") == "1"
            else []
        ),
        "📁 **Directories**: Input data in `./input/`, submission in `./submission/`, temp files in `./working/`",
        "",
        (
            "📦 **Packages & Internet**: numpy, pandas, sklearn, torch, transformers, timm, xgboost, lightgbm (all pre-installed). "
            "This environment is OFFLINE: NO internet access, NO `torch.hub.load()`, NO HuggingFace/checkpoint downloads, NO `pip install`. "
            "Use only local data under `./input/` and pre-installed libraries."
            if os.environ.get("MLEVOLVE_OFFLINE") == "1"
            else "📦 **Packages & Internet**: numpy, pandas, sklearn, torch, transformers, timm, xgboost, lightgbm (all pre-installed). torch.hub.load(), HuggingFace, etc. available during development."
        )
        + (f" Offline models at `{pretrain_model_dir}`" if pretrain_model_dir else ""),
        "",
        "⚠️ **API Compatibility**: LightGBM/XGBoost: ❌ `fit(..., early_stopping_rounds=...)` → ✅ LightGBM: `fit(..., callbacks=[lgb.early_stopping(...)])` ✅ XGBoost: `XGBClassifier(early_stopping_rounds=...)`",
        "• LightGBM native `lgb.train`: use `callbacks=[lgb.early_stopping(...), lgb.log_evaluation(...), lgb.record_evaluation(evals_result)]`; do not pass `early_stopping_rounds=`, `evals_result=`, or `verbose_eval=` directly.",
        "• XGBoost sklearn API: put `eval_metric` and `early_stopping_rounds` in the `XGBClassifier`/`XGBRegressor` constructor, not in `.fit(...)`. For `MultiOutputRegressor`, avoid early-stopping fit kwargs unless you explicitly fit one model per target.",
        "• PyTorch `ReduceLROnPlateau`: do not pass `verbose=`.",
        "• scikit-image rank filters: use `footprint=...`, not `selem=...`.",
        "• AdamW: ❌ `from transformers import AdamW` (deprecated) → ✅ `from torch.optim import AdamW`",
        "",
        "🚫 **Execution Guidelines**:",
        "• NO tqdm (not installed), NO verbose=1",
        "• Print only 1 line per epoch (minimize logging)",
        "• Use DataLoader with num_workers>=2 for speed",
        "",
        "⚠️  **Self-Check Before Finalizing**:",
        "□ Did predictions pass through model's learned weights during inference? (If NO → INVALID)",
        "□ Did I generate submission.csv in correct path with ALL test predictions?",
        "□ Did I print validation metric as the last line?",
        "□ Did I use the COMPLETE training dataset (not a tiny subset)?",
        "□ Did my local validation preserve the original task semantics instead of a simpler proxy?",
        "□ Is my reported `Final Validation Score` computed with the official metric definition rather than a proxy metric?",
        "□ Did I keep raw object/string/category columns out of numeric scalers, imputers, tensors, and model matrices?",
        "□ For image/text/materials tasks, did I avoid unavailable external resources and choose a CPU-safe first baseline that can finish within this run?",
    ]
    if expose_prediction:
        impl_guideline.append(
            "The implementation should include a predict() function, "
            "allowing users to seamlessly reuse the code to make predictions on new data. "
            "The prediction function should be well-documented, especially the function signature."
        )

    if k_fold_validation > 1:
        impl_guideline.append(
            f"The evaluation should be based on {k_fold_validation}-fold cross-validation but only if that's an appropriate evaluation for the task at hand."
        )

    return {"Implementation guideline": impl_guideline}
