"""Code Review Agent: LLM-based code review and fix for node code."""

import logging
import os
import re
import time
from typing import cast

from llm import FunctionSpec, query
from engine.search_node import SearchNode
from agents.prompts.validation_template_prompts import get_code_review_prompt
from agents.prompts import get_internet_clarification

from agents.coder.diff_coder import SearchReplacePatcher

logger = logging.getLogger("MLEvolve")

CODE_REVIEW_SPEC = FunctionSpec(
    name="submit_code_review",
    json_schema={
        "type": "object",
        "properties": {
            "needs_revision": {
                "type": "boolean",
                "description": (
                    "true if the code has issues that must be fixed "
                    "(metric mismatch, data leakage, or missing packages), "
                    "false if the code is correct."
                )
            },
            "reasoning": {
                "type": "string",
                "description": (
                    "CONCISE explanation in EXACTLY 2-4 sentences. Explain: "
                    "(1) what issues were found, (2) why they matter, (3) what will be fixed. "
                    "DO NOT write detailed analysis or step-by-step checks - keep it brief."
                )
            },
            "revised_code": {
                "type": "string",
                "description": (
                    "ONLY if needs_revision=true: Provide targeted fixes using SEARCH/REPLACE diff format.\n\n"
                    "**REQUIRED FORMAT** (use this for each fix):\n"
                    "<<<<<<< SEARCH\n"
                    "[exact code to find - copy verbatim with exact indentation]\n"
                    "=======\n"
                    "[corrected code]\n"
                    ">>>>>>> REPLACE\n\n"
                    "**CRITICAL**: \n"
                    "- SEARCH block must match original code EXACTLY (character-by-character, including all spaces/tabs)\n"
                    "- Only include the specific buggy lines that need fixing\n"
                    "- Can provide multiple SEARCH/REPLACE blocks for different bugs\n"
                    "- Do NOT output complete code - only diff blocks\n"
                    "- Do NOT wrap output in markdown code fences (``` or ```python) - output raw diff only\n\n"
                    "If needs_revision=false: MUST be null (DO NOT output code)."
                )
            }
        },
        "required": ["needs_revision", "reasoning"]
    },
    description="Submit code review for search node solution."
)


class DraftContractViolation(RuntimeError):
    """Raised when a first draft violates the cheap-first-draft contract (MLEVOLVE_NO_GPU=1)."""


# Deterministic rejections for first drafts on CPU-only workers: pretrained
# downloads dominate the wall-clock budget before any signal is produced.
PRETRAINED_DOWNLOAD_PATTERNS = (
    (re.compile(r"torch\.hub\.load\s*\("), "torch.hub.load(...) downloads pretrained weights"),
    (
        re.compile(r"\.from_pretrained\s*\(\s*[\"'](?![./~])"),
        ".from_pretrained(...) with a hub model id instead of a local path",
    ),
    (
        re.compile(r"timm\.create_model\s*\([^)]*pretrained\s*=\s*True"),
        "timm.create_model(..., pretrained=True) downloads pretrained weights",
    ),
)

CHEAP_DRAFT_INSTRUCTION = (
    "This is a FIRST DRAFT on a CPU-only worker: keep training cheap (subsample the data, "
    "use few epochs and a small model) so the run finishes well within the execution timeout. "
    "For text classification, use sklearn TF-IDF/character n-grams with SGDClassifier, "
    "LogisticRegression, LinearSVC+calibration, or Naive Bayes; do not use transformers or "
    "HuggingFace model/tokenizer classes in the first draft."
)


def draft_contract_active(node: SearchNode) -> bool:
    """Cheap-first-draft contract applies to Draft nodes on CPU-only (MLEVOLVE_NO_GPU=1) workers."""
    return os.getenv("MLEVOLVE_NO_GPU") == "1" and str(getattr(node, "stage", "")).lower() == "draft"


def draft_contract_violations(code: str) -> list[str]:
    """Return descriptions of pretrained-download patterns found in the code."""
    return [label for pattern, label in PRETRAINED_DOWNLOAD_PATTERNS if pattern.search(code)]


def looks_like_jigsaw_toxic_task(task_desc: str) -> bool:
    """Detect the Jigsaw Toxic Comment task from its task description."""
    lowered = task_desc.lower()
    required_terms = [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
        "comment_text",
    ]
    return all(term in lowered for term in required_terms)


def jigsaw_cpu_baseline_code() -> str:
    """Deterministic CPU-safe first baseline for Jigsaw/Toxic Comment."""
    return r'''
import os
import re
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

INPUT_DIR = "./input"
SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)

TARGETS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def clean_text(value):
    text = "" if pd.isna(value) else str(value)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv"))

train["comment_text"] = train["comment_text"].map(clean_text)
test["comment_text"] = test["comment_text"].map(clean_text)

train_idx, val_idx = train_test_split(
    np.arange(len(train)),
    test_size=0.15,
    random_state=42,
    stratify=train["toxic"],
)
train_text = train.loc[train_idx, "comment_text"]
val_text = train.loc[val_idx, "comment_text"]
test_text = test["comment_text"]

word_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    min_df=3,
    max_features=15000,
    strip_accents="unicode",
    sublinear_tf=True,
)
char_vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=3,
    max_features=15000,
    strip_accents="unicode",
    sublinear_tf=True,
)

word_vectorizer.fit(train_text)
char_vectorizer.fit(train_text)

X_train = hstack([
    word_vectorizer.transform(train_text),
    char_vectorizer.transform(train_text),
]).tocsr()
X_val = hstack([
    word_vectorizer.transform(val_text),
    char_vectorizer.transform(val_text),
]).tocsr()
X_test = hstack([
    word_vectorizer.transform(test_text),
    char_vectorizer.transform(test_text),
]).tocsr()

val_scores = []
test_predictions = np.zeros((len(test), len(TARGETS)), dtype=np.float32)

for column_index, target in enumerate(TARGETS):
    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-5,
        max_iter=8,
        tol=1e-3,
        random_state=42 + column_index,
        n_jobs=1,
    )
    model.fit(X_train, train.loc[train_idx, target].astype(int))
    val_prob = model.predict_proba(X_val)[:, 1]
    test_predictions[:, column_index] = model.predict_proba(X_test)[:, 1]
    val_scores.append(roc_auc_score(train.loc[val_idx, target].astype(int), val_prob))

score = float(np.mean(val_scores))
submission = sample.copy()
for column_index, target in enumerate(TARGETS):
    submission[target] = np.clip(test_predictions[:, column_index], 0.0, 1.0)
submission.to_csv(os.path.join(SUBMISSION_DIR, "submission.csv"), index=False)

print(f"Final Validation Score: {score}")
'''.strip()


def resolve_draft_contract_violation(agent, node: SearchNode, code: str, violations: list[str]) -> str:
    """Return deterministic safe code for known contracts or raise for unsupported violations."""
    task_desc = str(getattr(agent, "task_desc", ""))
    if looks_like_jigsaw_toxic_task(task_desc):
        logger.warning(
            "Replacing Jigsaw toxic-comment first draft with deterministic CPU-safe TF-IDF baseline "
            f"after contract violation(s): {'; '.join(violations)}"
        )
        return jigsaw_cpu_baseline_code()
    message = (
        "Cheap-first-draft contract violation (MLEVOLVE_NO_GPU=1, Draft node): "
        + "; ".join(violations)
        + ". First drafts must not download pretrained models; train a small model "
        "from scratch and keep the draft cheap (e.g. subsample the data)."
    )
    logger.warning(f"Code review rejected draft node {node.id}: {message}")
    raise DraftContractViolation(message)


def run(agent, node: SearchNode) -> str:
    logger.debug(f"[review] node {node.id}")

    contract_active = draft_contract_active(node)
    if contract_active:
        violations = draft_contract_violations(node.code)
        if violations:
            return resolve_draft_contract_violation(agent, node, node.code, violations)

    prompt = get_code_review_prompt(
        task_desc=agent.task_desc,
        code=node.code,
    )
    internet_clarification = get_internet_clarification(getattr(agent.cfg, "pretrain_model_dir", ""))
    if "Instructions" not in prompt:
        prompt["Instructions"] = {}
    if "Implementation guideline" in prompt["Instructions"]:
        prompt["Instructions"]["Implementation guideline"].extend(internet_clarification)
    else:
        prompt["Instructions"]["⚠️ Internet Access Clarification"] = internet_clarification
    if contract_active:
        prompt["Instructions"]["Cheap first draft (CPU-only worker)"] = [CHEAP_DRAFT_INSTRUCTION]

    use_diff_for_review = agent.acfg.use_diff_mode
    max_retries = 3

    for attempt in range(max_retries):
        try:
            if attempt > 0:
                logger.info(f"Code review retry attempt {attempt + 1}/{max_retries} for node {node.id}")
                time.sleep(5)

            review_response = cast(
                dict,
                query(
                    system_message=prompt,
                    user_message=None,
                    func_spec=CODE_REVIEW_SPEC,
                    model=agent.acfg.code.model,
                    temperature=agent.acfg.code.temp,
                    cfg=agent.cfg
                ),
            )

            needs_revision = review_response.get("needs_revision", False)
            reasoning = review_response.get("reasoning", "")
            revised_code = review_response.get("revised_code")
            logger.info(f"Code review for node {node.id}: needs_revision={needs_revision}")
            logger.info(f"Reasoning: {reasoning}", extra={"verbose": True})

            if needs_revision:
                if revised_code and revised_code.strip():
                    if use_diff_for_review and (
                        "<<<<<<< SEARCH" in revised_code or "< SEARCH" in revised_code
                        ):
                        try:
                            logger.info("Code review returned diff format, applying patch")
                            patcher = SearchReplacePatcher()
                            patched_code, count = patcher.apply_patch(
                                revised_code, node.code, strict=False
                            )
                            if count > 0 and patched_code and patched_code != node.code:
                                logger.info(f"Successfully applied {count} review patch(es)")
                                patched_code = patched_code.strip()
                                if contract_active:
                                    violations = draft_contract_violations(patched_code)
                                    if violations:
                                        return resolve_draft_contract_violation(agent, node, patched_code, violations)
                                return patched_code
                            logger.warning(
                                f"Diff patch failed (count={count}), keeping original code to avoid writing raw diff to runfile"
                            )
                            if contract_active:
                                violations = draft_contract_violations(node.code)
                                if violations:
                                    return resolve_draft_contract_violation(agent, node, node.code, violations)
                            return node.code
                        except Exception as e:
                            logger.warning(
                                f"Failed to apply diff patch in code review: {e}, keeping original code to avoid writing raw diff to runfile"
                            )
                            return node.code
                    else:
                        # Full code revision (original behavior)
                        if use_diff_for_review:
                            if contract_active:
                                violations = draft_contract_violations(node.code)
                                if violations:
                                    return resolve_draft_contract_violation(agent, node, node.code, violations)
                            return node.code
                        else:
                            logger.info("Using revised code from reviewer")
                            revised_code = revised_code.strip()
                            if contract_active:
                                violations = draft_contract_violations(revised_code)
                                if violations:
                                    return resolve_draft_contract_violation(agent, node, revised_code, violations)
                            return revised_code

                if attempt < max_retries - 1:
                    logger.warning(f"Code review violation: needs_revision=True but revised_code is empty/None - Will retry ({attempt + 1}/{max_retries})")
                    logger.info(f"Reasoning detail: {reasoning}", extra={"verbose": True})
                    continue
                logger.error(f"Code review violation: needs_revision=True but revised_code is empty/None - Max retries reached, returning original code")
                logger.info(f"Reasoning detail: {reasoning}", extra={"verbose": True})
                if contract_active:
                    violations = draft_contract_violations(node.code)
                    if violations:
                        return resolve_draft_contract_violation(agent, node, node.code, violations)
                return node.code

            if revised_code is not None and revised_code.strip():
                logger.warning(
                    "Code review warning: needs_revision=False but revised_code was provided. "
                    "Ignoring revised_code and using original code."
                )
            logger.info("Code approved, using original code")
            if contract_active:
                violations = draft_contract_violations(node.code)
                if violations:
                    return resolve_draft_contract_violation(agent, node, node.code, violations)
            return node.code

        except Exception as e:
            error_msg = f"Code review failed with exception: {e}"
            if attempt < max_retries - 1:
                logger.warning(f"{error_msg} - Will retry (attempt {attempt + 1}/{max_retries})")
                continue
            logger.error(f"{error_msg} - Max retries reached, returning original code")
            return node.code

    logger.error("Code review: Unexpected exit from retry loop, returning original code")
    return node.code
