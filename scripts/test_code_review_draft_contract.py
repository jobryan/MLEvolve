#!/usr/bin/env python3
"""Checks for the cheap-first-draft contract in the code review gate."""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]

# The local harness environment does not ship the agent runtime deps, so stub
# the direct imports of agents/code_review_agent.py and load it from its file
# path (bypassing agents/__init__.py, which pulls in the full agent stack).
LLM_CALLS: list[dict] = []


def _fake_query(system_message=None, user_message=None, func_spec=None, model=None, temperature=None, cfg=None):
    LLM_CALLS.append({"system_message": system_message})
    return {"needs_revision": False, "reasoning": "ok", "revised_code": None}


fake_llm = types.ModuleType("llm")
fake_llm.FunctionSpec = lambda **kwargs: SimpleNamespace(**kwargs)
fake_llm.query = _fake_query
sys.modules["llm"] = fake_llm

fake_engine = types.ModuleType("engine")
fake_search_node = types.ModuleType("engine.search_node")
fake_search_node.SearchNode = SimpleNamespace
sys.modules["engine"] = fake_engine
sys.modules["engine.search_node"] = fake_search_node

fake_agents = types.ModuleType("agents")
fake_prompts = types.ModuleType("agents.prompts")
fake_prompts.get_internet_clarification = lambda pretrain_model_dir: ["internet clarification line"]
fake_validation = types.ModuleType("agents.prompts.validation_template_prompts")
fake_validation.get_code_review_prompt = lambda task_desc, code: {
    "Instructions": {"Implementation guideline": ["baseline guideline"]}
}
fake_coder = types.ModuleType("agents.coder")
fake_diff_coder = types.ModuleType("agents.coder.diff_coder")
fake_diff_coder.SearchReplacePatcher = object
sys.modules["agents"] = fake_agents
sys.modules["agents.prompts"] = fake_prompts
sys.modules["agents.prompts.validation_template_prompts"] = fake_validation
sys.modules["agents.coder"] = fake_coder
sys.modules["agents.coder.diff_coder"] = fake_diff_coder

_spec = importlib.util.spec_from_file_location(
    "code_review_agent_under_test", ROOT / "agents/code_review_agent.py"
)
code_review_agent = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(code_review_agent)


def make_agent() -> SimpleNamespace:
    return SimpleNamespace(
        task_desc="task",
        cfg=SimpleNamespace(pretrain_model_dir=""),
        acfg=SimpleNamespace(use_diff_mode=True, code=SimpleNamespace(model="test-model", temp=0.0)),
    )


def make_node(code: str, stage: str = "draft") -> SimpleNamespace:
    return SimpleNamespace(id="node-1", stage=stage, code=code)


def run_review(code: str, stage: str = "draft", no_gpu: str | None = "1") -> str:
    saved = os.environ.get("MLEVOLVE_NO_GPU")
    try:
        os.environ.pop("MLEVOLVE_NO_GPU", None)
        if no_gpu is not None:
            os.environ["MLEVOLVE_NO_GPU"] = no_gpu
        LLM_CALLS.clear()
        return code_review_agent.run(make_agent(), make_node(code, stage=stage))
    finally:
        if saved is None:
            os.environ.pop("MLEVOLVE_NO_GPU", None)
        else:
            os.environ["MLEVOLVE_NO_GPU"] = saved


def assert_rejected(code: str) -> None:
    try:
        run_review(code)
    except code_review_agent.DraftContractViolation as exc:
        assert "Cheap-first-draft contract violation" in str(exc)
        assert not LLM_CALLS, "pre-check rejection must happen before the LLM review call"
        return
    raise AssertionError(f"expected DraftContractViolation for code: {code!r}")


def test_pretrained_downloads_rejected_on_no_gpu_drafts() -> None:
    assert_rejected("model = torch.hub.load('pytorch/vision', 'resnet18')\n")
    assert_rejected("model = AutoModel.from_pretrained('bert-base-uncased')\n")
    assert_rejected("m = timm.create_model('resnet18', pretrained=True)\n")


def test_local_paths_and_scratch_models_pass_precheck() -> None:
    # from_pretrained with a local path is allowed (no download).
    result = run_review("model = AutoModel.from_pretrained('/data/pretrain_models/bert')\n")
    assert LLM_CALLS, "clean draft should proceed to the LLM review"
    assert "from_pretrained" in result
    result = run_review("m = timm.create_model('resnet18', pretrained=False)\n")
    assert "timm.create_model" in result
    result = run_review("model = nn.Linear(10, 2)\n")
    assert "nn.Linear" in result


def test_contract_gated_on_env_and_stage() -> None:
    hub_code = "model = torch.hub.load('pytorch/vision', 'resnet18')\n"
    # No MLEVOLVE_NO_GPU: pretrained downloads are allowed even in drafts.
    result = run_review(hub_code, no_gpu=None)
    assert result == hub_code
    assert LLM_CALLS
    # Non-draft stages are exempt even with MLEVOLVE_NO_GPU=1.
    result = run_review(hub_code, stage="improve")
    assert result == hub_code
    assert LLM_CALLS


def test_soft_instruction_added_for_no_gpu_drafts() -> None:
    run_review("model = nn.Linear(10, 2)\n")
    prompt = LLM_CALLS[-1]["system_message"]
    assert "Cheap first draft (CPU-only worker)" in prompt["Instructions"]
    assert any(
        "FIRST DRAFT" in line
        for line in prompt["Instructions"]["Cheap first draft (CPU-only worker)"]
    )

    run_review("model = nn.Linear(10, 2)\n", no_gpu=None)
    prompt = LLM_CALLS[-1]["system_message"]
    assert "Cheap first draft (CPU-only worker)" not in prompt["Instructions"]

    run_review("model = nn.Linear(10, 2)\n", stage="improve")
    prompt = LLM_CALLS[-1]["system_message"]
    assert "Cheap first draft (CPU-only worker)" not in prompt["Instructions"]


def test_violation_helper_reports_all_patterns() -> None:
    code = (
        "a = torch.hub.load('x', 'y')\n"
        "b = AutoModel.from_pretrained('bert-base-uncased')\n"
        "c = timm.create_model('resnet18', pretrained=True)\n"
    )
    violations = code_review_agent.draft_contract_violations(code)
    assert len(violations) == 3
    assert code_review_agent.draft_contract_violations("x = 1\n") == []


def main() -> int:
    test_pretrained_downloads_rejected_on_no_gpu_drafts()
    test_local_paths_and_scratch_models_pass_precheck()
    test_contract_gated_on_env_and_stage()
    test_soft_instruction_added_for_no_gpu_drafts()
    test_violation_helper_reports_all_patterns()
    print("code review draft contract tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
