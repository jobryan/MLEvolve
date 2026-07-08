#!/usr/bin/env python3
"""Validate AWS worker setup artifacts for the ablation project."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ENV_CONFIG = ROOT / "configs/ablations/aws_worker_environment.json"
JOB_TEMPLATE = ROOT / "configs/ablations/aws_batch_job_definition.template.json"
PERMISSIONS_TEMPLATE = ROOT / "configs/ablations/aws_required_permissions_policy.template.json"
DOCKERFILE = ROOT / "docker/aws-ablation-worker/Dockerfile"
ENTRYPOINT = ROOT / "docker/aws-ablation-worker/entrypoint.sh"
BUILD_SCRIPT = ROOT / "scripts/build_push_aws_worker_image.sh"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def main() -> int:
    errors: list[str] = []
    for path in (ENV_CONFIG, JOB_TEMPLATE, PERMISSIONS_TEMPLATE, DOCKERFILE, ENTRYPOINT, BUILD_SCRIPT):
        require(path.exists(), f"missing {path}", errors)

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    env_config = load_json(ENV_CONFIG)
    job_template = load_json(JOB_TEMPLATE)
    permissions_template = load_json(PERMISSIONS_TEMPLATE)
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")

    require(env_config.get("environment_id") == "aws-ablation-worker-v1", "unexpected environment_id", errors)
    for command in ("python3", "mlebench", "kaggle", "docker"):
        require(command in env_config.get("required_commands", []), f"required command missing: {command}", errors)
    for env_name in ("KAGGLE_USERNAME", "KAGGLE_KEY", "OPENAI_API_KEY"):
        require(env_name in env_config.get("required_environment_variables", []), f"required env missing: {env_name}", errors)
    require("GEMINI_API_KEY" not in env_config.get("required_environment_variables", []), "GEMINI_API_KEY should not be required", errors)
    for env_name in ("MLEVOLVE_CODE_MODEL", "MLEVOLVE_FEEDBACK_MODEL", "MLEVOLVE_CODE_API_KEY", "MLEVOLVE_FEEDBACK_API_KEY", "MLEVOLVE_CODE_BASE_URL", "MLEVOLVE_FEEDBACK_BASE_URL"):
        require(env_name in env_config.get("optional_environment_variables", []), f"optional env missing: {env_name}", errors)

    container = job_template.get("containerProperties", {})
    require(job_template.get("type") == "container", "job definition type must be container", errors)
    require(container.get("image") == "<ECR_IMAGE_URI>", "job template image placeholder missing", errors)
    require(container.get("jobRoleArn") == "<BATCH_JOB_ROLE_ARN>", "job role placeholder missing", errors)
    secret_names = {secret.get("name") for secret in container.get("secrets", [])}
    for env_name in ("KAGGLE_USERNAME", "KAGGLE_KEY", "OPENAI_API_KEY"):
        require(env_name in secret_names, f"job template secret missing: {env_name}", errors)
    for env_name in ("GEMINI_API_KEY", "MLEVOLVE_CODE_API_KEY", "MLEVOLVE_FEEDBACK_API_KEY"):
        require(env_name not in secret_names, f"job template should not require secret: {env_name}", errors)

    require(
        "submitter_policy_document" in permissions_template,
        "permissions template missing submitter_policy_document",
        errors,
    )
    require(
        "worker_job_role_policy_document" in permissions_template,
        "permissions template missing worker_job_role_policy_document",
        errors,
    )
    submitter_policy_text = json.dumps(permissions_template.get("submitter_policy_document", {}), sort_keys=True)
    for action in ("batch:SubmitJob", "batch:DescribeJobs", "ecr:DescribeRepositories", "iam:PassRole"):
        require(action in submitter_policy_text, f"permissions template missing action: {action}", errors)
    for marker in ("<ECR_REPOSITORY_NAME>", "<SECRET_PREFIX>", "<BATCH_JOB_ROLE_NAME>"):
        require(marker in submitter_policy_text, f"permissions template missing placeholder: {marker}", errors)

    for expected in (
        "requirements_base.txt",
        "requirements_ml.txt",
        "requirements_domain.txt",
        "AI-Scientist-v2-ablation/requirements.txt",
        "pip install -e /opt/mle-bench",
        "ENTRYPOINT",
    ):
        require(expected in dockerfile, f"Dockerfile missing expected text: {expected}", errors)

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print("AWS worker setup artifacts validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
