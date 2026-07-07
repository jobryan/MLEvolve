# AWS Worker Setup For Ablation Runs

Status: setup guide
Date: 2026-06-23

## Purpose

T22 screening and later experiment phases should run on AWS workers, not the local Conductor workspace. The local workspace owns planning, manifests, schemas, analysis code, and review artifacts. AWS workers own runtime dependencies, benchmark data, model calls, grading, and artifact production.

## Required Worker Contract

The AWS worker must be able to run:

```bash
python3 scripts/check_ablation_runtime_readiness.py --environment-label aws-execution-host
python3 scripts/run_ablation_manifest.py --manifest <manifest.json>
```

The worker environment contract is defined in:

```text
configs/ablations/aws_worker_environment.json
```

## Recommended Execution Substrate

Use AWS Batch unless there is a strong reason to use ECS, EC2, or SageMaker directly.

Why AWS Batch:

- One immutable run manifest maps to one job.
- Failed jobs remain inspectable.
- CPU, memory, GPU, and timeout controls map cleanly from the run manifest.
- Job specs can be generated without changing experiment code.

## Worker Image

Recommended image shape:

- Base: CUDA-enabled Ubuntu image, for example `nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04`.
- Python: 3.11.
- Repo: this workspace plus `.context/external/AI-Scientist-v2-ablation`.
- Dependencies:
  - `requirements_base.txt`
  - `requirements_ml.txt`
  - `requirements_domain.txt`
  - `.context/external/AI-Scientist-v2-ablation/requirements.txt`
  - MLE-bench CLI/module
  - Kaggle CLI
  - Docker CLI/runtime access if MLE-bench execution path requires it

If the combined dependency stack conflicts, build separate MLEvolve and AI Scientist v2 worker images and route jobs by `manifest.system`.

## Secrets

Resolved secret values must not be committed or written into reports.

Required runtime environment variables:

```bash
KAGGLE_USERNAME
KAGGLE_KEY
OPENAI_API_KEY
MLEBENCH_DATASET_DIR
ABLATION_ARTIFACT_ROOT
```

Kaggle credential references are documented in:

```text
.context/ablation/kaggle_credentials_1password.md
```

Model-provider credential references and gaps are documented in:

```text
.context/ablation/model_credentials_1password.md
```

MLEvolve receives provider credentials through manifest-dispatch CLI overrides:

```bash
MLEVOLVE_CODE_API_KEY
MLEVOLVE_FEEDBACK_API_KEY
MLEVOLVE_CODE_MODEL
MLEVOLVE_FEEDBACK_MODEL
MLEVOLVE_CODE_BASE_URL
MLEVOLVE_FEEDBACK_BASE_URL
```

By default, the dispatcher routes MLEvolve to OpenAI-compatible models and uses `OPENAI_API_KEY`. Set `MLEVOLVE_CODE_MODEL` / `MLEVOLVE_FEEDBACK_MODEL` to change the model names, or set `MLEVOLVE_CODE_API_KEY` / `MLEVOLVE_FEEDBACK_API_KEY` when code and feedback roles should use a different provider credential.

Inject secrets into AWS through the selected secret manager path. The job specs generated here intentionally do not contain resolved secret values.

## Data Preparation

Before T22, prepare or verify the three smoke tasks on the AWS worker data volume:

```bash
mlebench prepare -c aerial-cactus-identification
mlebench prepare -c nomad2018-predict-transparent-conductors
mlebench prepare -c spooky-author-identification
```

The prepared dataset root must be available as:

```bash
MLEBENCH_DATASET_DIR=/path/to/mle-bench
```

## Readiness Gate

Run this on the AWS worker image/instance:

```bash
python3 scripts/check_ablation_runtime_readiness.py --environment-label aws-execution-host
```

T22 should not start until the report marks these groups ready or documents an intentionally unused missing check:

- `mlevolve_smoke`
- `ais_v2_smoke`
- `shared_mle_bench`
- `shared_mle_bench_prepare`
- `screening`

## Manifest Dry-Run Gate

Generate smoke manifests locally or on the worker:

```bash
python3 scripts/run_ablation_matrix.py --phase smoke --system mlevolve --force
python3 scripts/run_ablation_matrix.py --phase smoke --system ai_scientist_v2 --force
```

Then verify dispatch on the AWS worker:

```bash
python3 scripts/run_ablation_manifest.py --dry-run --manifest .context/ablation/run_manifests/<mlevolve-manifest>.json
python3 scripts/run_ablation_manifest.py --dry-run --manifest .context/ablation/run_manifests/<ais-v2-manifest>.json
```

## AWS Batch Job Specs

Before rendering runnable job specs, complete T21.6 provisioning:

- Push the worker image to ECR.
- Register the Batch job definition from `configs/ablations/aws_batch_job_definition.template.json`.
- Configure runtime secrets and roles.
- Fill the IAM/resource gaps described in `configs/ablations/aws_required_permissions_policy.template.json`.
- Identify the real Batch queue and active job definition revision.

Render AWS Batch job specs after manifests are generated and uploaded or made available to workers:

```bash
python3 scripts/build_aws_ablation_jobs.py \
  --manifest-jsonl .context/ablation/screening_manifests.jsonl \
  --job-queue <aws-batch-queue> \
  --job-definition <aws-batch-job-definition> \
  --manifest-base-uri s3://autoresearch-experiments-058264252788-us-east-1/mlevolve-ai-scientist-v2-ablation/manifests/screening \
  --output-jsonl .context/ablation/aws_jobs/screening_jobs.jsonl
```

Submit through the reviewed helper after replacing queue/job-definition placeholders:

```bash
python3 scripts/submit_aws_ablation_jobs.py \
  --jobs-jsonl .context/ablation/aws_jobs/screening_jobs.jsonl \
  --output-jsonl .context/ablation/aws_jobs/screening_submitted_jobs.jsonl \
  --region us-east-1
```

Only run submission after reviewing secret injection, queue, job definition, IAM role, S3 permissions, and cost limits.

Collect status after submission:

```bash
python3 scripts/collect_aws_batch_status.py \
  --submitted-jsonl .context/ablation/aws_jobs/screening_submitted_jobs.jsonl \
  --output-jsonl .context/ablation/aws_jobs/screening_status.jsonl \
  --summary-json .context/ablation/aws_jobs/screening_status_summary.json \
  --region us-east-1
```

## Artifact Sync

Every AWS job must preserve:

- manifest
- run log
- node logs
- generated code
- submission
- grader output
- audit output
- cost/token metadata if available

Recommended root:

```bash
ABLATION_ARTIFACT_ROOT=s3://autoresearch-experiments-058264252788-us-east-1/mlevolve-ai-scientist-v2-ablation/<phase>/<run_id>/
```

After jobs complete, sync or point the aggregator at the produced run/node JSONL files, then run:

```bash
python3 scripts/export_ablation_results.py ...
python3 scripts/analyze_ablation_results.py ...
python3 scripts/generate_ablation_audit_report.py ...
```
