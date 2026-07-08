#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  scripts/build_push_aws_worker_image.sh --repository <ecr-repository-name> [--tag <tag>] [--region <region>] [--account <account-id>] [--push]

Builds the AWS ablation worker image. Push is opt-in.

Required:
  --repository  Existing ECR repository name, for example mlevolve-ablation-worker

Options:
  --tag         Image tag. Default: aws-ablation-worker-v1
  --region      AWS region. Default: aws configure get region, then us-east-1
  --account     AWS account id. Default: aws sts get-caller-identity
  --push        Login to ECR and push the image
EOF
}

repository=""
tag="aws-ablation-worker-v1"
region="$(aws configure get region 2>/dev/null || true)"
region="${region:-us-east-1}"
account=""
push="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repository)
      repository="${2:?--repository requires a value}"
      shift 2
      ;;
    --tag)
      tag="${2:?--tag requires a value}"
      shift 2
      ;;
    --region)
      region="${2:?--region requires a value}"
      shift 2
      ;;
    --account)
      account="${2:?--account requires a value}"
      shift 2
      ;;
    --push)
      push="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$repository" ]]; then
  usage
  exit 2
fi

if [[ -z "$account" ]]; then
  account="$(aws sts get-caller-identity --query Account --output text)"
fi

image_uri="${account}.dkr.ecr.${region}.amazonaws.com/${repository}:${tag}"

docker build \
  -f docker/aws-ablation-worker/Dockerfile \
  -t "${image_uri}" \
  .

if [[ "$push" == "true" ]]; then
  aws ecr get-login-password --region "${region}" \
    | docker login --username AWS --password-stdin "${account}.dkr.ecr.${region}.amazonaws.com"
  docker push "${image_uri}"
fi

echo "${image_uri}"
