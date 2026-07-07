# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04 AS deps

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DEFAULT_TIMEOUT=300
ENV PIP_RETRIES=20
ENV MLEBENCH_DATASET_DIR=/mnt/mle-bench
ENV ABLATION_ARTIFACT_ROOT=s3://autoresearch-experiments-058264252788-us-east-1/mlevolve-ai-scientist-v2-ablation

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    curl \
    docker.io \
    git \
    jq \
    software-properties-common \
    rsync \
    unzip \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/pip \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3 \
    && PIP_NO_CACHE_DIR=false python3 -m pip install --upgrade --resume-retries 20 pip setuptools wheel

WORKDIR /opt/mlevolve

COPY requirements_base.txt requirements_ml.txt requirements_domain.txt ./
COPY .context/external/AI-Scientist-v2-ablation/requirements.txt /tmp/ai-scientist-v2-requirements.txt
COPY .context/external/mle-bench /opt/mle-bench

# Install broad runtime dependencies first. Some optional packages may be
# platform-specific; keep the image build log as the authoritative failure
# surface if a pin needs adjustment for the selected AWS base image.
RUN --mount=type=cache,target=/root/.cache/pip \
    tr -d '\r' < requirements_base.txt \
        | sed 's/^pillow==12\.0\.0$/pillow==11.3.0/' \
        > /tmp/requirements_base_aws.txt \
    && PIP_NO_CACHE_DIR=false python3 -m pip install --ignore-installed --resume-retries 20 -r /tmp/requirements_base_aws.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    pip_retry() { \
        attempt=1; \
        until PIP_NO_CACHE_DIR=false python3 -m pip install --resume-retries 20 "$@"; do \
            if [ "$attempt" -ge 6 ]; then return 1; fi; \
            sleep $((attempt * 15)); \
            attempt=$((attempt + 1)); \
        done; \
    } \
    && tr -d '\r' < requirements_ml.txt > /tmp/requirements_ml_aws.txt \
    && grep -E '^autogluon(\.|==)' /tmp/requirements_ml_aws.txt > /tmp/requirements_ml_autogluon_aws.txt \
    && grep -E '^(nvidia-|torch==|torchvision==|torchaudio==|triton==)' /tmp/requirements_ml_aws.txt > /tmp/requirements_ml_torch_aws.txt \
    && grep -Ev '^(autogluon(\.|==)|nvidia-|torch==|torchvision==|torchaudio==|triton==)' /tmp/requirements_ml_aws.txt > /tmp/requirements_ml_rest_aws.txt \
    && pip_retry -r /tmp/requirements_ml_torch_aws.txt \
    && pip_retry -r /tmp/requirements_ml_rest_aws.txt \
    && pip_retry --no-deps -r /tmp/requirements_ml_autogluon_aws.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    PIP_NO_CACHE_DIR=false python3 -m pip install --resume-retries 20 cmake ninja pybind11 Cython

FROM deps AS runtime

ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"

# Domain package dependency ranges conflict with the pinned ML stack in this
# worker image, so install only the requested top-level packages here.
# The retry wrapper below still performs: pip install -e /opt/mle-bench
RUN --mount=type=cache,target=/root/.cache/pip \
    pip_retry() { \
        attempt=1; \
        until PIP_NO_CACHE_DIR=false python3 -m pip install --resume-retries 20 "$@"; do \
            if [ "$attempt" -ge 6 ]; then return 1; fi; \
            sleep $((attempt * 15)); \
            attempt=$((attempt + 1)); \
        done; \
    } \
    && tr -d '\r' < requirements_domain.txt \
        | sed '/^umap==0\.1\.1$/d' \
        | sed '/^torch-dct==0\.1\.6$/d' \
        | sed '/^torchsort$/d' \
        | sed '/^torch_discounted_cumsum==1\.1\.0$/d' \
        | sed '/^seqeval==1\.2\.2$/d' \
        > /tmp/requirements_domain_aws.txt \
    && pip_retry --no-build-isolation --no-deps -r /tmp/requirements_domain_aws.txt \
    && pip_retry --no-build-isolation --no-deps --use-deprecated=legacy-resolver seqeval==1.2.2 \
    && pip_retry -r /tmp/ai-scientist-v2-requirements.txt \
    && pip_retry kaggle \
    && pip_retry -e /opt/mle-bench

COPY . /opt/mlevolve

RUN chmod +x /opt/mlevolve/docker/aws-ablation-worker/entrypoint.sh

ENTRYPOINT ["/opt/mlevolve/docker/aws-ablation-worker/entrypoint.sh"]
CMD ["python3", "scripts/check_ablation_runtime_readiness.py", "--environment-label", "aws-ablation-worker"]
