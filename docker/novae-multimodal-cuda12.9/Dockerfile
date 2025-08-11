FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

ARG NOVAE_VERSION

# command 'ps' for nextflow
RUN apt-get update && apt-get install -y procps && apt-get clean && rm -rf /var/lib/apt/lists/*

# to avoid permission issues with the numba cache
ENV NUMBA_CACHE_DIR=/tmp/numba_cache

RUN pip install --no-cache-dir novae[multimodal,conch]==${NOVAE_VERSION}

ENV PYTHONUNBUFFERED=1
