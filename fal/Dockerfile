# FROM python@sha256:0dab67c838514eef83e6c9d2c0e53e960fc94237635e8996d08caeec98937abc
FROM nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV HF_HUB_CACHE=/data/hf_cache
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV HF_XET_HIGH_PERFORMANCE=1
ENV HF_XET_CHUNK_CACHE_SIZE_BYTES=1000000000000
ENV HF_XET_NUM_CONCURRENT_RANGE_GETS=32

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt update -y && \
    apt-get install -y curl git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /repo && \
    git clone https://github.com/microsoft/VibeVoice.git /repo/VibeVoice

ENV PATH=/root/.local/bin:$PATH
RUN uv python install 3.10 && echo $PATH && uvx python -V

# solve torch first - this is expensive
RUN uv venv /app && \
    . /app/bin/activate && \
    uv pip install torch && \
    uv pip install flash-attn --no-build-isolation

# then install app requirements
RUN . /app/bin/activate && \
    uv pip install huggingface_hub hf_transfer /repo/VibeVoice && \
    uv pip list

# then tweak app env expectations
RUN . /app/bin/activate && \
    ln -s /app/bin/python /bin/python && \
    ln -s /app/bin/python /bin/python3 && \
    ln -s /app/bin/python /bin/python3.10 && \
    which -a python python3 python3.10 && \
    python -V && \
    python3 -V && \
    python3.10 -V && \
    PYTHONPATH=/repo/VibeVoice python3.10 -c 'import vibevoice; print(vibevoice); print(dir(vibevoice))'
