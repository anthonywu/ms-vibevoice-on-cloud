FROM python@sha256:0dab67c838514eef83e6c9d2c0e53e960fc94237635e8996d08caeec98937abc
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV HF_HUB_CACHE=/data/hf_cache
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV HF_XET_HIGH_PERFORMANCE=1
ENV HF_XET_CHUNK_CACHE_SIZE_BYTES=1000000000000
ENV HF_XET_NUM_CONCURRENT_RANGE_GETS=32

RUN mkdir -p /repo && \
    git clone https://github.com/microsoft/VibeVoice.git /repo/VibeVoice

RUN python -m pip install -U pip && \
    python -m pip install huggingface_hub hf_transfer /repo/VibeVoice && \
    python -m pip list && python -V && \
    python -c 'import vibevoice; print(vibevoice); print(dir(vibevoice)' || true
