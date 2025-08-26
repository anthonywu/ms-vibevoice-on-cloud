# VibeVoice Inference Implementations

Hosting https://github.com/microsoft/VibeVoice at various GPU providers for fun.

# Development

## Fal Serverless

- Local venv python must match python version in container
  - `uv venv --python 3.11`
  - `. .venv/bin/activate`
  - `uv pip install fal`
- Modify `Dockerfile` and `app,py` as needed
- Build and Run: `fal run app.py::VibeVoiceApp` and watch the logs
- Deploy: `fal deploy app.py::VibeVoiceApp`

Hosted Fal App: https://fal.ai/models/anthonywu/ms-vibevoice/

**Sad Notes**

- Fal's default `machine_type = 'S'` fails the download the model weights - leaving the HF cache in a corrupted state
- Python `3.10` is the latest runtime that can resolve `numba` -> `llvmlite` depedency chain
- `cudnn-devel` base image is required, `cudnn-runtime` does not have required `nvcc`
- Cannot use a standard python base image, does not have expected Nvidia configs

## Replicate

TBD - if anyone else doesn't get to it first.
