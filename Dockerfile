# ── RunPod Serverless Dockerfile for Hallo2 Talking-Head ─────────────────
# LEAN image — pretrained models stored on RunPod Network Volume (/runpod-volume).
# Code + deps only (~5-6 GB). Models (~12 GB) downloaded to volume once.
#
# Volume layout:
#   /runpod-volume/hallo2_models/  — all pretrained weights (hallo2, SD1.5, wav2vec, etc.)
#   /runpod-volume/hallo2_output/  — saved result videos
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Ubuntu 22.04 ships Python 3.10 — required by Hallo2
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev gcc g++ \
        ffmpeg git wget \
        libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 && \
    rm -rf /var/lib/apt/lists/*

# Clone Hallo2 source code (no model weights)
RUN git clone --depth 1 https://github.com/fudan-generative-vision/hallo2.git /app/hallo2

# PyTorch for CUDA 11.8 (matching Hallo2's tested config)
RUN pip3 install --no-cache-dir \
    torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
    --index-url https://download.pytorch.org/whl/cu118

# Hallo2 dependencies
RUN cd /app/hallo2 && pip3 install --no-cache-dir -r requirements.txt

# RunPod SDK + HuggingFace CLI for one-time model download
RUN pip3 install --no-cache-dir runpod huggingface_hub[cli]

COPY handler_runpod.py /app/handler_runpod.py

CMD ["python3", "/app/handler_runpod.py"]
