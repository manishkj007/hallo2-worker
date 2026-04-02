"""RunPod Serverless handler — Hallo2 talking-head generation.

Lean Docker image — all pretrained models on /runpod-volume/hallo2_models/.

Actions:
  - "download_models": One-time download of all pretrained weights to volume
  - "talking_head":    source_image_b64 + audio_b64 → realistic talking video
  - "list_outputs":    List saved videos on volume
  - "get_output":      Retrieve a saved video by name
"""

import os
import sys
import uuid
import time
import base64
import json
import subprocess
import shutil
import tempfile
from pathlib import Path

import runpod

# ── Paths ────────────────────────────────────────────────────────────────
VOLUME = "/runpod-volume"
MODEL_DIR = f"{VOLUME}/hallo2_models"
OUTPUT_DIR = f"{VOLUME}/hallo2_output"
HALLO2_DIR = "/app/hallo2"
TMPDIR = "/dev/shm" if os.path.isdir("/dev/shm") else "/tmp"

print(f"[init] MODEL_DIR={MODEL_DIR} volume={'yes' if os.path.isdir(VOLUME) else 'no'}")


def _models_ready():
    """Check if pretrained models exist on volume."""
    required = [
        f"{MODEL_DIR}/hallo2/net.pth",
        f"{MODEL_DIR}/stable-diffusion-v1-5/unet/diffusion_pytorch_model.safetensors",
        f"{MODEL_DIR}/wav2vec/wav2vec2-base-960h/model.safetensors",
        f"{MODEL_DIR}/motion_module/mm_sd_v15_v2.ckpt",
        f"{MODEL_DIR}/sd-vae-ft-mse/diffusion_pytorch_model.safetensors",
        f"{MODEL_DIR}/face_analysis/models/scrfd_10g_bnkps.onnx",
        f"{MODEL_DIR}/audio_separator/Kim_Vocal_2.onnx",
    ]
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        print(f"[models] missing: {missing}")
    return len(missing) == 0


def handle_download_models(job_input):
    """Download all pretrained models from HuggingFace to the network volume."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("[download] Starting model download from HuggingFace...")
    t0 = time.time()

    # Main Hallo2 weights (includes hallo2/, face_analysis/, audio_separator/,
    # motion_module/, sd-vae-ft-mse/, stable-diffusion-v1-5/, wav2vec/, facelib/, etc.)
    cmd = [
        "huggingface-cli", "download",
        "fudan-generative-ai/hallo2",
        "--local-dir", MODEL_DIR,
        "--local-dir-use-symlinks", "False",
    ]
    print(f"[download] {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    if r.returncode != 0:
        return {"error": f"HuggingFace download failed: {r.stderr[-500:]}"}
    print(f"[download] HF download done in {time.time()-t0:.0f}s")

    # Verify critical files
    if not _models_ready():
        return {"error": "Model download incomplete — some files still missing"}

    # List what we got
    total_size = 0
    file_count = 0
    for root, dirs, files in os.walk(MODEL_DIR):
        for f in files:
            fp = os.path.join(root, f)
            total_size += os.path.getsize(fp)
            file_count += 1

    return {
        "status": "models_downloaded",
        "files": file_count,
        "total_size_gb": round(total_size / 1e9, 2),
        "elapsed_seconds": round(time.time() - t0),
    }


def _write_config(img_path, audio_path, save_path, **kwargs):
    """Generate the inference config YAML with model paths pointing to volume."""
    config_content = f"""source_image: {img_path}
driving_audio: {audio_path}
save_path: {save_path}
cache_path: {TMPDIR}/.hallo2_cache
weight_dtype: fp16

data:
  n_motion_frames: 2
  n_sample_frames: 16
  source_image:
    width: 512
    height: 512
  driving_audio:
    sample_rate: 16000
  export_video:
    fps: 25

inference_steps: {kwargs.get('inference_steps', 40)}
cfg_scale: {kwargs.get('cfg_scale', 3.5)}

use_mask: true
mask_rate: 0.25
use_cut: true

audio_ckpt_dir: {MODEL_DIR}/hallo2
base_model_path: {MODEL_DIR}/stable-diffusion-v1-5
motion_module_path: {MODEL_DIR}/motion_module/mm_sd_v15_v2.ckpt

face_analysis:
  model_path: {MODEL_DIR}/face_analysis

wav2vec:
  model_path: {MODEL_DIR}/wav2vec/wav2vec2-base-960h
  features: all

audio_separator:
  model_path: {MODEL_DIR}/audio_separator/Kim_Vocal_2.onnx

vae:
  model_path: {MODEL_DIR}/sd-vae-ft-mse

face_expand_ratio: {kwargs.get('face_expand_ratio', 1.2)}
pose_weight: {kwargs.get('pose_weight', 1.0)}
face_weight: {kwargs.get('face_weight', 1.0)}
lip_weight: {kwargs.get('lip_weight', 1.5)}

unet_additional_kwargs:
  use_inflated_groupnorm: true
  unet_use_cross_frame_attention: false
  unet_use_temporal_attention: false
  use_motion_module: true
  use_audio_module: true
  motion_module_resolutions:
    - 1
    - 2
    - 4
    - 8
  motion_module_mid_block: true
  motion_module_decoder_only: false
  motion_module_type: Vanilla
  motion_module_kwargs:
    num_attention_heads: 8
    num_transformer_block: 1
    attention_block_types:
      - Temporal_Self
      - Temporal_Self
    temporal_position_encoding: true
    temporal_position_encoding_max_len: 32
    temporal_attention_dim_div: 1
  audio_attention_dim: 768
  stack_enable_blocks_name:
    - up
    - down
    - mid
  stack_enable_blocks_depth: [0,1,2,3]

enable_zero_snr: true

noise_scheduler_kwargs:
  beta_start: 0.00085
  beta_end: 0.012
  beta_schedule: linear
  clip_sample: false
  steps_offset: 1
  prediction_type: v_prediction
  rescale_betas_zero_snr: true
  timestep_spacing: trailing

sampler: DDIM
"""
    config_path = os.path.join(TMPDIR, f"hallo2_config_{uuid.uuid4().hex[:8]}.yaml")
    with open(config_path, "w") as f:
        f.write(config_content)
    return config_path


def handle_talking_head(job_input):
    """Generate a talking-head video from portrait image + audio."""
    t0 = time.time()

    # Validate models
    if not _models_ready():
        return {"error": "Models not downloaded. Run 'download_models' action first."}

    # Decode inputs
    image_b64 = job_input.get("source_image_b64")
    audio_b64 = job_input.get("audio_b64")
    if not image_b64 or not audio_b64:
        return {"error": "Both 'source_image_b64' and 'audio_b64' are required."}

    # Create working directory
    work_id = uuid.uuid4().hex[:8]
    work_dir = os.path.join(TMPDIR, f"hallo2_{work_id}")
    os.makedirs(work_dir, exist_ok=True)
    save_dir = os.path.join(work_dir, "output")
    os.makedirs(save_dir, exist_ok=True)

    try:
        # Write input image (must be square, face 50-70%)
        img_path = os.path.join(work_dir, "input.jpg")
        img_bytes = base64.b64decode(image_b64)
        with open(img_path, "wb") as f:
            f.write(img_bytes)
        print(f"[talking_head] image: {len(img_bytes)} bytes → {img_path}")

        # Write input audio (WAV format)
        audio_path = os.path.join(work_dir, "input.wav")
        audio_bytes = base64.b64decode(audio_b64)
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)
        print(f"[talking_head] audio: {len(audio_bytes)} bytes → {audio_path}")

        # Generate config YAML
        config_path = _write_config(
            img_path, audio_path, save_dir,
            inference_steps=job_input.get("inference_steps", 40),
            cfg_scale=job_input.get("cfg_scale", 3.5),
            pose_weight=job_input.get("pose_weight", 1.0),
            face_weight=job_input.get("face_weight", 1.0),
            lip_weight=job_input.get("lip_weight", 1.5),
            face_expand_ratio=job_input.get("face_expand_ratio", 1.2),
        )

        # Run Hallo2 inference
        cmd = [
            sys.executable,
            "scripts/inference_long.py",
            "--config", config_path,
            "--source_image", img_path,
            "--driving_audio", audio_path,
        ]
        print(f"[talking_head] running: {' '.join(cmd)}")
        r = subprocess.run(
            cmd,
            cwd=HALLO2_DIR,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour max
        )
        print(f"[talking_head] stdout (last 2000): {r.stdout[-2000:]}")
        if r.returncode != 0:
            print(f"[talking_head] stderr: {r.stderr[-2000:]}")
            return {"error": f"Hallo2 inference failed (exit {r.returncode}): {r.stderr[-500:]}"}

        # Find the merged output video
        # inference_long.py saves to: {save_dir}/{image_stem}/merge_video.mp4
        merged_video = os.path.join(save_dir, "input", "merge_video.mp4")
        if not os.path.exists(merged_video):
            # Search for any mp4 in the save directory
            for root, dirs, files in os.walk(save_dir):
                for f in files:
                    if f.endswith(".mp4") and "merge" in f:
                        merged_video = os.path.join(root, f)
                        break

        if not os.path.exists(merged_video):
            return {"error": f"Output video not found. Files: {list(os.walk(save_dir))}"}

        video_size = os.path.getsize(merged_video)
        print(f"[talking_head] output: {merged_video} ({video_size} bytes)")

        # Read and encode result
        with open(merged_video, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode("utf-8")

        elapsed = time.time() - t0

        # Optionally save to volume
        output_name = job_input.get("output_name")
        volume_path = None
        if output_name:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            volume_path = os.path.join(OUTPUT_DIR, f"{output_name}.mp4")
            shutil.copy2(merged_video, volume_path)
            print(f"[talking_head] saved to volume: {volume_path}")

        return {
            "video_b64": video_b64,
            "video_size_bytes": video_size,
            "elapsed_seconds": round(elapsed),
            "volume_path": volume_path,
        }

    finally:
        # Cleanup temp files
        shutil.rmtree(work_dir, ignore_errors=True)
        if os.path.exists(config_path):
            os.remove(config_path)


def handle_list_outputs(job_input):
    """List output files on the network volume."""
    if not os.path.isdir(OUTPUT_DIR):
        return {"files": []}
    files = []
    for f in sorted(os.listdir(OUTPUT_DIR)):
        fp = os.path.join(OUTPUT_DIR, f)
        files.append({"name": f, "size": os.path.getsize(fp)})
    return {"files": files}


def handle_get_output(job_input):
    """Retrieve a specific output file from the volume."""
    name = job_input.get("name")
    if not name:
        return {"error": "'name' is required"}
    fp = os.path.join(OUTPUT_DIR, name)
    if not os.path.exists(fp):
        return {"error": f"File not found: {name}"}
    # Security: prevent path traversal
    if not os.path.realpath(fp).startswith(os.path.realpath(OUTPUT_DIR)):
        return {"error": "Invalid path"}
    with open(fp, "rb") as f:
        data_b64 = base64.b64encode(f.read()).decode("utf-8")
    return {"data_b64": data_b64, "size": os.path.getsize(fp)}


# ── RunPod Handler ───────────────────────────────────────────────────────
def handler(job):
    job_input = job["input"]
    action = job_input.get("action", "talking_head")
    print(f"[handler] action={action}")

    if action == "download_models":
        return handle_download_models(job_input)
    elif action == "talking_head":
        return handle_talking_head(job_input)
    elif action == "list_outputs":
        return handle_list_outputs(job_input)
    elif action == "get_output":
        return handle_get_output(job_input)
    else:
        return {"error": f"Unknown action: {action}"}


if __name__ == "__main__":
    print("[hallo2-worker] Starting RunPod serverless handler...")
    runpod.serverless.start({"handler": handler})
