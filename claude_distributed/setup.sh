#!/bin/bash
# Setup script for distributed training on TPU v6e-16 (4 workers)
# Run this from worker 0. It will set up all workers via gcloud SSH.
set -e

TPU_NAME="train-v6e-16"
ZONE="europe-west4-a"
REPO_URL="https://github.com/floatingtrees/video-VAE.git"
# If you need authenticated access, set GH_TOKEN env var before running:
#   export GH_TOKEN=your_token
# Then the URL will be: https://${GH_TOKEN}@github.com/floatingtrees/video-VAE.git
if [ -n "${GH_TOKEN}" ]; then
  REPO_URL="https://floatingtrees:${GH_TOKEN}@github.com/floatingtrees/video-VAE.git"
fi
BRANCH="claude_commits"

# Packages needed for training (matching worker 0's working environment)
PACKAGES="jax[tpu] flax==0.10.7 optax orbax-checkpoint grain einops beartype jaxtyping flaxmodels opencv-python-headless h5py scipy numpy pillow rich tqdm sentry-sdk pydantic simplejson"

echo "=== Setting up ALL workers ==="

# Step 1: Install packages on all workers (including worker 0 for idempotency)
echo "--- Installing Python packages on all workers ---"
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --worker=all \
  --command="pip install ${PACKAGES} 2>&1 | tail -3" --internal-ip

# Step 2: Clone/update repo on all workers
echo "--- Cloning/updating repo on all workers ---"
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --worker=all \
  --command="
    if [ -d ~/video-VAE ]; then
      cd ~/video-VAE && git fetch origin && git checkout ${BRANCH} && git pull origin ${BRANCH}
    else
      git clone -b ${BRANCH} ${REPO_URL} ~/video-VAE
    fi
  " --internal-ip

# Step 3: Create dummy video data on workers that don't have real data
# Uses OpenCV (installed via opencv-python-headless) since ffmpeg may not be available.
echo "--- Creating dummy video data on workers without real data ---"
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --worker=all \
  --command='python3 -c "
import os, numpy as np, cv2

vdir = os.path.expanduser(\"~/data/videos/videos0/videos0\")
if os.path.isdir(vdir):
    count = len([f for f in os.listdir(vdir) if f.endswith(\".mp4\") and not f.startswith(\"dummy\")])
    if count > 50:
        print(f\"Real data exists ({count} videos), skipping.\")
        exit(0)

print(\"Creating dummy videos with OpenCV...\")
for shard in [0, 1]:
    d = os.path.expanduser(f\"~/data/videos/videos{shard}/videos{shard}\")
    os.makedirs(d, exist_ok=True)
    for i in range(200):
        path = os.path.join(d, f\"dummy_{i}.mp4\")
        if os.path.exists(path) and os.path.getsize(path) > 100:
            continue
        fourcc = cv2.VideoWriter_fourcc(*\"mp4v\")
        writer = cv2.VideoWriter(path, fourcc, 30.0, (64, 64))
        for _ in range(8):
            frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            writer.write(frame)
        writer.release()
    print(f\"Created 200 dummy videos in videos{shard}\")
print(\"Done.\")
"' --internal-ip

# Step 4: Verify setup on all workers
echo "--- Verifying setup on all workers ---"
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --worker=all \
  --command='
    echo "=== Worker $(hostname) ==="
    python3 -c "
import os
videos = []
for d in [\"videos0/videos0\", \"videos1/videos1\"]:
    p = os.path.expanduser(f\"~/data/videos/{d}\")
    if os.path.isdir(p):
        videos.extend([f for f in os.listdir(p) if f.endswith(\".mp4\")])
print(f\"Videos: {len(videos)}\")
"
    echo "Repo: $(ls ~/video-VAE/claude_distributed/distributed_train.py 2>/dev/null && echo OK || echo MISSING)"
  ' --internal-ip

echo ""
echo "=== Setup complete! ==="
echo "To run distributed training:"
echo "  gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --worker=all \\"
echo "    --command='cd ~/video-VAE/claude_distributed && python3 distributed_train.py' --internal-ip"
