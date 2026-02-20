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
# The dataloader needs video files to exist, even if they're tiny dummy files.
# Each worker gets its own unique shard via grain's ShardOptions, so the
# dataloader just needs files to exist at the expected paths.
echo "--- Creating dummy video data on workers without real data ---"
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --worker=all \
  --command='
    if [ ! -d ~/data/videos/videos0 ]; then
      echo "No real data found, creating dummy videos..."
      mkdir -p ~/data/videos/videos0/videos0
      mkdir -p ~/data/videos/videos1/videos1
      # Create dummy mp4 files using ffmpeg (tiny 8-frame 64x64 videos)
      for i in $(seq 1 200); do
        ffmpeg -y -f lavfi -i "color=c=blue:s=64x64:d=0.27:r=30" \
          -c:v libx264 -pix_fmt yuv420p -loglevel quiet \
          ~/data/videos/videos0/videos0/dummy_${i}.mp4 2>/dev/null
      done
      for i in $(seq 1 200); do
        ffmpeg -y -f lavfi -i "color=c=red:s=64x64:d=0.27:r=30" \
          -c:v libx264 -pix_fmt yuv420p -loglevel quiet \
          ~/data/videos/videos1/videos1/dummy_${i}.mp4 2>/dev/null
      done
      echo "Created 400 dummy videos"
    else
      echo "Real data already exists, skipping dummy creation."
    fi
  ' --internal-ip

# Step 4: Verify setup on all workers
echo "--- Verifying setup on all workers ---"
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --worker=all \
  --command='
    echo "=== Worker $(hostname) ==="
    python3 -c "import jax; print(f\"JAX {jax.__version__}, devices: {jax.local_device_count()}\")"
    echo "Videos: $(find ~/data/videos -name \"*.mp4\" 2>/dev/null | wc -l)"
    echo "Repo: $(ls ~/video-VAE/claude_distributed/distributed_train.py 2>/dev/null && echo OK || echo MISSING)"
  ' --internal-ip

echo ""
echo "=== Setup complete! ==="
echo "To run distributed training:"
echo "  gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --worker=all \\"
echo "    --command='cd ~/video-VAE/claude_distributed && python3 distributed_train.py' --internal-ip"
