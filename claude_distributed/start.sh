#!/bin/bash
# Setup script for all TPU workers.
# Run on all workers via:
#   gcloud compute tpus tpu-vm ssh train-v6e-16 --zone=europe-west4-a --worker=all \
#     --command='cd ~/video-VAE/claude_distributed && bash start.sh' --internal-ip
set -e

echo "=== Setting up worker $(hostname) ==="

# System packages
sudo apt-get update -qq
sudo apt-get install -y -qq python3 python3-pip ffmpeg > /dev/null 2>&1

# gcsfuse for data access
export GCSFUSE_REPO=gcsfuse-$(lsb_release -c -s)
echo "deb [signed-by=/usr/share/keyrings/cloud.google.asc] https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list > /dev/null
curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo tee /usr/share/keyrings/cloud.google.asc > /dev/null
sudo apt-get update -qq
sudo apt-get install -y -qq gcsfuse > /dev/null 2>&1

# Mount training data
mkdir -p ~/data
mountpoint -q ~/data || gcsfuse --implicit-dirs tpus-487818-training-data ~/data
echo "Data mounted: $(ls ~/data/videos/ 2>/dev/null | wc -l) video dirs"

# JAX + TPU
pip install -U "jax[tpu]" 2>&1 | tail -1

# ML dependencies
pip install flax optax orbax-checkpoint einops beartype jaxtyping flaxmodels 2>&1 | tail -1

# Data + video
pip install grain opencv-python-headless 2>&1 | tail -1

# Logging
pip install wandb 2>&1 | tail -1

# Wandb API key
export WANDB_API_KEY=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

echo "=== Worker $(hostname) setup complete ==="
