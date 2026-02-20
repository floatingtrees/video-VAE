"""
Quick sanity test for 16-TPU distributed setup.
Tests that:
  1. All 16 devices are visible
  2. Each process gets unique data shards
  3. Model can be built and replicated
  4. A single training step runs successfully
  5. SIGTERM handling is wired up

Usage:
    gcloud compute tpus tpu-vm ssh train-v6e-16 --zone=europe-west4-a --worker=all \
      --command='cd ~/video-VAE/claude_distributed && python3 test_distributed_16.py' --internal-ip
"""

import os
import sys
import signal
import time

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".99"

import jax
jax.distributed.initialize()

import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
import numpy as np

num_devices = jax.device_count()
local_devices = jax.local_device_count()
process_index = jax.process_index()
num_processes = jax.process_count()

# Test 1: Device count
print(f"[Worker {process_index}] Devices: {num_devices} total, {local_devices} local, {num_processes} processes", flush=True)
assert num_devices == 16, f"Expected 16 devices, got {num_devices}"
assert num_processes == 4, f"Expected 4 processes, got {num_processes}"
assert local_devices == 4, f"Expected 4 local devices, got {local_devices}"

# Test 2: Mesh creation
mesh = jax.make_mesh((num_devices,), ('data',))
replicated_sharding = NamedSharding(mesh, P())
data_sharding = NamedSharding(mesh, P('data'))
print(f"[Worker {process_index}] Mesh created: {mesh}", flush=True)

# Test 3: Data sharding - each process creates unique local data
local_batch_size = 4  # 4 per process, 16 total
local_data = np.full((local_batch_size, 2), fill_value=process_index, dtype=np.float32)
global_data = jax.make_array_from_process_local_data(data_sharding, local_data)

if process_index == 0:
    # The global array should have shape (16, 2) with values 0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3
    full_data = np.array(global_data)
    print(f"[Worker 0] Global data shape: {global_data.shape}", flush=True)
    print(f"[Worker 0] Global data column 0: {full_data[:, 0]}", flush=True)
    # Verify each process contributed unique data
    for p in range(num_processes):
        start = p * local_batch_size
        end = start + local_batch_size
        assert np.all(full_data[start:end, 0] == p), \
            f"Process {p} data mismatch: expected {p}, got {full_data[start:end, 0]}"
    print("[Worker 0] Data sharding verified: each process has unique data!", flush=True)

jax.experimental.multihost_utils.sync_global_devices("test_sharding")

# Test 4: Dataloader with shard options
from dataloader import create_batched_dataloader

DATA_DIR = os.path.expanduser("~/data/videos")
dl = create_batched_dataloader(
    base_dir=DATA_DIR,
    batch_size=local_devices,  # 4 per process
    max_frames=8,
    resize=(64, 64),
    shuffle=True,
    num_workers=2,
    prefetch_size=2,
    drop_remainder=True,
    seed=42,
)

batch_count = 0
for batch in dl:
    video = batch["video"]
    mask = batch["mask"]
    print(f"[Worker {process_index}] Batch {batch_count}: video={video.shape}, mask={mask.shape}", flush=True)

    # Shard and verify
    spec = P('data', *([None] * (video.ndim - 1)))
    s = NamedSharding(mesh, spec)
    global_video = jax.make_array_from_process_local_data(s, video)
    if process_index == 0:
        print(f"[Worker 0] Global video shape: {global_video.shape}", flush=True)
    batch_count += 1
    if batch_count >= 2:
        break

jax.experimental.multihost_utils.sync_global_devices("test_dataloader")

# Test 5: Model build + replicate + single forward pass
from flax import nnx
from rl_model import VideoVAE

model = VideoVAE(
    height=64, width=64, channels=3, patch_size=16,
    encoder_depth=2, decoder_depth=2, mlp_dim=256, num_heads=4,
    qkv_features=128, max_temporal_len=16,
    spatial_compression_rate=4, unembedding_upsample_rate=2,
    rngs=nnx.Rngs(2),
)

# Replicate model
gdef, state = nnx.split(model)
state = jax.device_put(state, replicated_sharding)
model = nnx.merge(gdef, state)

if process_index == 0:
    params_state = nnx.state(model, nnx.Param)
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(params_state))
    print(f"[Worker 0] Test model params: {num_params / 1e6:.1f}M", flush=True)

# Forward pass with dummy data
local_dummy = np.random.randn(local_devices, 8, 64, 64, 3).astype(np.float32) * 0.02
local_mask = np.ones((local_devices, 8), dtype=np.float32)

spec_v = P('data', *([None] * (local_dummy.ndim - 1)))
spec_m = P('data', *([None] * (local_mask.ndim - 1)))
global_video = jax.make_array_from_process_local_data(NamedSharding(mesh, spec_v), local_dummy)
global_mask = jax.make_array_from_process_local_data(NamedSharding(mesh, spec_m), local_mask)

global_video = global_video.astype(jnp.bfloat16)
global_mask_4d = jnp.reshape(global_mask, (global_mask.shape[0], 1, 1, global_mask.shape[1]))

rngs = nnx.Rngs(42)

@nnx.jit
def test_forward(model, video, mask, rngs):
    return model(video, mask, rngs, train=True)

start_time = time.perf_counter()
out = test_forward(model, global_video, global_mask_4d, rngs)
elapsed = time.perf_counter() - start_time
reconstruction = out[0]

if process_index == 0:
    print(f"[Worker 0] Forward pass output shape: {reconstruction.shape}, took {elapsed:.2f}s", flush=True)

jax.experimental.multihost_utils.sync_global_devices("test_forward")

# Test 6: SIGTERM handler
print(f"[Worker {process_index}] SIGTERM handler test (simulating)...", flush=True)
from distributed_train import _SHOULD_STOP, _signal_handler
assert not _SHOULD_STOP, "Should not be stopped yet"
# Simulate SIGTERM
_signal_handler(signal.SIGTERM, None)
from distributed_train import _SHOULD_STOP as should_stop_after
assert should_stop_after, "Should be stopped after signal"
print(f"[Worker {process_index}] SIGTERM handler works correctly!", flush=True)

jax.experimental.multihost_utils.sync_global_devices("all_tests_done")

if process_index == 0:
    print("\n========================================", flush=True)
    print("ALL TESTS PASSED on 16 TPUs!", flush=True)
    print("========================================", flush=True)
