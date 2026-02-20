"""
Dataloader tests for distributed TPU training.

Tests:
  1. list_video_files discovers videos in nested layout
  2. load_video returns correct shapes and value ranges
  3. VideoDataSource + LoadVideoTransform pipeline
  4. create_batched_dataloader yields correct batch shapes
  5. Each process gets unique shards (no data duplication)
  6. shard_batch / make_array_from_process_local_data works correctly
  7. Multiple epochs produce different orderings when shuffle=True

Usage (run across all workers):
    gcloud compute tpus tpu-vm ssh train-v6e-16 --zone=europe-west4-a --worker=all \
      --command='cd ~/video-VAE/claude_distributed && python3 test_dataloader.py' --internal-ip
"""

import os
import sys
import time
import hashlib

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

mesh = jax.make_mesh((num_devices,), ('data',))
data_sharding = NamedSharding(mesh, P('data'))

DATA_DIR = os.path.expanduser("~/data/videos")
PASS_COUNT = 0
FAIL_COUNT = 0


def log(msg):
    if process_index == 0:
        print(f"  {msg}", flush=True)


def test_pass(name):
    global PASS_COUNT
    PASS_COUNT += 1
    if process_index == 0:
        print(f"  PASS: {name}", flush=True)


def test_fail(name, err):
    global FAIL_COUNT
    FAIL_COUNT += 1
    print(f"  FAIL [{process_index}]: {name} - {err}", flush=True)


# ──────────────────────────────────────────────────────────────────────
# Test 1: list_video_files
# ──────────────────────────────────────────────────────────────────────
if process_index == 0:
    print("\n[Test 1] list_video_files", flush=True)

from dataloader import list_video_files

try:
    video_paths = list_video_files(DATA_DIR)
    assert len(video_paths) > 0, f"No videos found in {DATA_DIR}"
    # All paths should be valid files
    sample_check = video_paths[:10]
    for p in sample_check:
        assert os.path.isfile(p), f"Not a file: {p}"
        assert p.endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')), f"Not a video: {p}"
    test_pass(f"Found {len(video_paths)} videos, all valid paths")
except Exception as e:
    test_fail("list_video_files", e)

jax.experimental.multihost_utils.sync_global_devices("test1")


# ──────────────────────────────────────────────────────────────────────
# Test 2: load_video shapes and value range
# ──────────────────────────────────────────────────────────────────────
if process_index == 0:
    print("\n[Test 2] load_video", flush=True)

from dataloader import load_video

try:
    video_paths = list_video_files(DATA_DIR)
    test_path = video_paths[0]
    MAX_FRAMES = 16
    RESIZE = (64, 64)

    video, mask = load_video(test_path, max_frames=MAX_FRAMES, resize=RESIZE, crop_size=128)

    # Shape checks
    assert video.shape == (MAX_FRAMES, 64, 64, 3), f"Bad video shape: {video.shape}"
    assert mask.shape == (MAX_FRAMES,), f"Bad mask shape: {mask.shape}"

    # Value range
    assert video.min() >= 0.0, f"Video min below 0: {video.min()}"
    assert video.max() <= 1.0, f"Video max above 1: {video.max()}"

    # Mask is 0 or 1
    assert set(np.unique(mask)).issubset({0.0, 1.0}), f"Mask has unexpected values: {np.unique(mask)}"

    # At least some real frames
    assert mask.sum() > 0, "No real frames in mask"

    # Padded frames should be zeros
    if mask.sum() < MAX_FRAMES:
        first_pad = int(mask.sum())
        assert np.allclose(video[first_pad:], 0.0), "Padded frames not zero"

    test_pass(f"video={video.shape}, mask={mask.shape}, "
              f"real_frames={int(mask.sum())}, range=[{video.min():.3f}, {video.max():.3f}]")
except Exception as e:
    test_fail("load_video", e)

jax.experimental.multihost_utils.sync_global_devices("test2")


# ──────────────────────────────────────────────────────────────────────
# Test 3: VideoDataSource + LoadVideoTransform
# ──────────────────────────────────────────────────────────────────────
if process_index == 0:
    print("\n[Test 3] VideoDataSource + LoadVideoTransform", flush=True)

from dataloader import VideoDataSource, LoadVideoTransform

try:
    ds = VideoDataSource(DATA_DIR)
    assert len(ds) > 0, "Empty data source"

    # First item is a path string
    path = ds[0]
    assert isinstance(path, str), f"Expected str, got {type(path)}"
    assert os.path.isfile(path), f"Not a file: {path}"

    # Transform produces correct dict
    transform = LoadVideoTransform(max_frames=8, resize=(64, 64), crop_size=128)
    result = transform.map(path)
    assert "video" in result and "mask" in result, f"Missing keys: {result.keys()}"
    assert result["video"].shape == (8, 64, 64, 3), f"Bad shape: {result['video'].shape}"
    assert result["mask"].shape == (8,), f"Bad mask shape: {result['mask'].shape}"

    test_pass(f"DataSource len={len(ds)}, transform output shapes correct")
except Exception as e:
    test_fail("VideoDataSource + LoadVideoTransform", e)

jax.experimental.multihost_utils.sync_global_devices("test3")


# ──────────────────────────────────────────────────────────────────────
# Test 4: create_batched_dataloader yields correct shapes
# ──────────────────────────────────────────────────────────────────────
if process_index == 0:
    print("\n[Test 4] create_batched_dataloader batch shapes", flush=True)

from dataloader import create_batched_dataloader

LOCAL_BATCH = local_devices  # 4 per process
MAX_F = 8
RESIZE_HW = (64, 64)

try:
    dl = create_batched_dataloader(
        base_dir=DATA_DIR,
        batch_size=LOCAL_BATCH,
        max_frames=MAX_F,
        resize=RESIZE_HW,
        shuffle=True,
        num_workers=2,
        prefetch_size=2,
        drop_remainder=True,
        seed=42,
    )

    batch = next(iter(dl))
    video = batch["video"]
    mask = batch["mask"]

    assert video.shape == (LOCAL_BATCH, MAX_F, RESIZE_HW[0], RESIZE_HW[1], 3), \
        f"Bad batch video shape: {video.shape}"
    assert mask.shape == (LOCAL_BATCH, MAX_F), \
        f"Bad batch mask shape: {mask.shape}"
    assert video.dtype == np.float32, f"Expected float32, got {video.dtype}"

    test_pass(f"video={video.shape}, mask={mask.shape}")
except Exception as e:
    test_fail("create_batched_dataloader shapes", e)

jax.experimental.multihost_utils.sync_global_devices("test4")


# ──────────────────────────────────────────────────────────────────────
# Test 5: Each process gets UNIQUE shards (no duplication)
# ──────────────────────────────────────────────────────────────────────
if process_index == 0:
    print("\n[Test 5] Unique data shards per process", flush=True)

try:
    dl = create_batched_dataloader(
        base_dir=DATA_DIR,
        batch_size=LOCAL_BATCH,
        max_frames=MAX_F,
        resize=RESIZE_HW,
        shuffle=True,
        num_workers=2,
        prefetch_size=2,
        drop_remainder=True,
        seed=42,
    )

    batch = next(iter(dl))
    video = batch["video"]

    # Hash the local video data to compare across processes
    local_hash = hashlib.md5(video.tobytes()).hexdigest()

    # Gather hashes - use a small numeric encoding instead
    # Each process creates a local array with hash bytes
    hash_bytes = np.frombuffer(bytes.fromhex(local_hash), dtype=np.uint8).astype(np.float32)
    # Pad to fixed size (16 bytes for md5)
    local_hash_arr = hash_bytes.reshape(1, -1)  # (1, 16)

    spec = P('data', None)
    s = NamedSharding(mesh, spec)
    # We need a consistent size: each process contributes 1 row
    # But make_array_from_process_local_data needs local_devices rows per process
    local_hash_broadcast = np.broadcast_to(local_hash_arr, (local_devices, 16))
    global_hashes = jax.make_array_from_process_local_data(s, local_hash_broadcast)

    if process_index == 0:
        all_hashes = np.array(global_hashes)
        # Check one hash per process (take first row from each group of local_devices)
        unique_hashes = set()
        for p in range(num_processes):
            row = tuple(all_hashes[p * local_devices].tolist())
            unique_hashes.add(row)
        assert len(unique_hashes) == num_processes, \
            f"Expected {num_processes} unique data shards, got {len(unique_hashes)}"
        test_pass(f"All {num_processes} processes have unique data")
    else:
        test_pass("Contributed unique data shard")

except Exception as e:
    test_fail("unique shards", e)

jax.experimental.multihost_utils.sync_global_devices("test5")


# ──────────────────────────────────────────────────────────────────────
# Test 6: shard_batch (make_array_from_process_local_data) correctness
# ──────────────────────────────────────────────────────────────────────
if process_index == 0:
    print("\n[Test 6] shard_batch correctness", flush=True)

try:
    # Each process creates local data tagged with its process_index
    local_video = np.full((LOCAL_BATCH, MAX_F, 4, 4, 3), fill_value=float(process_index),
                          dtype=np.float32)
    local_mask = np.full((LOCAL_BATCH, MAX_F), fill_value=float(process_index),
                         dtype=np.float32)
    local_batch = {"video": local_video, "mask": local_mask}

    # Shard batch (same logic as distributed_train.py)
    sharded = {}
    for key, val in local_batch.items():
        ndim = val.ndim
        spec = P('data', *([None] * (ndim - 1)))
        s = NamedSharding(mesh, spec)
        sharded[key] = jax.make_array_from_process_local_data(s, val)

    global_video = sharded["video"]
    global_mask = sharded["mask"]

    expected_global_batch = LOCAL_BATCH * num_processes
    assert global_video.shape[0] == expected_global_batch, \
        f"Global batch size mismatch: {global_video.shape[0]} != {expected_global_batch}"
    assert global_video.shape == (expected_global_batch, MAX_F, 4, 4, 3), \
        f"Global video shape: {global_video.shape}"
    assert global_mask.shape == (expected_global_batch, MAX_F), \
        f"Global mask shape: {global_mask.shape}"

    # Verify data integrity on process 0
    if process_index == 0:
        full_video = np.array(global_video)
        for p in range(num_processes):
            start = p * LOCAL_BATCH
            end = start + LOCAL_BATCH
            expected_val = float(p)
            actual_vals = full_video[start:end, 0, 0, 0, 0]
            assert np.allclose(actual_vals, expected_val), \
                f"Process {p} data mismatch: expected {expected_val}, got {actual_vals}"

    test_pass(f"Global shapes correct, data integrity verified across {num_processes} processes")
except Exception as e:
    test_fail("shard_batch", e)

jax.experimental.multihost_utils.sync_global_devices("test6")


# ──────────────────────────────────────────────────────────────────────
# Test 7: Different seeds produce different orderings
# ──────────────────────────────────────────────────────────────────────
if process_index == 0:
    print("\n[Test 7] Different seeds produce different data", flush=True)

try:
    dl1 = create_batched_dataloader(
        base_dir=DATA_DIR, batch_size=LOCAL_BATCH, max_frames=MAX_F,
        resize=RESIZE_HW, shuffle=True, num_workers=2, prefetch_size=2,
        drop_remainder=True, seed=100,
    )
    dl2 = create_batched_dataloader(
        base_dir=DATA_DIR, batch_size=LOCAL_BATCH, max_frames=MAX_F,
        resize=RESIZE_HW, shuffle=True, num_workers=2, prefetch_size=2,
        drop_remainder=True, seed=200,
    )

    b1 = next(iter(dl1))["video"]
    b2 = next(iter(dl2))["video"]

    # Different seeds should yield different data (with high probability)
    are_different = not np.allclose(b1, b2, atol=1e-3)
    assert are_different, "Same seed=100 and seed=200 produced identical batches"

    test_pass("Different seeds produce different orderings")
except Exception as e:
    test_fail("different seeds", e)

jax.experimental.multihost_utils.sync_global_devices("test7")


# ──────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────
jax.experimental.multihost_utils.sync_global_devices("all_tests")

if process_index == 0:
    print(f"\n{'='*50}", flush=True)
    print(f"DATALOADER TESTS: {PASS_COUNT} passed, {FAIL_COUNT} failed", flush=True)
    if FAIL_COUNT == 0:
        print("ALL DATALOADER TESTS PASSED!", flush=True)
    print(f"{'='*50}", flush=True)

if FAIL_COUNT > 0:
    sys.exit(1)
