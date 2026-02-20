"""
Dataloader tests (local CPU mode, no distributed init needed).

Tests:
  1. list_video_files discovers videos in nested layout
  2. load_video returns correct shapes and value ranges
  3. VideoDataSource + LoadVideoTransform pipeline
  4. create_batched_dataloader yields correct batch shapes
  5. Data sharding across simulated devices
  6. Different seeds produce different orderings

Usage:
    JAX_PLATFORMS=cpu JAX_NUM_CPU_DEVICES=4 python3 test_dataloader.py
"""

import os
import sys
import hashlib

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_NUM_CPU_DEVICES", "4")

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P, Mesh
import numpy as np

num_devices = jax.device_count()
print(f"Running with {num_devices} CPU devices", flush=True)

mesh = Mesh(jax.devices(), axis_names=('data',))
data_sharding = NamedSharding(mesh, P('data'))

DATA_DIR = os.path.expanduser("~/data/videos")
PASS_COUNT = 0
FAIL_COUNT = 0


def test_pass(name):
    global PASS_COUNT
    PASS_COUNT += 1
    print(f"  PASS: {name}", flush=True)


def test_fail(name, err):
    global FAIL_COUNT
    FAIL_COUNT += 1
    print(f"  FAIL: {name} - {err}", flush=True)


# ──────────────────────────────────────────────────────────────────────
# Test 1: list_video_files
# ──────────────────────────────────────────────────────────────────────
print("\n[Test 1] list_video_files", flush=True)

from dataloader import list_video_files

try:
    video_paths = list_video_files(DATA_DIR)
    assert len(video_paths) > 0, f"No videos found in {DATA_DIR}"
    sample_check = video_paths[:10]
    for p in sample_check:
        assert os.path.isfile(p), f"Not a file: {p}"
        assert p.endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')), f"Not a video: {p}"
    test_pass(f"Found {len(video_paths)} videos, all valid paths")
except Exception as e:
    test_fail("list_video_files", e)


# ──────────────────────────────────────────────────────────────────────
# Test 2: load_video shapes and value range
# ──────────────────────────────────────────────────────────────────────
print("\n[Test 2] load_video", flush=True)

from dataloader import load_video

try:
    video_paths = list_video_files(DATA_DIR)
    test_path = video_paths[0]
    MAX_FRAMES = 16
    RESIZE = (64, 64)

    video, mask = load_video(test_path, max_frames=MAX_FRAMES, resize=RESIZE, crop_size=128)

    assert video.shape == (MAX_FRAMES, 64, 64, 3), f"Bad video shape: {video.shape}"
    assert mask.shape == (MAX_FRAMES,), f"Bad mask shape: {mask.shape}"
    assert video.min() >= 0.0, f"Video min below 0: {video.min()}"
    assert video.max() <= 1.0, f"Video max above 1: {video.max()}"
    assert set(np.unique(mask)).issubset({0.0, 1.0}), f"Mask values: {np.unique(mask)}"
    assert mask.sum() > 0, "No real frames in mask"

    if mask.sum() < MAX_FRAMES:
        first_pad = int(mask.sum())
        assert np.allclose(video[first_pad:], 0.0), "Padded frames not zero"

    test_pass(f"video={video.shape}, mask={mask.shape}, "
              f"real_frames={int(mask.sum())}, range=[{video.min():.3f}, {video.max():.3f}]")
except Exception as e:
    test_fail("load_video", e)


# ──────────────────────────────────────────────────────────────────────
# Test 3: VideoDataSource + LoadVideoTransform
# ──────────────────────────────────────────────────────────────────────
print("\n[Test 3] VideoDataSource + LoadVideoTransform", flush=True)

from dataloader import VideoDataSource, LoadVideoTransform

try:
    ds = VideoDataSource(DATA_DIR)
    assert len(ds) > 0, "Empty data source"

    path = ds[0]
    assert isinstance(path, str), f"Expected str, got {type(path)}"
    assert os.path.isfile(path), f"Not a file: {path}"

    transform = LoadVideoTransform(max_frames=8, resize=(64, 64), crop_size=128)
    result = transform.map(path)
    assert "video" in result and "mask" in result, f"Missing keys: {result.keys()}"
    assert result["video"].shape == (8, 64, 64, 3), f"Bad shape: {result['video'].shape}"
    assert result["mask"].shape == (8,), f"Bad mask shape: {result['mask'].shape}"

    test_pass(f"DataSource len={len(ds)}, transform output shapes correct")
except Exception as e:
    test_fail("VideoDataSource + LoadVideoTransform", e)


# ──────────────────────────────────────────────────────────────────────
# Test 4: create_batched_dataloader yields correct shapes
# ──────────────────────────────────────────────────────────────────────
print("\n[Test 4] create_batched_dataloader batch shapes", flush=True)

from dataloader import create_batched_dataloader

LOCAL_BATCH = 4
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


# ──────────────────────────────────────────────────────────────────────
# Test 5: Data sharding across devices
# ──────────────────────────────────────────────────────────────────────
print("\n[Test 5] Data sharding across devices", flush=True)

try:
    local_video = np.arange(num_devices * 2).reshape(num_devices, 2).astype(np.float32)
    global_video = jax.device_put(local_video, data_sharding)

    assert global_video.shape == (num_devices, 2), f"Shape mismatch: {global_video.shape}"

    # Each shard should be on a different device
    shards = global_video.addressable_shards
    devices_used = {s.device for s in shards}
    assert len(devices_used) == num_devices, \
        f"Expected {num_devices} devices, got {len(devices_used)}"

    # Verify data is intact
    full = np.array(global_video)
    assert np.allclose(full, local_video), "Data corrupted during sharding"

    test_pass(f"Sharded across {len(devices_used)} devices, data intact")
except Exception as e:
    test_fail("data sharding", e)


# ──────────────────────────────────────────────────────────────────────
# Test 6: Different seeds produce different orderings
# ──────────────────────────────────────────────────────────────────────
print("\n[Test 6] Different seeds produce different data", flush=True)

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

    are_different = not np.allclose(b1, b2, atol=1e-3)
    assert are_different, "Same seed=100 and seed=200 produced identical batches"

    test_pass("Different seeds produce different orderings")
except Exception as e:
    test_fail("different seeds", e)


# ──────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────
print(f"\n{'='*50}", flush=True)
print(f"DATALOADER TESTS: {PASS_COUNT} passed, {FAIL_COUNT} failed", flush=True)
if FAIL_COUNT == 0:
    print("ALL DATALOADER TESTS PASSED!", flush=True)
print(f"{'='*50}", flush=True)

if FAIL_COUNT > 0:
    sys.exit(1)
