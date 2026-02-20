"""
Dataloader tests (local CPU mode, no distributed init needed).

Usage:
    JAX_PLATFORMS=cpu JAX_NUM_CPU_DEVICES=4 python3 test_dataloader.py
"""

import os
import sys

if __name__ == '__main__':
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_NUM_CPU_DEVICES", "4")

    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P, Mesh
    import numpy as np
    import hashlib

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

    # ── Test 1: list_video_files ──
    print("\n[Test 1] list_video_files", flush=True)
    from dataloader import list_video_files
    try:
        video_paths = list_video_files(DATA_DIR)
        assert len(video_paths) > 0, f"No videos found in {DATA_DIR}"
        for p in video_paths[:10]:
            assert os.path.isfile(p), f"Not a file: {p}"
        test_pass(f"Found {len(video_paths)} videos")
    except Exception as e:
        test_fail("list_video_files", e)

    # ── Test 2: load_video ──
    print("\n[Test 2] load_video", flush=True)
    from dataloader import load_video
    try:
        video, mask = load_video(video_paths[0], max_frames=16, resize=(64, 64), crop_size=128)
        assert video.shape == (16, 64, 64, 3), f"Bad shape: {video.shape}"
        assert mask.shape == (16,), f"Bad mask: {mask.shape}"
        assert 0.0 <= video.min() and video.max() <= 1.0, "Out of range"
        assert mask.sum() > 0, "No real frames"
        test_pass(f"video={video.shape}, real_frames={int(mask.sum())}")
    except Exception as e:
        test_fail("load_video", e)

    # ── Test 3: VideoDataSource + LoadVideoTransform ──
    print("\n[Test 3] VideoDataSource + LoadVideoTransform", flush=True)
    from dataloader import VideoDataSource, LoadVideoTransform
    try:
        ds = VideoDataSource(DATA_DIR)
        assert len(ds) > 0
        transform = LoadVideoTransform(max_frames=8, resize=(64, 64), crop_size=128)
        result = transform.map(ds[0])
        assert result["video"].shape == (8, 64, 64, 3)
        assert result["mask"].shape == (8,)
        test_pass(f"DataSource len={len(ds)}, transform OK")
    except Exception as e:
        test_fail("VideoDataSource + LoadVideoTransform", e)

    # ── Test 4: create_batched_dataloader ──
    print("\n[Test 4] create_batched_dataloader batch shapes", flush=True)
    from dataloader import create_batched_dataloader
    try:
        dl = create_batched_dataloader(
            base_dir=DATA_DIR, batch_size=4, max_frames=8, resize=(64, 64),
            shuffle=True, num_workers=0, prefetch_size=2, drop_remainder=True, seed=42,
        )
        batch = next(iter(dl))
        assert batch["video"].shape == (4, 8, 64, 64, 3), f"Bad shape: {batch['video'].shape}"
        assert batch["mask"].shape == (4, 8), f"Bad mask: {batch['mask'].shape}"
        test_pass(f"video={batch['video'].shape}, mask={batch['mask'].shape}")
    except Exception as e:
        test_fail("create_batched_dataloader", e)

    # ── Test 5: Data sharding ──
    print("\n[Test 5] Data sharding across devices", flush=True)
    try:
        local_video = np.arange(num_devices * 2).reshape(num_devices, 2).astype(np.float32)
        global_video = jax.device_put(jnp.array(local_video), data_sharding)
        assert global_video.shape == (num_devices, 2)
        shards = global_video.addressable_shards
        assert len({s.device for s in shards}) == num_devices
        assert np.allclose(np.array(global_video), local_video)
        test_pass(f"Sharded across {num_devices} devices, data intact")
    except Exception as e:
        test_fail("data sharding", e)

    # ── Test 6: Different seeds ──
    print("\n[Test 6] Different seeds produce different data", flush=True)
    try:
        dl1 = create_batched_dataloader(
            base_dir=DATA_DIR, batch_size=4, max_frames=8, resize=(64, 64),
            shuffle=True, num_workers=0, prefetch_size=2, drop_remainder=True, seed=100,
        )
        dl2 = create_batched_dataloader(
            base_dir=DATA_DIR, batch_size=4, max_frames=8, resize=(64, 64),
            shuffle=True, num_workers=0, prefetch_size=2, drop_remainder=True, seed=200,
        )
        b1 = next(iter(dl1))["video"]
        b2 = next(iter(dl2))["video"]
        assert not np.allclose(b1, b2, atol=1e-3), "Same data with different seeds"
        test_pass("Different seeds produce different orderings")
    except Exception as e:
        test_fail("different seeds", e)

    # ── Summary ──
    print(f"\n{'='*50}", flush=True)
    print(f"DATALOADER TESTS: {PASS_COUNT} passed, {FAIL_COUNT} failed", flush=True)
    if FAIL_COUNT == 0:
        print("ALL DATALOADER TESTS PASSED!", flush=True)
    print(f"{'='*50}", flush=True)
    if FAIL_COUNT > 0:
        sys.exit(1)
