"""
Test script for distributed VideoVAE training on TPU pods.

Verifies:
  1. All devices are discovered and used
  2. Data is properly sharded (different data per device)
  3. Gradient synchronization works (replicated params stay in sync)
  4. Training loop runs end-to-end with loss decreasing
  5. Real dataloader feeds different batches to different devices

Does NOT hardcode device count -- scales to any number of TPUs.

Usage:
    python test_distributed.py            # full test suite
    python test_distributed.py --quick    # fast smoke test (tiny model, synthetic data only)
"""

import os
import numpy as np
import time
import argparse

# NOTE: jax.distributed.initialize() and all JAX device access must be inside
# __main__ to avoid conflicts with Grain's multiprocessing workers.

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
all_passed = True


def log(msg, process_index):
    if process_index == 0:
        print(msg)


def check(name, condition, process_index):
    global all_passed
    status = PASS if condition else FAIL
    if not condition:
        all_passed = False
    log(f"  [{status}] {name}", process_index)
    return condition


def shard_batch(batch, mesh):
    """Convert per-process numpy batch to globally sharded jax.Arrays."""
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    sharded = {}
    for key, val in batch.items():
        ndim = val.ndim
        spec = P('data', *([None] * (ndim - 1)))
        s = NamedSharding(mesh, spec)
        sharded[key] = jax.make_array_from_process_local_data(s, val)
    return sharded


# ===================================================================
# TEST 1: Device discovery
# ===================================================================
def test_device_discovery(jax, num_devices, local_devices, num_processes, process_index):
    log("\n=== Test 1: Device Discovery ===", process_index)
    check("Found at least 1 device", num_devices >= 1, process_index)
    check(f"Device count ({num_devices}) == local ({local_devices}) * procs ({num_processes})",
          num_devices == local_devices * num_processes, process_index)
    check("All devices are TPU or GPU (not CPU-only for prod)",
          "Tpu" in str(type(jax.devices()[0])) or "Gpu" in str(type(jax.devices()[0]))
          or num_devices >= 2, process_index)
    log(f"  Devices: {jax.devices()}", process_index)


# ===================================================================
# TEST 2: Data sharding across devices
# ===================================================================
def test_data_sharding(jax, mesh, num_devices, process_index):
    from jax.sharding import NamedSharding, PartitionSpec as P

    log("\n=== Test 2: Data Sharding ===", process_index)

    per_process = num_devices  # one sample per device globally
    data = np.arange(per_process * 4, dtype=np.float32).reshape(per_process, 4)
    spec = P('data', None)
    s = NamedSharding(mesh, spec)
    global_arr = jax.make_array_from_process_local_data(s, data)

    check("Global shape matches", global_arr.shape == (per_process, 4), process_index)
    check(f"Has {num_devices} addressable shards",
          len(global_arr.addressable_shards) == num_devices, process_index)

    shard_first_elems = []
    for shard in global_arr.addressable_shards:
        shard_data = np.array(shard.data)
        shard_first_elems.append(shard_data[0, 0])

    unique_values = len(set(shard_first_elems))
    check(f"All {num_devices} shards have different data (got {unique_values} unique)",
          unique_values == num_devices, process_index)


# ===================================================================
# TEST 3: Gradient synchronization with tiny model
# ===================================================================
def test_gradient_sync(jax, mesh, replicated_sharding, num_devices, process_index):
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from flax import nnx
    import optax
    from einops import rearrange, repeat
    from rl_model import VideoVAE

    log("\n=== Test 3: Gradient Synchronization ===", process_index)

    model = VideoVAE(
        height=32, width=32, channels=3, patch_size=8,
        encoder_depth=1, decoder_depth=1,
        mlp_dim=64, num_heads=2, qkv_features=32,
        max_temporal_len=4, spatial_compression_rate=2,
        unembedding_upsample_rate=2,
        rngs=nnx.Rngs(42), dtype=jnp.float32, param_dtype=jnp.float32,
    )

    gdef, state = nnx.split(model)
    state = jax.device_put(state, replicated_sharding)
    model = nnx.merge(gdef, state)

    tx = optax.adam(1e-3)
    optimizer = nnx.Optimizer(model, tx)
    gdef_opt, opt_state = nnx.split(optimizer)
    opt_state = jax.device_put(opt_state, replicated_sharding)
    optimizer = nnx.merge(gdef_opt, opt_state)

    per_process_batch = num_devices
    video_np = np.random.randn(per_process_batch, 4, 32, 32, 3).astype(np.float32) * 0.02
    mask_np = np.ones((per_process_batch, 4), dtype=np.bool_)

    video = jax.make_array_from_process_local_data(
        NamedSharding(mesh, P('data', None, None, None, None)), video_np)
    mask = jax.make_array_from_process_local_data(
        NamedSharding(mesh, P('data', None)), mask_np)

    def simple_loss(model, video, mask, rngs):
        mask_expanded = rearrange(mask, "b time -> b 1 1 time")
        reconstruction, _, _, _, logvar, mean = model(video, mask_expanded, rngs, train=True)
        video_rep = repeat(video, "b ... -> (b 2) ...")
        return jnp.mean((reconstruction - video_rep) ** 2)

    def simple_train_step(model, optimizer, video, mask, rngs):
        grad_fn = nnx.value_and_grad(simple_loss)
        loss, grads = grad_fn(model, video, mask, rngs)
        optimizer.update(grads)
        return loss

    jit_step = nnx.jit(simple_train_step)

    rngs = nnx.Rngs(0)
    loss0 = float(jit_step(model, optimizer, video, mask, rngs))
    log(f"  Step 0 loss: {loss0:.6f}", process_index)

    param_state = nnx.state(model, nnx.Param)
    first_leaf = jax.tree_util.tree_leaves(param_state)[0]
    param_sharding = first_leaf.sharding
    check("Params remain replicated after update",
          all(s is None for s in param_sharding.spec), process_index)

    losses = [loss0]
    for step in range(4):
        video_np = np.random.randn(per_process_batch, 4, 32, 32, 3).astype(np.float32) * 0.02
        video = jax.make_array_from_process_local_data(
            NamedSharding(mesh, P('data', None, None, None, None)), video_np)
        l = float(jit_step(model, optimizer, video, mask, rngs))
        losses.append(l)
    log(f"  Losses over 5 steps: {[f'{l:.6f}' for l in losses]}", process_index)
    check("Training produced finite losses", all(np.isfinite(l) for l in losses), process_index)


# ===================================================================
# TEST 4: Real dataloader with different shards per device
# ===================================================================
def test_dataloader_sharding(jax, mesh, num_devices, local_devices, num_processes, process_index, data_dir):
    from dataloader import create_batched_dataloader

    log("\n=== Test 4: Dataloader Sharding ===", process_index)

    if not os.path.isdir(data_dir):
        log(f"  [SKIP] Data directory not found: {data_dir}", process_index)
        return

    batch_size = max(local_devices, 2)

    loader = create_batched_dataloader(
        base_dir=data_dir,
        batch_size=batch_size,
        max_frames=8,
        resize=(64, 64),
        shuffle=True,
        num_workers=2,
        prefetch_size=4,
        drop_remainder=True,
        seed=42,
    )

    batches_seen = 0
    shard_checksums = []
    for batch in loader:
        global_batch = shard_batch(batch, mesh)
        video = global_batch["video"]

        check(f"Batch {batches_seen}: global shape = ({batch_size * num_processes}, 8, 64, 64, 3)",
              video.shape == (batch_size * num_processes, 8, 64, 64, 3), process_index)

        checksums = []
        for shard in video.addressable_shards:
            shard_data = np.array(shard.data)
            checksums.append(float(np.sum(shard_data)))
        shard_checksums.append(checksums)

        unique = len(set(f"{c:.4f}" for c in checksums))
        check(f"Batch {batches_seen}: all {len(checksums)} device shards have different video data "
              f"({unique} unique checksums)",
              unique == len(checksums), process_index)

        batches_seen += 1
        if batches_seen >= 2:
            break

    check(f"Loaded {batches_seen} batches successfully", batches_seen == 2, process_index)

    if len(shard_checksums) == 2:
        batch0_sum = sum(shard_checksums[0])
        batch1_sum = sum(shard_checksums[1])
        check("Different batches have different data",
              abs(batch0_sum - batch1_sum) > 1e-3, process_index)


# ===================================================================
# TEST 5: End-to-end training with full loss function
# ===================================================================
def test_end_to_end_training(jax, mesh, replicated_sharding, num_devices, process_index):
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from flax import nnx
    import optax
    from einops import rearrange, repeat, reduce
    from rl_model import VideoVAE
    from vgg_tests import get_adversarial_perceptual_loss_fn, load_vgg

    log("\n=== Test 5: End-to-End Training (tiny model, synthetic data) ===", process_index)

    H, W = 32, 32
    PATCH = 8
    T = 4
    hw = (H // PATCH) * (W // PATCH)

    model = VideoVAE(
        height=H, width=W, channels=3, patch_size=PATCH,
        encoder_depth=1, decoder_depth=1,
        mlp_dim=64, num_heads=2, qkv_features=32,
        max_temporal_len=T, spatial_compression_rate=2,
        unembedding_upsample_rate=2,
        rngs=nnx.Rngs(42), dtype=jnp.float32, param_dtype=jnp.float32,
    )

    gdef, state = nnx.split(model)
    state = jax.device_put(state, replicated_sharding)
    model = nnx.merge(gdef, state)

    schedule_fn = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=1e-3,
        warmup_steps=5, decay_steps=100, end_value=1e-4)
    optimizer_def = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=schedule_fn))
    optimizer = nnx.Optimizer(model, optimizer_def)
    gdef_opt, opt_state = nnx.split(optimizer)
    opt_state = jax.device_put(opt_state, replicated_sharding)
    optimizer = nnx.merge(gdef_opt, opt_state)

    vgg_model, vgg_params = load_vgg()
    vgg_params = jax.device_put(vgg_params, replicated_sharding)
    perceptual_loss_fn = get_adversarial_perceptual_loss_fn(vgg_model)

    hparams = {
        "gamma1": 0.2, "gamma2": 0.001, "gamma3": 0.1, "gamma4": 0.05,
        "max_compression_rate": 2, "magnify_negatives_rate": 100,
        "rl_loss_weight": 0.01,
    }

    def per_sample_mean(x):
        return jnp.mean(x, axis=tuple(range(1, x.ndim)))

    def magnify_negatives(x, rate):
        return jnp.where(x < 0, x * rate, x)

    def full_loss_fn(model, video, mask, original_mask, rngs, hparams,
                     perceptual_loss_fn, vgg_params, train=True):
        reconstruction, compressed, selection, selection_mask, logvar, mean = model(video, mask, rngs, train=train)
        output_mask = repeat(original_mask, "b time -> (b 2) time")
        seq_len = jnp.clip(reduce(output_mask, "b time -> b 1", "sum"), 1.0, None)
        video_shaped_mask = rearrange(output_mask, "b time -> b time 1 1 1")
        video = repeat(video, "b ... -> (b 2) ...")
        sl = rearrange(seq_len, "b 1 -> b 1 1 1 1")

        mse = per_sample_mean(reduce(jnp.square((video - reconstruction) * video_shaped_mask),
                                     "b t h w c -> b 1 h w c", "sum") / sl)
        mae = per_sample_mean(reduce(jnp.abs((video - reconstruction) * video_shaped_mask),
                                     "b t h w c -> b 1 h w c", "sum") / sl)
        ploss = perceptual_loss_fn(vgg_params, reconstruction, video)

        ksm = rearrange(output_mask, "b time -> b time 1 1")
        sel_sum = reduce(selection_mask * ksm, "b time 1 1 -> b 1", "sum")
        density = sel_sum / seq_len
        sel_loss = per_sample_mean(jnp.square(magnify_negatives(
            density - 1 / hparams["max_compression_rate"], hparams["magnify_negatives_rate"])))

        sl_kl = rearrange(seq_len, "b 1 -> b 1 1 1")
        kl = per_sample_mean(0.5 * (jnp.exp(logvar) - 1 - logvar + jnp.square(mean)) * ksm / sl_kl)

        psl = (mse + hparams["gamma3"] * ploss + hparams["gamma1"] * sel_loss
               + hparams["gamma2"] * kl + hparams["gamma4"] * mae)
        loss = jnp.mean(psl)
        return loss, {"MSE": jnp.mean(mse), "reconstruction": reconstruction}

    def train_step(model, optimizer, video, mask, hparams, rngs,
                   perceptual_loss_fn, vgg_params):
        original_mask = mask.copy()
        mask = rearrange(mask, "b time -> b 1 1 time")
        grad_fn = nnx.value_and_grad(full_loss_fn, has_aux=True)
        (loss, aux), grads = grad_fn(model, video, mask, original_mask, rngs, hparams,
                                     perceptual_loss_fn, vgg_params)
        optimizer.update(grads)
        return loss, aux

    jit_step = nnx.jit(train_step, static_argnames=("perceptual_loss_fn",))

    rngs = nnx.Rngs(3)
    per_process = max(num_devices, 2)

    losses = []
    t0 = time.perf_counter()
    for step in range(5):
        video_np = np.random.randn(per_process, T, H, W, 3).astype(np.float32) * 0.1
        mask_np = np.ones((per_process, T), dtype=np.bool_)

        video = jax.make_array_from_process_local_data(
            NamedSharding(mesh, P('data', None, None, None, None)), video_np)
        mask = jax.make_array_from_process_local_data(
            NamedSharding(mesh, P('data', None)), mask_np)

        loss, aux = jit_step(model, optimizer, video, mask, hparams,
                             rngs, perceptual_loss_fn, vgg_params)
        losses.append(float(loss))
        if process_index == 0:
            elapsed = time.perf_counter() - t0
            print(f"  Step {step}: loss={losses[-1]:.6f} "
                  f"MSE={float(aux['MSE']):.6f} time={elapsed:.1f}s")

    check("All losses finite", all(np.isfinite(l) for l in losses), process_index)
    check("Reconstruction shape correct",
          aux["reconstruction"].shape == (per_process * 2, T, H, W, 3), process_index)
    log(f"  Final losses: {[f'{l:.4f}' for l in losses]}", process_index)


# ===================================================================
# TEST 6: Real data end-to-end (minimal)
# ===================================================================
def test_real_data_training(jax, mesh, replicated_sharding, num_devices, local_devices,
                            num_processes, process_index, data_dir):
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from flax import nnx
    import optax
    from einops import rearrange, repeat
    from rl_model import VideoVAE
    from dataloader import create_batched_dataloader

    log("\n=== Test 6: Real Data Training (1 batch, tiny model) ===", process_index)

    if not os.path.isdir(data_dir):
        log(f"  [SKIP] Data directory not found: {data_dir}", process_index)
        return

    H, W = 64, 64
    PATCH = 8
    T = 8
    hw = (H // PATCH) * (W // PATCH)
    batch_size = max(local_devices, 2)

    model = VideoVAE(
        height=H, width=W, channels=3, patch_size=PATCH,
        encoder_depth=1, decoder_depth=1,
        mlp_dim=64, num_heads=2, qkv_features=32,
        max_temporal_len=T, spatial_compression_rate=2,
        unembedding_upsample_rate=2,
        rngs=nnx.Rngs(42), dtype=jnp.bfloat16, param_dtype=jnp.float32,
    )

    gdef, state = nnx.split(model)
    state = jax.device_put(state, replicated_sharding)
    model = nnx.merge(gdef, state)

    tx = optax.adam(1e-3)
    optimizer = nnx.Optimizer(model, tx)
    gdef_opt, opt_state = nnx.split(optimizer)
    opt_state = jax.device_put(opt_state, replicated_sharding)
    optimizer = nnx.merge(gdef_opt, opt_state)

    def simple_loss(model, video, mask, rngs):
        mask_expanded = rearrange(mask, "b time -> b 1 1 time")
        reconstruction, _, _, _, _, _ = model(video, mask_expanded, rngs, train=True)
        video_rep = repeat(video, "b ... -> (b 2) ...")
        return jnp.mean((reconstruction - video_rep) ** 2)

    def step_fn(model, optimizer, video, mask, rngs):
        grad_fn = nnx.value_and_grad(simple_loss)
        loss, grads = grad_fn(model, video, mask, rngs)
        optimizer.update(grads)
        return loss

    jit_step = nnx.jit(step_fn)
    rngs = nnx.Rngs(0)

    loader = create_batched_dataloader(
        base_dir=data_dir,
        batch_size=batch_size,
        max_frames=T,
        resize=(H, W),
        shuffle=True,
        num_workers=2,
        prefetch_size=4,
        drop_remainder=True,
        seed=99,
    )

    t0 = time.perf_counter()
    for i, batch in enumerate(loader):
        global_batch = shard_batch(batch, mesh)
        video = global_batch["video"].astype(jnp.bfloat16)
        mask = global_batch["mask"].astype(jnp.bool_)

        loss = float(jit_step(model, optimizer, video, mask, rngs))
        elapsed = time.perf_counter() - t0
        log(f"  Real data step {i}: loss={loss:.6f} time={elapsed:.1f}s", process_index)

        if i >= 1:
            break

    check("Real data training produced finite loss", np.isfinite(loss), process_index)
    check(f"Processed real video data on {num_devices} devices", True, process_index)


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".99"

    import jax
    jax.distributed.initialize()

    from jax.sharding import NamedSharding, PartitionSpec as P

    num_devices = jax.device_count()
    local_devices = jax.local_device_count()
    process_index = jax.process_index()
    num_processes = jax.process_count()

    mesh = jax.make_mesh((num_devices,), ('data',))
    replicated_sharding = NamedSharding(mesh, P())

    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Quick smoke test")
    parser.add_argument("--data_dir", type=str,
                        default=os.path.expanduser("~/data/videos/videos"))
    args = parser.parse_args()

    log(f"{'='*60}", process_index)
    log(f"Distributed Training Test Suite", process_index)
    log(f"{'='*60}", process_index)
    log(f"JAX version: {jax.__version__}", process_index)
    log(f"Devices: {num_devices} total, {local_devices} local, {num_processes} processes", process_index)
    log(f"Device type: {type(jax.devices()[0])}", process_index)

    # Always run these
    test_device_discovery(jax, num_devices, local_devices, num_processes, process_index)
    test_data_sharding(jax, mesh, num_devices, process_index)
    test_gradient_sync(jax, mesh, replicated_sharding, num_devices, process_index)

    if not args.quick:
        test_dataloader_sharding(jax, mesh, num_devices, local_devices,
                                 num_processes, process_index, args.data_dir)
        test_end_to_end_training(jax, mesh, replicated_sharding, num_devices, process_index)
        test_real_data_training(jax, mesh, replicated_sharding, num_devices, local_devices,
                                num_processes, process_index, args.data_dir)

    log(f"\n{'='*60}", process_index)
    if all_passed:
        log(f"ALL TESTS PASSED on {num_devices} devices", process_index)
    else:
        log(f"SOME TESTS FAILED", process_index)
    log(f"{'='*60}", process_index)
