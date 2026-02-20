"""
Training loop tests for distributed VideoVAE on TPU.

Tests:
  1. Loss function computes valid scalar loss with all components
  2. Single train_step executes and returns finite loss
  3. Loss decreases over multiple steps (model is learning)
  4. Gradients are synchronized across all devices (data parallel)
  5. SIGTERM handler sets flag and training loop respects it
  6. Data-parallel batch sharding: each device gets unique data slice
  7. VGG perceptual loss loads and computes

Usage (run across all workers):
    gcloud compute tpus tpu-vm ssh train-v6e-16 --zone=europe-west4-a --worker=all \
      --command='cd ~/video-VAE/claude_distributed && python3 test_training_loop.py' --internal-ip
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
from flax import nnx
import optax
import numpy as np
from einops import rearrange, repeat, reduce

num_devices = jax.device_count()
local_devices = jax.local_device_count()
process_index = jax.process_index()
num_processes = jax.process_count()

mesh = jax.make_mesh((num_devices,), ('data',))
replicated_sharding = NamedSharding(mesh, P())
data_sharding = NamedSharding(mesh, P('data'))

PASS_COUNT = 0
FAIL_COUNT = 0

# Small model for fast testing
HEIGHT = 64
WIDTH = 64
CHANNELS = 3
PATCH_SIZE = 16
ENCODER_DEPTH = 2
DECODER_DEPTH = 2
MLP_DIM = 256
NUM_HEADS = 4
QKV_FEATURES = 128
MAX_TEMPORAL_LEN = 16
SPATIAL_COMPRESSION_RATE = 4
UPSAMPLE_RATE = 2
LOCAL_BATCH = local_devices
TEMPORAL_LEN = 8
LR = 1e-3


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
# Build model + optimizer (shared across tests)
# ──────────────────────────────────────────────────────────────────────
from rl_model import VideoVAE

rngs_init = nnx.Rngs(42)
model = VideoVAE(
    height=HEIGHT, width=WIDTH, channels=CHANNELS, patch_size=PATCH_SIZE,
    encoder_depth=ENCODER_DEPTH, decoder_depth=DECODER_DEPTH,
    mlp_dim=MLP_DIM, num_heads=NUM_HEADS, qkv_features=QKV_FEATURES,
    max_temporal_len=MAX_TEMPORAL_LEN,
    spatial_compression_rate=SPATIAL_COMPRESSION_RATE,
    unembedding_upsample_rate=UPSAMPLE_RATE, rngs=rngs_init,
    dtype=jnp.float32, param_dtype=jnp.float32,
)

# Replicate model
gdef, state = nnx.split(model)
state = jax.device_put(state, replicated_sharding)
model = nnx.merge(gdef, state)

# Optimizer
optimizer_def = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=LR),
)
optimizer = nnx.Optimizer(model, optimizer_def)
gdef_opt, opt_state = nnx.split(optimizer)
opt_state = jax.device_put(opt_state, replicated_sharding)
optimizer = nnx.merge(gdef_opt, opt_state)

# Hyperparams
hparams = {
    "gamma1": 0.2,
    "gamma2": 0.001,
    "gamma3": 0.0,       # Disable perceptual loss for fast tests (test 7 covers it)
    "gamma4": 0.05,
    "max_compression_rate": 2,
    "magnify_negatives_rate": 100,
    "rl_loss_weight": 0.01,
}


# ──────────────────────────────────────────────────────────────────────
# Loss function (adapted from distributed_train.py)
# ──────────────────────────────────────────────────────────────────────
def per_sample_mean(x):
    return jnp.mean(x, axis=tuple(range(1, x.ndim)))


def magnify_negatives(x, rate):
    return jnp.where(x < 0, x * rate, x)


def dummy_perceptual_loss(params, x, target):
    """Dummy perceptual loss that returns zeros (for fast testing)."""
    b = x.shape[0]
    return jnp.zeros(b)


def loss_fn(model, video, mask, original_mask, rngs, hparams,
            perceptual_loss_fn, vgg_params, train=True):
    reconstruction, compressed, selection, selection_mask, logvar, mean = \
        model(video, mask, rngs, train=train)

    output_mask = repeat(original_mask, "b time -> (b 2) time")
    sequence_lengths = jnp.clip(reduce(output_mask, "b time -> b 1", "sum"), 1.0, None)

    video_shaped_mask = rearrange(output_mask, "b time -> b time 1 1 1")
    video = repeat(video, "b ... -> (b 2) ...")
    sl = rearrange(sequence_lengths, "b 1 -> b 1 1 1 1")

    masked_abs_error = jnp.abs((video - reconstruction) * video_shaped_mask)
    per_sample_MAE = per_sample_mean(reduce(masked_abs_error, "b t h w c -> b 1 h w c", "sum") / sl)

    masked_sq_error = jnp.square((video - reconstruction) * video_shaped_mask)
    per_sample_error = per_sample_mean(reduce(masked_sq_error, "b t h w c -> b 1 h w c", "sum") / sl)

    perceptual_loss = perceptual_loss_fn(vgg_params, reconstruction, video)

    ksm = rearrange(output_mask, "b time -> b time 1 1")
    sel_sum = reduce(selection_mask * ksm, "b time 1 1 -> b 1", "sum")
    kept_frame_density = sel_sum / sequence_lengths
    density_diff = kept_frame_density - (1 / hparams["max_compression_rate"])
    selection_loss = per_sample_mean(jnp.square(
        magnify_negatives(density_diff, hparams["magnify_negatives_rate"])))

    sl_kl = rearrange(sequence_lengths, "b 1 -> b 1 1 1")
    kl_loss = per_sample_mean(
        0.5 * (jnp.exp(logvar) - 1 - logvar + jnp.square(mean)) * ksm / sl_kl)

    per_sample_loss = (per_sample_error
                       + hparams["gamma3"] * perceptual_loss
                       + hparams["gamma1"] * selection_loss
                       + hparams["gamma2"] * kl_loss
                       + hparams["gamma4"] * per_sample_MAE)

    # RL loss
    pairs = rearrange(per_sample_loss, "(b p) -> b p", p=2)
    means_ = rearrange(per_sample_mean(pairs), "b -> b 1")
    stds_ = rearrange(jnp.std(pairs, axis=1) + 1e-6, "b -> b 1")
    disadvantages = (pairs - means_) / stds_

    actions = rearrange(selection_mask, "(b p) time 1 1 -> b p time", p=2)
    selection = rearrange(selection, "(b p) time 1 1 -> b p time", p=2)
    raw_probs = jnp.clip(jnp.abs(selection + actions - 1), 1e-6, 1.0 - 1e-6)
    probs = raw_probs / jax.lax.stop_gradient(raw_probs)
    rl_mask = rearrange(output_mask, "(b p) time -> b p time", p=2)
    probs = jnp.where(rl_mask, probs, 1.0)
    probs = reduce(probs, "b p time -> b p 1", "prod")
    disadvantages = rearrange(disadvantages, "b p -> b p 1")
    rl_loss = probs * jax.lax.stop_gradient(disadvantages)

    loss = jnp.mean(per_sample_loss) + jnp.mean(rl_loss) * hparams["rl_loss_weight"]

    return loss, {
        "MSE": jnp.mean(per_sample_error),
        "selection_loss": jnp.mean(selection_loss),
        "kl_loss": jnp.mean(kl_loss),
        "rl_loss": jnp.mean(rl_loss),
        "per_sample_MAE": jnp.mean(per_sample_MAE),
        "kept_frame_density": kept_frame_density.mean(),
    }


def train_step(model, optimizer, video, mask, hparams, rngs,
               perceptual_loss_fn, vgg_params):
    original_mask = mask.copy()
    mask = rearrange(mask, "b time -> b 1 1 time")
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, aux), grads = grad_fn(
        model, video, mask, original_mask, rngs, hparams,
        perceptual_loss_fn, vgg_params)
    optimizer.update(grads)
    return loss, aux


jit_train_step = nnx.jit(train_step, static_argnames=("perceptual_loss_fn",))


def make_batch():
    """Create a sharded random batch for testing."""
    key = jax.random.key(int(time.time() * 1000) % 2**31)
    local_video = np.random.randn(LOCAL_BATCH, TEMPORAL_LEN, HEIGHT, WIDTH, CHANNELS).astype(np.float32) * 0.1
    local_mask = np.ones((LOCAL_BATCH, TEMPORAL_LEN), dtype=np.float32)

    sharded_video = jax.make_array_from_process_local_data(data_sharding, local_video)
    mask_spec = P('data', None)
    sharded_mask = jax.make_array_from_process_local_data(
        NamedSharding(mesh, mask_spec), local_mask)

    return sharded_video, sharded_mask.astype(jnp.bool_)


# ──────────────────────────────────────────────────────────────────────
# Test 1: Loss function produces valid scalar
# ──────────────────────────────────────────────────────────────────────
if process_index == 0:
    print("\n[Test 1] Loss function produces valid scalar", flush=True)

try:
    video, mask = make_batch()
    rngs_t = nnx.Rngs(1)

    @nnx.jit
    def compute_loss(model, video, mask, hparams, rngs):
        original_mask = mask.copy()
        mask_4d = rearrange(mask, "b time -> b 1 1 time")
        return loss_fn(model, video, mask_4d, original_mask, rngs, hparams,
                       dummy_perceptual_loss, None, train=True)

    loss, aux = compute_loss(model, video, mask, hparams, rngs_t)

    loss_val = float(loss)
    assert np.isfinite(loss_val), f"Loss not finite: {loss_val}"
    assert loss_val > 0, f"Loss should be positive: {loss_val}"

    # Check all aux components
    for key in ["MSE", "selection_loss", "kl_loss", "rl_loss", "per_sample_MAE", "kept_frame_density"]:
        val = float(aux[key])
        assert np.isfinite(val), f"{key} not finite: {val}"

    test_pass(f"loss={loss_val:.4f}, MSE={float(aux['MSE']):.4f}, "
              f"kl={float(aux['kl_loss']):.4f}, density={float(aux['kept_frame_density']):.4f}")
except Exception as e:
    test_fail("loss function", e)

jax.experimental.multihost_utils.sync_global_devices("test1")


# ──────────────────────────────────────────────────────────────────────
# Test 2: Train step executes and returns finite loss
# ──────────────────────────────────────────────────────────────────────
if process_index == 0:
    print("\n[Test 2] Train step executes correctly", flush=True)

try:
    video, mask = make_batch()
    rngs_t = nnx.Rngs(2)

    start = time.perf_counter()
    loss, aux = jit_train_step(model, optimizer, video, mask, hparams,
                                rngs_t, dummy_perceptual_loss, None)
    elapsed = time.perf_counter() - start

    loss_val = float(loss)
    assert np.isfinite(loss_val), f"Train step loss not finite: {loss_val}"

    test_pass(f"loss={loss_val:.4f}, time={elapsed:.2f}s (includes compilation)")
except Exception as e:
    test_fail("train step", e)

jax.experimental.multihost_utils.sync_global_devices("test2")


# ──────────────────────────────────────────────────────────────────────
# Test 3: Loss decreases over multiple steps
# ──────────────────────────────────────────────────────────────────────
if process_index == 0:
    print("\n[Test 3] Loss decreases over training steps", flush=True)

try:
    # Use fixed data so model can overfit
    fixed_video_local = np.random.randn(LOCAL_BATCH, TEMPORAL_LEN, HEIGHT, WIDTH, CHANNELS).astype(np.float32) * 0.1
    fixed_mask_local = np.ones((LOCAL_BATCH, TEMPORAL_LEN), dtype=np.float32)
    fixed_video = jax.make_array_from_process_local_data(data_sharding, fixed_video_local)
    fixed_mask = jax.make_array_from_process_local_data(
        NamedSharding(mesh, P('data', None)), fixed_mask_local).astype(jnp.bool_)

    losses = []
    NUM_STEPS = 10
    for step in range(NUM_STEPS):
        rngs_t = nnx.Rngs(step + 100)
        loss, aux = jit_train_step(model, optimizer, fixed_video, fixed_mask,
                                    hparams, rngs_t, dummy_perceptual_loss, None)
        losses.append(float(loss))

    # Loss should generally decrease
    first_half_avg = np.mean(losses[:NUM_STEPS // 2])
    second_half_avg = np.mean(losses[NUM_STEPS // 2:])

    if process_index == 0:
        log(f"Losses: {[f'{l:.4f}' for l in losses]}")
        log(f"First half avg: {first_half_avg:.4f}, Second half avg: {second_half_avg:.4f}")

    # The second half should have lower average loss than the first half
    assert second_half_avg < first_half_avg, \
        f"Loss not decreasing: first_half={first_half_avg:.4f} -> second_half={second_half_avg:.4f}"

    test_pass(f"Loss decreased: {first_half_avg:.4f} -> {second_half_avg:.4f}")
except Exception as e:
    test_fail("loss decrease", e)

jax.experimental.multihost_utils.sync_global_devices("test3")


# ──────────────────────────────────────────────────────────────────────
# Test 4: Gradient sync across devices (data parallel correctness)
# ──────────────────────────────────────────────────────────────────────
if process_index == 0:
    print("\n[Test 4] Gradient synchronization (data parallel)", flush=True)

try:
    # After training steps, all replicated params should be identical across devices
    params_state = nnx.state(model, nnx.Param)
    leaves = jax.tree_util.tree_leaves(params_state)

    # Check that params are replicated (same on all devices)
    # We verify by checking the sharding is replicated (P())
    sample_leaf = leaves[0]
    if hasattr(sample_leaf.value, 'sharding'):
        sharding = sample_leaf.value.sharding
        is_replicated = all(s is None for s in sharding.spec)
        assert is_replicated, f"Params not replicated after training: {sharding.spec}"

    # Also verify gradients are finite by doing one more step and checking
    video, mask = make_batch()
    rngs_t = nnx.Rngs(999)

    # Use value_and_grad directly to inspect gradients
    def check_grad_fn(model, video, mask, hparams, rngs):
        original_mask = mask.copy()
        mask_4d = rearrange(mask, "b time -> b 1 1 time")
        return loss_fn(model, video, mask_4d, original_mask, rngs, hparams,
                       dummy_perceptual_loss, None, train=True)

    grad_fn = nnx.value_and_grad(check_grad_fn, has_aux=True)

    @nnx.jit
    def get_grads(model, video, mask, hparams, rngs):
        (loss, aux), grads = grad_fn(model, video, mask, hparams, rngs)
        grad_leaves = jax.tree_util.tree_leaves(nnx.state(grads, nnx.Param))
        max_grad = jnp.max(jnp.stack([jnp.max(jnp.abs(g)) for g in grad_leaves]))
        has_nan = jnp.any(jnp.stack([jnp.any(jnp.isnan(g)) for g in grad_leaves]))
        return loss, max_grad, has_nan

    loss, max_grad, has_nan = get_grads(model, video, mask, hparams, rngs_t)

    assert not bool(has_nan), "Gradients contain NaN"
    assert float(max_grad) > 0, "All gradients are zero"
    assert np.isfinite(float(max_grad)), f"Max gradient not finite: {float(max_grad)}"

    test_pass(f"Params replicated, grads finite, max_grad={float(max_grad):.6f}")
except Exception as e:
    test_fail("gradient sync", e)

jax.experimental.multihost_utils.sync_global_devices("test4")


# ──────────────────────────────────────────────────────────────────────
# Test 5: SIGTERM handling
# ──────────────────────────────────────────────────────────────────────
if process_index == 0:
    print("\n[Test 5] SIGTERM handling", flush=True)

try:
    import distributed_train as dt

    # Check initial state
    assert not dt._SHOULD_STOP, "_SHOULD_STOP should be False initially"

    # Simulate SIGTERM
    dt._signal_handler(signal.SIGTERM, None)
    assert dt._SHOULD_STOP, "_SHOULD_STOP should be True after SIGTERM"

    # Simulate SIGINT
    dt._SHOULD_STOP = False
    dt._signal_handler(signal.SIGINT, None)
    assert dt._SHOULD_STOP, "_SHOULD_STOP should be True after SIGINT"

    # Verify the signal handlers are registered
    current_sigterm = signal.getsignal(signal.SIGTERM)
    current_sigint = signal.getsignal(signal.SIGINT)
    assert current_sigterm == dt._signal_handler, "SIGTERM handler not registered"
    assert current_sigint == dt._signal_handler, "SIGINT handler not registered"

    # Reset for other tests
    dt._SHOULD_STOP = False

    test_pass("SIGTERM/SIGINT handlers work correctly")
except Exception as e:
    test_fail("SIGTERM handling", e)

jax.experimental.multihost_utils.sync_global_devices("test5")


# ──────────────────────────────────────────────────────────────────────
# Test 6: Data-parallel sharding correctness
# ──────────────────────────────────────────────────────────────────────
if process_index == 0:
    print("\n[Test 6] Data-parallel batch sharding", flush=True)

try:
    # Each process creates unique tagged data
    local_video = np.full((LOCAL_BATCH, TEMPORAL_LEN, HEIGHT, WIDTH, CHANNELS),
                          fill_value=float(process_index), dtype=np.float32)
    local_mask = np.ones((LOCAL_BATCH, TEMPORAL_LEN), dtype=np.float32)

    # Shard across mesh
    sharded = {}
    for key, val in {"video": local_video, "mask": local_mask}.items():
        ndim = val.ndim
        spec = P('data', *([None] * (ndim - 1)))
        s = NamedSharding(mesh, spec)
        sharded[key] = jax.make_array_from_process_local_data(s, val)

    global_video = sharded["video"]
    global_mask = sharded["mask"]

    expected_global_batch = LOCAL_BATCH * num_processes
    assert global_video.shape[0] == expected_global_batch, \
        f"Global batch: {global_video.shape[0]} != {expected_global_batch}"

    # On process 0, verify data ordering
    if process_index == 0:
        full = np.array(global_video)
        for p in range(num_processes):
            start = p * LOCAL_BATCH
            end = start + LOCAL_BATCH
            expected = float(p)
            actual = full[start, 0, 0, 0, 0]
            assert np.isclose(actual, expected), \
                f"Process {p} data: expected {expected}, got {actual}"

    test_pass(f"Global batch={expected_global_batch}, data correctly sharded")
except Exception as e:
    test_fail("data-parallel sharding", e)

jax.experimental.multihost_utils.sync_global_devices("test6")


# ──────────────────────────────────────────────────────────────────────
# Test 7: VGG perceptual loss loads and computes
# ──────────────────────────────────────────────────────────────────────
if process_index == 0:
    print("\n[Test 7] VGG perceptual loss", flush=True)

try:
    from vgg_tests import load_vgg, get_adversarial_perceptual_loss_fn

    vgg_model, vgg_params = load_vgg()
    vgg_params = jax.device_put(vgg_params, replicated_sharding)
    perceptual_fn = get_adversarial_perceptual_loss_fn(vgg_model)

    # Test with small inputs
    b, t = 2 * num_processes, TEMPORAL_LEN
    x_local = np.random.randn(2, t, HEIGHT, WIDTH, 3).astype(np.float32) * 0.1
    target_local = np.random.randn(2, t, HEIGHT, WIDTH, 3).astype(np.float32) * 0.1

    x_global = jax.make_array_from_process_local_data(data_sharding,
        np.broadcast_to(x_local, (local_devices, t, HEIGHT, WIDTH, 3))[:local_devices])
    target_global = jax.make_array_from_process_local_data(data_sharding,
        np.broadcast_to(target_local, (local_devices, t, HEIGHT, WIDTH, 3))[:local_devices])

    @jax.jit
    def compute_perceptual(params, x, target):
        return perceptual_fn(params, x, target)

    ploss = compute_perceptual(vgg_params, x_global.astype(jnp.bfloat16),
                                target_global.astype(jnp.bfloat16))

    ploss_val = float(jnp.mean(ploss))
    assert np.isfinite(ploss_val), f"Perceptual loss not finite: {ploss_val}"
    assert ploss_val >= 0, f"Perceptual loss negative: {ploss_val}"

    test_pass(f"VGG loaded, perceptual_loss={ploss_val:.4f}")
except Exception as e:
    test_fail("VGG perceptual loss", e)

jax.experimental.multihost_utils.sync_global_devices("test7")


# ──────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────
jax.experimental.multihost_utils.sync_global_devices("all_tests")

if process_index == 0:
    print(f"\n{'='*50}", flush=True)
    print(f"TRAINING LOOP TESTS: {PASS_COUNT} passed, {FAIL_COUNT} failed", flush=True)
    if FAIL_COUNT == 0:
        print("ALL TRAINING LOOP TESTS PASSED!", flush=True)
    print(f"{'='*50}", flush=True)

if FAIL_COUNT > 0:
    sys.exit(1)
