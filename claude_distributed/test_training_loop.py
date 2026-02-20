"""
Training loop tests (local CPU mode).

Tests:
  1. Loss function produces valid scalar with all components
  2. Train step executes and returns finite loss
  3. Loss decreases over multiple steps
  4. Gradients are finite and nonzero
  5. SIGTERM handler sets flag correctly
  6. Data-parallel batch sharding correctness
  7. VGG perceptual loss loads and computes

Usage:
    JAX_PLATFORMS=cpu JAX_NUM_CPU_DEVICES=4 python3 test_training_loop.py
"""

import os
import sys
import signal
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_NUM_CPU_DEVICES", "4")

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P, Mesh
from flax import nnx
import optax
import numpy as np
from einops import rearrange, repeat, reduce

num_devices = jax.device_count()
print(f"Running with {num_devices} CPU devices", flush=True)

mesh = Mesh(jax.devices(), axis_names=('data',))
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
BATCH = num_devices
TEMPORAL_LEN = 8
LR = 1e-3


def test_pass(name):
    global PASS_COUNT
    PASS_COUNT += 1
    print(f"  PASS: {name}", flush=True)


def test_fail(name, err):
    global FAIL_COUNT
    FAIL_COUNT += 1
    print(f"  FAIL: {name} - {err}", flush=True)


# ──────────────────────────────────────────────────────────────────────
# Build model + optimizer
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

gdef, state = nnx.split(model)
state = jax.device_put(state, replicated_sharding)
model = nnx.merge(gdef, state)

optimizer_def = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=LR),
)
optimizer = nnx.Optimizer(model, optimizer_def)
gdef_opt, opt_state = nnx.split(optimizer)
opt_state = jax.device_put(opt_state, replicated_sharding)
optimizer = nnx.merge(gdef_opt, opt_state)

hparams = {
    "gamma1": 0.2,
    "gamma2": 0.001,
    "gamma3": 0.0,       # Disable perceptual for fast tests (test 7 covers it)
    "gamma4": 0.05,
    "max_compression_rate": 2,
    "magnify_negatives_rate": 100,
    "rl_loss_weight": 0.01,
}


# ──────────────────────────────────────────────────────────────────────
# Loss function (from distributed_train.py)
# ──────────────────────────────────────────────────────────────────────
def per_sample_mean(x):
    return jnp.mean(x, axis=tuple(range(1, x.ndim)))

def magnify_negatives(x, rate):
    return jnp.where(x < 0, x * rate, x)

def dummy_perceptual_loss(params, x, target):
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
    video = jax.device_put(
        jnp.array(np.random.randn(BATCH, TEMPORAL_LEN, HEIGHT, WIDTH, CHANNELS).astype(np.float32) * 0.1),
        data_sharding)
    mask = jax.device_put(
        jnp.ones((BATCH, TEMPORAL_LEN), dtype=jnp.bool_),
        NamedSharding(mesh, P('data', None)))
    return video, mask


# ──────────────────────────────────────────────────────────────────────
# Test 1: Loss function produces valid scalar
# ──────────────────────────────────────────────────────────────────────
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

    for key in ["MSE", "selection_loss", "kl_loss", "rl_loss", "per_sample_MAE", "kept_frame_density"]:
        val = float(aux[key])
        assert np.isfinite(val), f"{key} not finite: {val}"

    test_pass(f"loss={loss_val:.4f}, MSE={float(aux['MSE']):.4f}, "
              f"kl={float(aux['kl_loss']):.4f}, density={float(aux['kept_frame_density']):.4f}")
except Exception as e:
    test_fail("loss function", e)


# ──────────────────────────────────────────────────────────────────────
# Test 2: Train step
# ──────────────────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────────────
# Test 3: Loss decreases
# ──────────────────────────────────────────────────────────────────────
print("\n[Test 3] Loss decreases over training steps", flush=True)

try:
    fixed_video, fixed_mask = make_batch()

    losses = []
    NUM_STEPS = 10
    for step in range(NUM_STEPS):
        rngs_t = nnx.Rngs(step + 100)
        loss, aux = jit_train_step(model, optimizer, fixed_video, fixed_mask,
                                    hparams, rngs_t, dummy_perceptual_loss, None)
        losses.append(float(loss))

    first_half = np.mean(losses[:NUM_STEPS // 2])
    second_half = np.mean(losses[NUM_STEPS // 2:])

    print(f"  Losses: {[f'{l:.4f}' for l in losses]}", flush=True)
    print(f"  First half avg: {first_half:.4f}, Second half avg: {second_half:.4f}", flush=True)

    assert second_half < first_half, \
        f"Loss not decreasing: {first_half:.4f} -> {second_half:.4f}"

    test_pass(f"Loss decreased: {first_half:.4f} -> {second_half:.4f}")
except Exception as e:
    test_fail("loss decrease", e)


# ──────────────────────────────────────────────────────────────────────
# Test 4: Gradients are finite and nonzero
# ──────────────────────────────────────────────────────────────────────
print("\n[Test 4] Gradient sanity check", flush=True)

try:
    video, mask = make_batch()
    rngs_t = nnx.Rngs(999)

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
    assert np.isfinite(float(max_grad)), f"Max gradient not finite"

    test_pass(f"max_grad={float(max_grad):.6f}, no NaN")
except Exception as e:
    test_fail("gradient sanity", e)


# ──────────────────────────────────────────────────────────────────────
# Test 5: SIGTERM handling
# ──────────────────────────────────────────────────────────────────────
print("\n[Test 5] SIGTERM handling", flush=True)

try:
    import distributed_train as dt

    # Save original state
    original = dt._SHOULD_STOP

    assert not dt._SHOULD_STOP, "_SHOULD_STOP should be False initially"

    dt._signal_handler(signal.SIGTERM, None)
    assert dt._SHOULD_STOP, "Should be True after SIGTERM"

    dt._SHOULD_STOP = False
    dt._signal_handler(signal.SIGINT, None)
    assert dt._SHOULD_STOP, "Should be True after SIGINT"

    current_sigterm = signal.getsignal(signal.SIGTERM)
    current_sigint = signal.getsignal(signal.SIGINT)
    assert current_sigterm == dt._signal_handler, "SIGTERM handler not registered"
    assert current_sigint == dt._signal_handler, "SIGINT handler not registered"

    dt._SHOULD_STOP = False
    test_pass("SIGTERM/SIGINT handlers work correctly")
except Exception as e:
    test_fail("SIGTERM handling", e)


# ──────────────────────────────────────────────────────────────────────
# Test 6: Data-parallel sharding
# ──────────────────────────────────────────────────────────────────────
print("\n[Test 6] Data-parallel batch sharding", flush=True)

try:
    tagged = np.arange(num_devices).reshape(num_devices, 1, 1, 1, 1).astype(np.float32)
    tagged = np.broadcast_to(tagged, (num_devices, TEMPORAL_LEN, HEIGHT, WIDTH, CHANNELS)).copy()
    sharded_video = jax.device_put(jnp.array(tagged), data_sharding)

    assert sharded_video.shape == (num_devices, TEMPORAL_LEN, HEIGHT, WIDTH, CHANNELS)

    shards = sharded_video.addressable_shards
    devices_used = {s.device for s in shards}
    assert len(devices_used) == num_devices, f"Expected {num_devices} devices"

    full = np.array(sharded_video)
    for d in range(num_devices):
        assert np.isclose(full[d, 0, 0, 0, 0], float(d)), \
            f"Device {d} data mismatch: {full[d, 0, 0, 0, 0]}"

    test_pass(f"Sharded across {num_devices} devices, data correctly tagged")
except Exception as e:
    test_fail("data-parallel sharding", e)


# ──────────────────────────────────────────────────────────────────────
# Test 7: VGG perceptual loss
# ──────────────────────────────────────────────────────────────────────
print("\n[Test 7] VGG perceptual loss", flush=True)

try:
    from vgg_tests import load_vgg, get_adversarial_perceptual_loss_fn

    vgg_model, vgg_params = load_vgg()
    perceptual_fn = get_adversarial_perceptual_loss_fn(vgg_model)

    x = jnp.ones((2, TEMPORAL_LEN, HEIGHT, WIDTH, 3), dtype=jnp.bfloat16) * 0.5
    target = jnp.ones((2, TEMPORAL_LEN, HEIGHT, WIDTH, 3), dtype=jnp.bfloat16) * 0.3

    ploss = jax.jit(perceptual_fn)(vgg_params, x, target)

    ploss_val = float(jnp.mean(ploss))
    assert np.isfinite(ploss_val), f"Perceptual loss not finite: {ploss_val}"
    assert ploss_val >= 0, f"Perceptual loss negative: {ploss_val}"

    test_pass(f"VGG loaded, perceptual_loss={ploss_val:.4f}")
except Exception as e:
    test_fail("VGG perceptual loss", e)


# ──────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────
print(f"\n{'='*50}", flush=True)
print(f"TRAINING LOOP TESTS: {PASS_COUNT} passed, {FAIL_COUNT} failed", flush=True)
if FAIL_COUNT == 0:
    print("ALL TRAINING LOOP TESTS PASSED!", flush=True)
print(f"{'='*50}", flush=True)

if FAIL_COUNT > 0:
    sys.exit(1)
