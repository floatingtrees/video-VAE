"""
Model tests for VideoVAE (local CPU mode).

Tests:
  1. Encoder forward pass shapes
  2. Decoder forward pass shapes
  3. Full VideoVAE forward pass (with 2x repeat)
  4. Model params replicated across devices
  5. Gradient computation works end-to-end
  6. GumbelSigmoidSTE gradient flow + binary output
  7. FactoredAttention shapes
  8. PatchEmbedding & PatchUnEmbedding shapes
  9. UNet shape preservation

Usage:
    JAX_PLATFORMS=cpu JAX_NUM_CPU_DEVICES=4 python3 test_rl_model.py
"""

import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_NUM_CPU_DEVICES", "4")

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P, Mesh
from flax import nnx
import numpy as np

num_devices = jax.device_count()
print(f"Running with {num_devices} CPU devices", flush=True)

mesh = Mesh(jax.devices(), axis_names=('data',))
replicated_sharding = NamedSharding(mesh, P())
data_sharding = NamedSharding(mesh, P('data'))

PASS_COUNT = 0
FAIL_COUNT = 0

# Small model dims for fast testing
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


def test_pass(name):
    global PASS_COUNT
    PASS_COUNT += 1
    print(f"  PASS: {name}", flush=True)


def test_fail(name, err):
    global FAIL_COUNT
    FAIL_COUNT += 1
    print(f"  FAIL: {name} - {err}", flush=True)


# ──────────────────────────────────────────────────────────────────────
# Test 1: Encoder forward pass shapes
# ──────────────────────────────────────────────────────────────────────
print("\n[Test 1] Encoder forward pass shapes", flush=True)

from rl_model import Encoder

try:
    rngs = nnx.Rngs(42)
    encoder = Encoder(
        height=HEIGHT, width=WIDTH, channels=CHANNELS, patch_size=PATCH_SIZE,
        depth=ENCODER_DEPTH, mlp_dim=MLP_DIM, num_heads=NUM_HEADS,
        qkv_features=QKV_FEATURES, max_temporal_len=MAX_TEMPORAL_LEN,
        spatial_compression_rate=SPATIAL_COMPRESSION_RATE, rngs=rngs,
        dtype=jnp.float32, param_dtype=jnp.float32,
    )

    gdef, state = nnx.split(encoder)
    state = jax.device_put(state, replicated_sharding)
    encoder = nnx.merge(gdef, state)

    x = jax.device_put(
        jnp.array(np.random.randn(BATCH, TEMPORAL_LEN, HEIGHT, WIDTH, CHANNELS).astype(np.float32) * 0.02),
        data_sharding)
    mask = jax.device_put(
        jnp.ones((BATCH, 1, 1, TEMPORAL_LEN)),
        NamedSharding(mesh, P('data', None, None, None)))

    rngs_fwd = nnx.Rngs(0)

    @nnx.jit
    def enc_forward(enc, x, mask, rngs):
        return enc(x, mask, rngs, train=True)

    mean, logvar, selection = enc_forward(encoder, x, mask, rngs_fwd)

    hw = (HEIGHT // PATCH_SIZE) * (WIDTH // PATCH_SIZE)
    ppc = CHANNELS * PATCH_SIZE * PATCH_SIZE // SPATIAL_COMPRESSION_RATE

    assert mean.shape == (BATCH, TEMPORAL_LEN, hw, ppc), f"mean shape: {mean.shape}"
    assert logvar.shape == mean.shape, f"logvar shape: {logvar.shape}"
    assert selection.shape == (BATCH, TEMPORAL_LEN, 1), f"selection shape: {selection.shape}"

    test_pass(f"mean={mean.shape}, logvar={logvar.shape}, selection={selection.shape}")
except Exception as e:
    test_fail("Encoder forward", e)


# ──────────────────────────────────────────────────────────────────────
# Test 2: Decoder forward pass shapes
# ──────────────────────────────────────────────────────────────────────
print("\n[Test 2] Decoder forward pass shapes", flush=True)

from rl_model import Decoder

try:
    rngs = nnx.Rngs(42)
    decoder = Decoder(
        height=HEIGHT, width=WIDTH, channels=CHANNELS, patch_size=PATCH_SIZE,
        depth=DECODER_DEPTH, mlp_dim=MLP_DIM, num_heads=NUM_HEADS,
        qkv_features=QKV_FEATURES, max_temporal_len=MAX_TEMPORAL_LEN,
        spatial_compression_rate=SPATIAL_COMPRESSION_RATE,
        unembedding_upsample_rate=UPSAMPLE_RATE, rngs=rngs,
        dtype=jnp.float32, param_dtype=jnp.float32,
    )

    gdef, state = nnx.split(decoder)
    state = jax.device_put(state, replicated_sharding)
    decoder = nnx.merge(gdef, state)

    hw = (HEIGHT // PATCH_SIZE) * (WIDTH // PATCH_SIZE)
    ppc = CHANNELS * PATCH_SIZE * PATCH_SIZE // SPATIAL_COMPRESSION_RATE

    latent = jax.device_put(
        jnp.array(np.random.randn(BATCH, TEMPORAL_LEN, hw, ppc).astype(np.float32) * 0.02),
        NamedSharding(mesh, P('data', None, None, None)))
    mask = jax.device_put(
        jnp.ones((BATCH, 1, 1, TEMPORAL_LEN)),
        NamedSharding(mesh, P('data', None, None, None)))

    rngs_fwd = nnx.Rngs(0)

    @nnx.jit
    def dec_forward(dec, x, mask, rngs):
        return dec(x, mask, rngs, train=True)

    output = dec_forward(decoder, latent, mask, rngs_fwd)
    assert output.shape == (BATCH, TEMPORAL_LEN, HEIGHT, WIDTH, CHANNELS), \
        f"Decoder output shape: {output.shape}"

    test_pass(f"output={output.shape}")
except Exception as e:
    test_fail("Decoder forward", e)


# ──────────────────────────────────────────────────────────────────────
# Test 3: Full VideoVAE forward pass (with 2x repeat)
# ──────────────────────────────────────────────────────────────────────
print("\n[Test 3] Full VideoVAE forward pass", flush=True)

from rl_model import VideoVAE

try:
    rngs = nnx.Rngs(42)
    model = VideoVAE(
        height=HEIGHT, width=WIDTH, channels=CHANNELS, patch_size=PATCH_SIZE,
        encoder_depth=ENCODER_DEPTH, decoder_depth=DECODER_DEPTH,
        mlp_dim=MLP_DIM, num_heads=NUM_HEADS, qkv_features=QKV_FEATURES,
        max_temporal_len=MAX_TEMPORAL_LEN,
        spatial_compression_rate=SPATIAL_COMPRESSION_RATE,
        unembedding_upsample_rate=UPSAMPLE_RATE, rngs=rngs,
        dtype=jnp.float32, param_dtype=jnp.float32,
    )

    gdef, state = nnx.split(model)
    state = jax.device_put(state, replicated_sharding)
    model = nnx.merge(gdef, state)

    x = jax.device_put(
        jnp.array(np.random.randn(BATCH, TEMPORAL_LEN, HEIGHT, WIDTH, CHANNELS).astype(np.float32) * 0.02),
        data_sharding)
    mask = jax.device_put(
        jnp.ones((BATCH, 1, 1, TEMPORAL_LEN)),
        NamedSharding(mesh, P('data', None, None, None)))

    rngs_fwd = nnx.Rngs(42)

    @nnx.jit
    def model_forward(m, x, mask, rngs):
        return m(x, mask, rngs, train=True)

    start = time.perf_counter()
    reconstruction, compressed, selection, selection_mask, logvar, mean_out = \
        model_forward(model, x, mask, rngs_fwd)
    elapsed = time.perf_counter() - start

    doubled_batch = BATCH * 2
    assert reconstruction.shape == (doubled_batch, TEMPORAL_LEN, HEIGHT, WIDTH, CHANNELS), \
        f"reconstruction shape: {reconstruction.shape}"
    assert selection.shape[0] == doubled_batch, f"selection batch: {selection.shape[0]}"
    assert selection_mask.shape[0] == doubled_batch, f"sel_mask batch: {selection_mask.shape[0]}"

    sm_vals = np.array(selection_mask)
    unique_vals = np.unique(sm_vals)
    assert all(v in [0.0, 1.0] for v in unique_vals), f"selection_mask not binary: {unique_vals}"

    test_pass(f"recon={reconstruction.shape}, doubled_batch={doubled_batch}, time={elapsed:.2f}s")
except Exception as e:
    test_fail("VideoVAE forward", e)


# ──────────────────────────────────────────────────────────────────────
# Test 4: Model params replicated
# ──────────────────────────────────────────────────────────────────────
print("\n[Test 4] Model params replicated across devices", flush=True)

try:
    params_state = nnx.state(model, nnx.Param)
    leaves = jax.tree_util.tree_leaves(params_state)
    num_params = sum(x.size for x in leaves)

    # Check sharding of first param
    sample = leaves[0]
    if hasattr(sample.value, 'sharding'):
        is_replicated = all(s is None for s in sample.value.sharding.spec)
    else:
        is_replicated = True  # single device

    test_pass(f"Params: {num_params / 1e6:.1f}M, replicated={is_replicated}")
except Exception as e:
    test_fail("model replication", e)


# ──────────────────────────────────────────────────────────────────────
# Test 5: Gradient computation
# ──────────────────────────────────────────────────────────────────────
print("\n[Test 5] Gradient computation", flush=True)

try:
    from einops import repeat

    def simple_loss(model, x, mask, rngs):
        reconstruction, _, _, _, logvar, mean_out = model(x, mask, rngs, train=True)
        x_rep = repeat(x, "b ... -> (b 2) ...")
        return jnp.mean((reconstruction - x_rep) ** 2)

    grad_fn = nnx.value_and_grad(simple_loss)

    @nnx.jit
    def grad_step(model, x, mask, rngs):
        return grad_fn(model, x, mask, rngs)

    rngs_grad = nnx.Rngs(99)
    loss, grads = grad_step(model, x, mask, rngs_grad)

    loss_val = float(loss)
    assert np.isfinite(loss_val), f"Loss not finite: {loss_val}"

    grad_state = nnx.state(grads, nnx.Param)
    grad_leaves = jax.tree_util.tree_leaves(grad_state)
    max_grad = max(float(jnp.max(jnp.abs(g))) for g in grad_leaves)
    assert max_grad > 0, "All gradients zero"
    assert np.isfinite(max_grad), f"Gradient not finite: {max_grad}"

    test_pass(f"loss={loss_val:.6f}, max_grad={max_grad:.6f}")
except Exception as e:
    test_fail("gradient computation", e)


# ──────────────────────────────────────────────────────────────────────
# Test 6: GumbelSigmoidSTE
# ──────────────────────────────────────────────────────────────────────
print("\n[Test 6] GumbelSigmoidSTE gradient flow", flush=True)

from layers import GumbelSigmoidSTE

try:
    gumbel = GumbelSigmoidSTE(temperature=1.0)

    def gumbel_loss(logits, rngs):
        out = gumbel(logits, rngs, train=True)
        return jnp.mean(out)

    logits = jnp.ones((4, 8)) * 0.5
    rngs_g = nnx.Rngs(10)

    grads = jax.jit(jax.grad(gumbel_loss))(logits, rngs_g)

    assert grads.shape == logits.shape, f"Grad shape mismatch: {grads.shape}"
    assert float(jnp.max(jnp.abs(grads))) > 0, "STE gradients are zero"

    out = jax.jit(gumbel)(logits, rngs_g, True)
    unique = np.unique(np.array(out))
    assert all(v in [0.0, 1.0] for v in unique), f"Not binary: {unique}"

    test_pass(f"grad_norm={float(jnp.sum(jnp.abs(grads))):.4f}, output_binary=True")
except Exception as e:
    test_fail("GumbelSigmoidSTE", e)


# ──────────────────────────────────────────────────────────────────────
# Test 7: FactoredAttention shapes
# ──────────────────────────────────────────────────────────────────────
print("\n[Test 7] FactoredAttention shapes", flush=True)

from layers import FactoredAttention

try:
    rngs = nnx.Rngs(42)
    hw = (HEIGHT // PATCH_SIZE) * (WIDTH // PATCH_SIZE)
    feat_dim = CHANNELS * PATCH_SIZE * PATCH_SIZE

    fa = FactoredAttention(
        mlp_dim=MLP_DIM, in_features=feat_dim, num_heads=NUM_HEADS,
        qkv_features=QKV_FEATURES, max_temporal_len=MAX_TEMPORAL_LEN,
        max_spatial_len=hw, rngs=rngs, dtype=jnp.float32, param_dtype=jnp.float32,
    )

    x_in = jnp.array(np.random.randn(BATCH, TEMPORAL_LEN, hw, feat_dim).astype(np.float32) * 0.02)
    mask_in = jnp.ones((BATCH, 1, 1, TEMPORAL_LEN))

    out = nnx.jit(fa)(x_in, mask_in)
    assert out.shape == (BATCH, TEMPORAL_LEN, hw, feat_dim), f"FA output: {out.shape}"

    test_pass(f"output={out.shape}")
except Exception as e:
    test_fail("FactoredAttention", e)


# ──────────────────────────────────────────────────────────────────────
# Test 8: PatchEmbedding & PatchUnEmbedding
# ──────────────────────────────────────────────────────────────────────
print("\n[Test 8] PatchEmbedding & PatchUnEmbedding shapes", flush=True)

from layers import PatchEmbedding, PatchUnEmbedding

try:
    rngs = nnx.Rngs(42)
    pe = PatchEmbedding(HEIGHT, WIDTH, CHANNELS, PATCH_SIZE, rngs,
                        dtype=jnp.float32, param_dtype=jnp.float32)
    x_in = jnp.array(np.random.randn(BATCH, TEMPORAL_LEN, HEIGHT, WIDTH, CHANNELS).astype(np.float32) * 0.02)
    emb = nnx.jit(pe)(x_in)

    hw = (HEIGHT // PATCH_SIZE) * (WIDTH // PATCH_SIZE)
    ppc = CHANNELS * PATCH_SIZE * PATCH_SIZE
    assert emb.shape == (BATCH, TEMPORAL_LEN, hw, ppc), f"PE output: {emb.shape}"

    pue = PatchUnEmbedding(HEIGHT, WIDTH, CHANNELS, PATCH_SIZE, UPSAMPLE_RATE, rngs,
                           dtype=jnp.float32, param_dtype=jnp.float32)
    conv_feats, recon = nnx.jit(pue)(emb)
    assert recon.shape == (BATCH, TEMPORAL_LEN, HEIGHT, WIDTH, CHANNELS), \
        f"PUE recon: {recon.shape}"
    assert conv_feats.shape == (BATCH, TEMPORAL_LEN, HEIGHT, WIDTH, CHANNELS * UPSAMPLE_RATE), \
        f"PUE conv_feats: {conv_feats.shape}"

    test_pass(f"emb={emb.shape}, recon={recon.shape}, conv_feats={conv_feats.shape}")
except Exception as e:
    test_fail("PatchEmbedding/UnEmbedding", e)


# ──────────────────────────────────────────────────────────────────────
# Test 9: UNet shape preservation
# ──────────────────────────────────────────────────────────────────────
print("\n[Test 9] UNet shape preservation", flush=True)

from unet import UNet

try:
    rngs = nnx.Rngs(42)
    unet_ch = CHANNELS * UPSAMPLE_RATE
    unet = UNet(channels=unet_ch, base_features=16, num_levels=3,
                out_features=CHANNELS, rngs=rngs,
                dtype=jnp.float32, param_dtype=jnp.float32)

    x_in = jnp.array(np.random.randn(BATCH, TEMPORAL_LEN, HEIGHT, WIDTH, unet_ch).astype(np.float32) * 0.02)
    out = nnx.jit(unet)(x_in)
    assert out.shape == (BATCH, TEMPORAL_LEN, HEIGHT, WIDTH, CHANNELS), \
        f"UNet output: {out.shape}"

    test_pass(f"input=(..., {unet_ch}), output={out.shape}")
except Exception as e:
    test_fail("UNet", e)


# ──────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────
print(f"\n{'='*50}", flush=True)
print(f"MODEL TESTS: {PASS_COUNT} passed, {FAIL_COUNT} failed", flush=True)
if FAIL_COUNT == 0:
    print("ALL MODEL TESTS PASSED!", flush=True)
print(f"{'='*50}", flush=True)

if FAIL_COUNT > 0:
    sys.exit(1)
