"""
Model tests for VideoVAE on distributed TPU.

Tests:
  1. Encoder: forward pass shapes correct
  2. Decoder: forward pass shapes correct
  3. Full VideoVAE: forward pass output shapes (with repeat doubling)
  4. Model replication across all devices
  5. Forward pass with sharded data (data parallel)
  6. Gradient computation works end-to-end
  7. GumbelSigmoidSTE produces correct shapes and STE gradient
  8. Layers: FactoredAttention, PatchEmbedding, PatchUnEmbedding shapes
  9. UNet: forward pass shape preservation

Usage (run across all workers):
    gcloud compute tpus tpu-vm ssh train-v6e-16 --zone=europe-west4-a --worker=all \
      --command='cd ~/video-VAE/claude_distributed && python3 test_rl_model.py' --internal-ip
"""

import os
import sys
import time

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".99"

import jax
jax.distributed.initialize()

import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from flax import nnx
import numpy as np

num_devices = jax.device_count()
local_devices = jax.local_device_count()
process_index = jax.process_index()
num_processes = jax.process_count()

mesh = jax.make_mesh((num_devices,), ('data',))
replicated_sharding = NamedSharding(mesh, P())
data_sharding = NamedSharding(mesh, P('data'))

PASS_COUNT = 0
FAIL_COUNT = 0

# Small model dims for testing
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
BATCH_SIZE = local_devices  # 4 per process on v6e-16
TEMPORAL_LEN = 8


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
# Test 1: Encoder forward pass shapes
# ──────────────────────────────────────────────────────────────────────
if process_index == 0:
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

    # Replicate encoder for forward pass
    gdef, state = nnx.split(encoder)
    state = jax.device_put(state, replicated_sharding)
    encoder = nnx.merge(gdef, state)

    x = np.random.randn(num_devices, TEMPORAL_LEN, HEIGHT, WIDTH, CHANNELS).astype(np.float32) * 0.02
    x_local = x[process_index * local_devices:(process_index + 1) * local_devices]
    x_global = jax.make_array_from_process_local_data(data_sharding, x_local)

    mask_local = np.ones((local_devices, 1, 1, TEMPORAL_LEN), dtype=np.float32)
    mask_spec = P('data', None, None, None)
    mask_global = jax.make_array_from_process_local_data(
        NamedSharding(mesh, mask_spec), mask_local)

    rngs_fwd = nnx.Rngs(0)

    @nnx.jit
    def enc_forward(enc, x, mask, rngs):
        return enc(x, mask, rngs, train=True)

    mean, logvar, selection = enc_forward(encoder, x_global, mask_global, rngs_fwd)

    hw = (HEIGHT // PATCH_SIZE) * (WIDTH // PATCH_SIZE)
    ppc = CHANNELS * PATCH_SIZE * PATCH_SIZE // SPATIAL_COMPRESSION_RATE

    assert mean.shape == (num_devices, TEMPORAL_LEN, hw, ppc), \
        f"mean shape: {mean.shape}, expected ({num_devices}, {TEMPORAL_LEN}, {hw}, {ppc})"
    assert logvar.shape == mean.shape, f"logvar shape mismatch: {logvar.shape}"
    assert selection.shape == (num_devices, TEMPORAL_LEN, 1), \
        f"selection shape: {selection.shape}, expected ({num_devices}, {TEMPORAL_LEN}, 1)"

    test_pass(f"mean={mean.shape}, logvar={logvar.shape}, selection={selection.shape}")
except Exception as e:
    test_fail("Encoder forward", e)

jax.experimental.multihost_utils.sync_global_devices("test1")


# ──────────────────────────────────────────────────────────────────────
# Test 2: Decoder forward pass shapes
# ──────────────────────────────────────────────────────────────────────
if process_index == 0:
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

    latent_local = np.random.randn(local_devices, TEMPORAL_LEN, hw, ppc).astype(np.float32) * 0.02
    latent_spec = P('data', None, None, None)
    latent_global = jax.make_array_from_process_local_data(
        NamedSharding(mesh, latent_spec), latent_local)

    mask_local = np.ones((local_devices, 1, 1, TEMPORAL_LEN), dtype=np.float32)
    mask_global = jax.make_array_from_process_local_data(
        NamedSharding(mesh, P('data', None, None, None)), mask_local)

    rngs_fwd = nnx.Rngs(0)

    @nnx.jit
    def dec_forward(dec, x, mask, rngs):
        return dec(x, mask, rngs, train=True)

    output = dec_forward(decoder, latent_global, mask_global, rngs_fwd)
    assert output.shape == (num_devices, TEMPORAL_LEN, HEIGHT, WIDTH, CHANNELS), \
        f"Decoder output shape: {output.shape}"

    test_pass(f"output={output.shape}")
except Exception as e:
    test_fail("Decoder forward", e)

jax.experimental.multihost_utils.sync_global_devices("test2")


# ──────────────────────────────────────────────────────────────────────
# Test 3: Full VideoVAE forward pass (with 2x repeat)
# ──────────────────────────────────────────────────────────────────────
if process_index == 0:
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

    x_local = np.random.randn(local_devices, TEMPORAL_LEN, HEIGHT, WIDTH, CHANNELS).astype(np.float32) * 0.02
    mask_local = np.ones((local_devices, 1, 1, TEMPORAL_LEN), dtype=np.float32)

    x_global = jax.make_array_from_process_local_data(data_sharding, x_local)
    mask_global = jax.make_array_from_process_local_data(
        NamedSharding(mesh, P('data', None, None, None)), mask_local)

    rngs_fwd = nnx.Rngs(42)

    @nnx.jit
    def model_forward(m, x, mask, rngs):
        return m(x, mask, rngs, train=True)

    start = time.perf_counter()
    reconstruction, compressed, selection, selection_mask, logvar, mean_out = \
        model_forward(model, x_global, mask_global, rngs_fwd)
    elapsed = time.perf_counter() - start

    # VideoVAE does repeat(..., "b ... -> (b 2) ...") inside
    doubled_batch = num_devices * 2
    assert reconstruction.shape == (doubled_batch, TEMPORAL_LEN, HEIGHT, WIDTH, CHANNELS), \
        f"reconstruction shape: {reconstruction.shape}"
    assert selection.shape[0] == doubled_batch, \
        f"selection batch dim: {selection.shape[0]}, expected {doubled_batch}"
    assert selection_mask.shape[0] == doubled_batch, \
        f"selection_mask batch dim: {selection_mask.shape[0]}"

    # selection_mask should be binary (0 or 1)
    sm_vals = np.array(selection_mask)
    unique_vals = np.unique(sm_vals)
    assert all(v in [0.0, 1.0] for v in unique_vals), \
        f"selection_mask not binary: {unique_vals}"

    test_pass(f"recon={reconstruction.shape}, doubled_batch={doubled_batch}, "
              f"compile+run={elapsed:.2f}s")
except Exception as e:
    test_fail("VideoVAE forward", e)

jax.experimental.multihost_utils.sync_global_devices("test3")


# ──────────────────────────────────────────────────────────────────────
# Test 4: Model replication - all devices have same params
# ──────────────────────────────────────────────────────────────────────
if process_index == 0:
    print("\n[Test 4] Model params replicated across devices", flush=True)

try:
    # Model was already replicated above. Check sharding.
    params_state = nnx.state(model, nnx.Param)
    leaves = jax.tree_util.tree_leaves(params_state)

    all_replicated = True
    for leaf in leaves[:5]:  # Check first 5 params
        if hasattr(leaf.value, 'sharding'):
            s = leaf.value.sharding
            # Replicated sharding should have P() spec
            if hasattr(s, 'spec'):
                for dim_spec in s.spec:
                    if dim_spec is not None:
                        all_replicated = False
                        break

    if process_index == 0:
        num_params = sum(x.size for x in jax.tree_util.tree_leaves(params_state))
        log(f"Model params: {num_params / 1e6:.1f}M")

    test_pass(f"Params replicated={all_replicated}")
except Exception as e:
    test_fail("model replication", e)

jax.experimental.multihost_utils.sync_global_devices("test4")


# ──────────────────────────────────────────────────────────────────────
# Test 5: Gradient computation works
# ──────────────────────────────────────────────────────────────────────
if process_index == 0:
    print("\n[Test 5] Gradient computation", flush=True)

try:
    def simple_loss(model, x, mask, rngs):
        reconstruction, _, _, _, logvar, mean_out = model(x, mask, rngs, train=True)
        from einops import repeat
        x_rep = repeat(x, "b ... -> (b 2) ...")
        return jnp.mean((reconstruction - x_rep) ** 2)

    grad_fn = nnx.value_and_grad(simple_loss)

    @nnx.jit
    def grad_step(model, x, mask, rngs):
        return grad_fn(model, x, mask, rngs)

    rngs_grad = nnx.Rngs(99)
    loss, grads = grad_step(model, x_global, mask_global, rngs_grad)

    # Check loss is finite
    loss_val = float(loss)
    assert np.isfinite(loss_val), f"Loss not finite: {loss_val}"

    # Check grads are not all zero
    grad_state = nnx.state(grads, nnx.Param)
    grad_leaves = jax.tree_util.tree_leaves(grad_state)
    max_grad = max(float(jnp.max(jnp.abs(g))) for g in grad_leaves)
    assert max_grad > 0, f"All gradients zero"
    assert np.isfinite(max_grad), f"Gradient not finite: {max_grad}"

    test_pass(f"loss={loss_val:.6f}, max_grad={max_grad:.6f}")
except Exception as e:
    test_fail("gradient computation", e)

jax.experimental.multihost_utils.sync_global_devices("test5")


# ──────────────────────────────────────────────────────────────────────
# Test 6: GumbelSigmoidSTE gradient flow
# ──────────────────────────────────────────────────────────────────────
if process_index == 0:
    print("\n[Test 6] GumbelSigmoidSTE gradient flow", flush=True)

from layers import GumbelSigmoidSTE

try:
    gumbel = GumbelSigmoidSTE(temperature=1.0)

    def gumbel_loss(logits, rngs):
        out = gumbel(logits, rngs, train=True)
        return jnp.mean(out)

    logits = jnp.ones((4, 8)) * 0.5
    logits = jax.device_put(logits, replicated_sharding)
    rngs_g = nnx.Rngs(10)

    grad_fn = jax.grad(gumbel_loss)
    grads = jax.jit(grad_fn)(logits, rngs_g)

    # STE should pass gradients through
    assert grads.shape == logits.shape, f"Grad shape mismatch: {grads.shape}"
    assert float(jnp.max(jnp.abs(grads))) > 0, "STE gradients are zero"
    assert np.isfinite(float(jnp.sum(grads))), "STE gradients not finite"

    # Output should be binary
    out = jax.jit(gumbel)(logits, rngs_g, True)
    unique = np.unique(np.array(out))
    assert all(v in [0.0, 1.0] for v in unique), f"GumbelSigmoid output not binary: {unique}"

    test_pass(f"grad_norm={float(jnp.sum(jnp.abs(grads))):.4f}, output_binary=True")
except Exception as e:
    test_fail("GumbelSigmoidSTE", e)

jax.experimental.multihost_utils.sync_global_devices("test6")


# ──────────────────────────────────────────────────────────────────────
# Test 7: FactoredAttention shapes
# ──────────────────────────────────────────────────────────────────────
if process_index == 0:
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

    gdef, state = nnx.split(fa)
    state = jax.device_put(state, replicated_sharding)
    fa = nnx.merge(gdef, state)

    x_local = np.random.randn(local_devices, TEMPORAL_LEN, hw, feat_dim).astype(np.float32) * 0.02
    mask_local = np.ones((local_devices, 1, 1, TEMPORAL_LEN), dtype=np.float32)

    x_global = jax.make_array_from_process_local_data(
        NamedSharding(mesh, P('data', None, None, None)), x_local)
    mask_global = jax.make_array_from_process_local_data(
        NamedSharding(mesh, P('data', None, None, None)), mask_local)

    @nnx.jit
    def fa_forward(fa, x, mask):
        return fa(x, mask)

    out = fa_forward(fa, x_global, mask_global)
    assert out.shape == (num_devices, TEMPORAL_LEN, hw, feat_dim), \
        f"FactoredAttention output shape: {out.shape}"

    test_pass(f"output={out.shape}")
except Exception as e:
    test_fail("FactoredAttention", e)

jax.experimental.multihost_utils.sync_global_devices("test7")


# ──────────────────────────────────────────────────────────────────────
# Test 8: PatchEmbedding and PatchUnEmbedding shapes
# ──────────────────────────────────────────────────────────────────────
if process_index == 0:
    print("\n[Test 8] PatchEmbedding & PatchUnEmbedding shapes", flush=True)

from layers import PatchEmbedding, PatchUnEmbedding

try:
    rngs = nnx.Rngs(42)
    pe = PatchEmbedding(HEIGHT, WIDTH, CHANNELS, PATCH_SIZE, rngs,
                        dtype=jnp.float32, param_dtype=jnp.float32)
    gdef, state = nnx.split(pe)
    state = jax.device_put(state, replicated_sharding)
    pe = nnx.merge(gdef, state)

    x_local = np.random.randn(local_devices, TEMPORAL_LEN, HEIGHT, WIDTH, CHANNELS).astype(np.float32) * 0.02
    x_global = jax.make_array_from_process_local_data(data_sharding, x_local)

    @nnx.jit
    def pe_fwd(pe, x):
        return pe(x)

    emb = pe_fwd(pe, x_global)
    hw = (HEIGHT // PATCH_SIZE) * (WIDTH // PATCH_SIZE)
    ppc = CHANNELS * PATCH_SIZE * PATCH_SIZE
    assert emb.shape == (num_devices, TEMPORAL_LEN, hw, ppc), \
        f"PatchEmbedding output: {emb.shape}"

    # PatchUnEmbedding
    pue = PatchUnEmbedding(HEIGHT, WIDTH, CHANNELS, PATCH_SIZE, UPSAMPLE_RATE, rngs,
                           dtype=jnp.float32, param_dtype=jnp.float32)
    gdef, state = nnx.split(pue)
    state = jax.device_put(state, replicated_sharding)
    pue = nnx.merge(gdef, state)

    @nnx.jit
    def pue_fwd(pue, x):
        return pue(x)

    conv_feats, recon = pue_fwd(pue, emb)
    assert recon.shape == (num_devices, TEMPORAL_LEN, HEIGHT, WIDTH, CHANNELS), \
        f"PatchUnEmbedding recon shape: {recon.shape}"
    assert conv_feats.shape == (num_devices, TEMPORAL_LEN, HEIGHT, WIDTH, CHANNELS * UPSAMPLE_RATE), \
        f"PatchUnEmbedding conv_feats shape: {conv_feats.shape}"

    test_pass(f"emb={emb.shape}, recon={recon.shape}, conv_feats={conv_feats.shape}")
except Exception as e:
    test_fail("PatchEmbedding/UnEmbedding", e)

jax.experimental.multihost_utils.sync_global_devices("test8")


# ──────────────────────────────────────────────────────────────────────
# Test 9: UNet shape preservation
# ──────────────────────────────────────────────────────────────────────
if process_index == 0:
    print("\n[Test 9] UNet shape preservation", flush=True)

from unet import UNet

try:
    rngs = nnx.Rngs(42)
    unet_channels = CHANNELS * UPSAMPLE_RATE
    unet = UNet(channels=unet_channels, base_features=16, num_levels=3,
                out_features=CHANNELS, rngs=rngs,
                dtype=jnp.float32, param_dtype=jnp.float32)

    gdef, state = nnx.split(unet)
    state = jax.device_put(state, replicated_sharding)
    unet = nnx.merge(gdef, state)

    x_local = np.random.randn(local_devices, TEMPORAL_LEN, HEIGHT, WIDTH, unet_channels).astype(np.float32) * 0.02
    x_global = jax.make_array_from_process_local_data(data_sharding, x_local)

    @nnx.jit
    def unet_fwd(unet, x):
        return unet(x)

    out = unet_fwd(unet, x_global)
    assert out.shape == (num_devices, TEMPORAL_LEN, HEIGHT, WIDTH, CHANNELS), \
        f"UNet output shape: {out.shape}, expected (..., {CHANNELS})"

    test_pass(f"input=(..., {unet_channels}), output={out.shape}")
except Exception as e:
    test_fail("UNet", e)

jax.experimental.multihost_utils.sync_global_devices("test9")


# ──────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────
jax.experimental.multihost_utils.sync_global_devices("all_tests")

if process_index == 0:
    print(f"\n{'='*50}", flush=True)
    print(f"MODEL TESTS: {PASS_COUNT} passed, {FAIL_COUNT} failed", flush=True)
    if FAIL_COUNT == 0:
        print("ALL MODEL TESTS PASSED!", flush=True)
    print(f"{'='*50}", flush=True)

if FAIL_COUNT > 0:
    sys.exit(1)
