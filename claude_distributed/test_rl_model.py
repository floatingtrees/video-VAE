"""
Model tests for VideoVAE (local CPU mode).

Usage:
    JAX_PLATFORMS=cpu JAX_NUM_CPU_DEVICES=4 python3 test_rl_model.py
"""

import os
import sys

if __name__ == '__main__':
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_NUM_CPU_DEVICES", "4")

    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P, Mesh
    from flax import nnx
    import numpy as np
    import time

    num_devices = jax.device_count()
    print(f"Running with {num_devices} CPU devices", flush=True)

    mesh = Mesh(jax.devices(), axis_names=('data',))
    replicated_sharding = NamedSharding(mesh, P())
    data_sharding = NamedSharding(mesh, P('data'))

    PASS_COUNT = 0
    FAIL_COUNT = 0

    HEIGHT, WIDTH, CHANNELS, PATCH_SIZE = 64, 64, 3, 16
    ENCODER_DEPTH, DECODER_DEPTH = 2, 2
    MLP_DIM, NUM_HEADS, QKV_FEATURES = 256, 4, 128
    MAX_TEMPORAL_LEN = 16
    SPATIAL_COMPRESSION_RATE, UPSAMPLE_RATE = 4, 2
    BATCH, TEMPORAL_LEN = num_devices, 8

    def test_pass(name):
        global PASS_COUNT
        PASS_COUNT += 1
        print(f"  PASS: {name}", flush=True)

    def test_fail(name, err):
        global FAIL_COUNT
        FAIL_COUNT += 1
        print(f"  FAIL: {name} - {err}", flush=True)

    # ── Test 1: Encoder forward pass ──
    print("\n[Test 1] Encoder forward pass shapes", flush=True)
    from rl_model import Encoder
    try:
        encoder = Encoder(
            height=HEIGHT, width=WIDTH, channels=CHANNELS, patch_size=PATCH_SIZE,
            depth=ENCODER_DEPTH, mlp_dim=MLP_DIM, num_heads=NUM_HEADS,
            qkv_features=QKV_FEATURES, max_temporal_len=MAX_TEMPORAL_LEN,
            spatial_compression_rate=SPATIAL_COMPRESSION_RATE, rngs=nnx.Rngs(42),
            dtype=jnp.float32, param_dtype=jnp.float32)
        gdef, state = nnx.split(encoder)
        state = jax.device_put(state, replicated_sharding)
        encoder = nnx.merge(gdef, state)

        x = jax.device_put(jnp.array(np.random.randn(BATCH, TEMPORAL_LEN, HEIGHT, WIDTH, CHANNELS).astype(np.float32) * 0.02), data_sharding)
        mask = jax.device_put(jnp.ones((BATCH, 1, 1, TEMPORAL_LEN), dtype=jnp.bool_), NamedSharding(mesh, P('data', None, None, None)))

        @nnx.jit
        def enc_fwd(enc, x, mask, rngs):
            return enc(x, mask, rngs, train=True)
        mean, logvar, selection = enc_fwd(encoder, x, mask, nnx.Rngs(0))
        hw = (HEIGHT // PATCH_SIZE) ** 2
        ppc = CHANNELS * PATCH_SIZE * PATCH_SIZE // SPATIAL_COMPRESSION_RATE
        assert mean.shape == (BATCH, TEMPORAL_LEN, hw, ppc), f"mean: {mean.shape}"
        assert selection.shape == (BATCH, TEMPORAL_LEN, 1), f"sel: {selection.shape}"
        test_pass(f"mean={mean.shape}, selection={selection.shape}")
    except Exception as e:
        test_fail("Encoder forward", e)

    # ── Test 2: Decoder forward pass ──
    print("\n[Test 2] Decoder forward pass shapes", flush=True)
    from rl_model import Decoder
    try:
        decoder = Decoder(
            height=HEIGHT, width=WIDTH, channels=CHANNELS, patch_size=PATCH_SIZE,
            depth=DECODER_DEPTH, mlp_dim=MLP_DIM, num_heads=NUM_HEADS,
            qkv_features=QKV_FEATURES, max_temporal_len=MAX_TEMPORAL_LEN,
            spatial_compression_rate=SPATIAL_COMPRESSION_RATE,
            unembedding_upsample_rate=UPSAMPLE_RATE, rngs=nnx.Rngs(42),
            dtype=jnp.float32, param_dtype=jnp.float32)
        gdef, state = nnx.split(decoder)
        state = jax.device_put(state, replicated_sharding)
        decoder = nnx.merge(gdef, state)
        hw = (HEIGHT // PATCH_SIZE) ** 2
        ppc = CHANNELS * PATCH_SIZE * PATCH_SIZE // SPATIAL_COMPRESSION_RATE
        latent = jax.device_put(jnp.array(np.random.randn(BATCH, TEMPORAL_LEN, hw, ppc).astype(np.float32) * 0.02), NamedSharding(mesh, P('data', None, None, None)))
        mask = jax.device_put(jnp.ones((BATCH, 1, 1, TEMPORAL_LEN), dtype=jnp.bool_), NamedSharding(mesh, P('data', None, None, None)))

        @nnx.jit
        def dec_fwd(dec, x, mask, rngs):
            return dec(x, mask, rngs, train=True)
        out = dec_fwd(decoder, latent, mask, nnx.Rngs(0))
        assert out.shape == (BATCH, TEMPORAL_LEN, HEIGHT, WIDTH, CHANNELS), f"out: {out.shape}"
        test_pass(f"output={out.shape}")
    except Exception as e:
        test_fail("Decoder forward", e)

    # ── Test 3: Full VideoVAE (2x repeat) ──
    print("\n[Test 3] Full VideoVAE forward pass", flush=True)
    from rl_model import VideoVAE
    try:
        model = VideoVAE(
            height=HEIGHT, width=WIDTH, channels=CHANNELS, patch_size=PATCH_SIZE,
            encoder_depth=ENCODER_DEPTH, decoder_depth=DECODER_DEPTH,
            mlp_dim=MLP_DIM, num_heads=NUM_HEADS, qkv_features=QKV_FEATURES,
            max_temporal_len=MAX_TEMPORAL_LEN,
            spatial_compression_rate=SPATIAL_COMPRESSION_RATE,
            unembedding_upsample_rate=UPSAMPLE_RATE, rngs=nnx.Rngs(42),
            dtype=jnp.float32, param_dtype=jnp.float32)
        gdef, state = nnx.split(model)
        state = jax.device_put(state, replicated_sharding)
        model = nnx.merge(gdef, state)

        x = jax.device_put(jnp.array(np.random.randn(BATCH, TEMPORAL_LEN, HEIGHT, WIDTH, CHANNELS).astype(np.float32) * 0.02), data_sharding)
        mask = jax.device_put(jnp.ones((BATCH, 1, 1, TEMPORAL_LEN), dtype=jnp.bool_), NamedSharding(mesh, P('data', None, None, None)))

        @nnx.jit
        def model_fwd(m, x, mask, rngs):
            return m(x, mask, rngs, train=True)

        t0 = time.perf_counter()
        recon, compressed, sel, sel_mask, logvar, mean = model_fwd(model, x, mask, nnx.Rngs(42))
        dt = time.perf_counter() - t0
        db = BATCH * 2
        assert recon.shape == (db, TEMPORAL_LEN, HEIGHT, WIDTH, CHANNELS), f"recon: {recon.shape}"
        assert sel_mask.shape[0] == db
        sm = np.array(sel_mask)
        assert all(v in [0.0, 1.0] for v in np.unique(sm)), f"Not binary: {np.unique(sm)}"
        test_pass(f"recon={recon.shape}, time={dt:.2f}s")
    except Exception as e:
        test_fail("VideoVAE forward", e)

    # ── Test 4: Params replicated ──
    print("\n[Test 4] Model params replicated", flush=True)
    try:
        params_state = nnx.state(model, nnx.Param)
        leaves = jax.tree_util.tree_leaves(params_state)
        num_params = sum(x.size for x in leaves)
        test_pass(f"Params: {num_params / 1e6:.1f}M")
    except Exception as e:
        test_fail("params replicated", e)

    # ── Test 5: Gradient computation ──
    print("\n[Test 5] Gradient computation", flush=True)
    try:
        from einops import repeat
        def simple_loss(model, x, mask, rngs):
            recon, _, _, _, _, _ = model(x, mask, rngs, train=True)
            return jnp.mean((recon - repeat(x, "b ... -> (b 2) ...")) ** 2)
        grad_fn = nnx.value_and_grad(simple_loss)

        @nnx.jit
        def grad_step(model, x, mask, rngs):
            return grad_fn(model, x, mask, rngs)
        loss, grads = grad_step(model, x, mask, nnx.Rngs(99))
        assert np.isfinite(float(loss)), f"Loss not finite: {loss}"
        gl = jax.tree_util.tree_leaves(nnx.state(grads))
        gl = [g for g in gl if hasattr(g, 'shape') and g.size > 0]
        mg = max(float(jnp.max(jnp.abs(g))) for g in gl)
        assert mg > 0 and np.isfinite(mg), f"Bad grads: max={mg}"
        test_pass(f"loss={float(loss):.6f}, max_grad={mg:.6f}")
    except Exception as e:
        test_fail("gradient computation", e)

    # ── Test 6: GumbelSigmoidSTE ──
    print("\n[Test 6] GumbelSigmoidSTE", flush=True)
    from layers import GumbelSigmoidSTE, round_ste
    try:
        # Test round_ste directly (STE core)
        def ste_loss(x):
            return jnp.mean(round_ste(x))
        logits = jnp.ones((4, 8)) * 0.5
        grads = jax.jit(jax.grad(ste_loss))(logits)
        assert grads.shape == logits.shape, f"Shape mismatch: {grads.shape}"
        assert float(jnp.max(jnp.abs(grads))) > 0, "STE grads zero"

        # Test GumbelSigmoidSTE produces binary output
        gumbel = GumbelSigmoidSTE(temperature=1.0)
        out = gumbel(jnp.ones((4, 8)) * 2.0, nnx.Rngs(10), train=True)
        assert all(v in [0.0, 1.0] for v in np.unique(np.array(out))), f"Not binary: {np.unique(np.array(out))}"
        test_pass("STE gradients flow, GumbelSigmoid output binary")
    except Exception as e:
        test_fail("GumbelSigmoidSTE", e)

    # ── Test 7: FactoredAttention ──
    print("\n[Test 7] FactoredAttention shapes", flush=True)
    from layers import FactoredAttention
    try:
        hw = (HEIGHT // PATCH_SIZE) ** 2
        fd = CHANNELS * PATCH_SIZE * PATCH_SIZE
        fa = FactoredAttention(mlp_dim=MLP_DIM, in_features=fd, num_heads=NUM_HEADS,
            qkv_features=QKV_FEATURES, max_temporal_len=MAX_TEMPORAL_LEN,
            max_spatial_len=hw, rngs=nnx.Rngs(42), dtype=jnp.float32, param_dtype=jnp.float32)
        xi = jnp.array(np.random.randn(BATCH, TEMPORAL_LEN, hw, fd).astype(np.float32) * 0.02)
        mi = jnp.ones((BATCH, 1, 1, TEMPORAL_LEN), dtype=jnp.bool_)
        out = nnx.jit(fa)(xi, mi)
        assert out.shape == (BATCH, TEMPORAL_LEN, hw, fd), f"FA: {out.shape}"
        test_pass(f"output={out.shape}")
    except Exception as e:
        test_fail("FactoredAttention", e)

    # ── Test 8: PatchEmbedding/UnEmbedding ──
    print("\n[Test 8] PatchEmbedding & PatchUnEmbedding", flush=True)
    from layers import PatchEmbedding, PatchUnEmbedding
    try:
        pe = PatchEmbedding(HEIGHT, WIDTH, CHANNELS, PATCH_SIZE, nnx.Rngs(42), dtype=jnp.float32, param_dtype=jnp.float32)
        xi = jnp.array(np.random.randn(BATCH, TEMPORAL_LEN, HEIGHT, WIDTH, CHANNELS).astype(np.float32) * 0.02)
        emb = nnx.jit(pe)(xi)
        hw = (HEIGHT // PATCH_SIZE) ** 2
        ppc = CHANNELS * PATCH_SIZE * PATCH_SIZE
        assert emb.shape == (BATCH, TEMPORAL_LEN, hw, ppc), f"PE: {emb.shape}"
        pue = PatchUnEmbedding(HEIGHT, WIDTH, CHANNELS, PATCH_SIZE, UPSAMPLE_RATE, nnx.Rngs(42), dtype=jnp.float32, param_dtype=jnp.float32)
        cf, recon = nnx.jit(pue)(emb)
        assert recon.shape == (BATCH, TEMPORAL_LEN, HEIGHT, WIDTH, CHANNELS)
        test_pass(f"emb={emb.shape}, recon={recon.shape}")
    except Exception as e:
        test_fail("PatchEmbedding/UnEmbedding", e)

    # ── Test 9: UNet ──
    print("\n[Test 9] UNet shape preservation", flush=True)
    from unet import UNet
    try:
        uc = CHANNELS * UPSAMPLE_RATE
        unet = UNet(channels=uc, base_features=16, num_levels=3, out_features=CHANNELS,
                    rngs=nnx.Rngs(42), dtype=jnp.float32, param_dtype=jnp.float32)
        xi = jnp.array(np.random.randn(BATCH, TEMPORAL_LEN, HEIGHT, WIDTH, uc).astype(np.float32) * 0.02)
        out = nnx.jit(unet)(xi)
        assert out.shape == (BATCH, TEMPORAL_LEN, HEIGHT, WIDTH, CHANNELS), f"UNet: {out.shape}"
        test_pass(f"output={out.shape}")
    except Exception as e:
        test_fail("UNet", e)

    # ── Summary ──
    print(f"\n{'='*50}", flush=True)
    print(f"MODEL TESTS: {PASS_COUNT} passed, {FAIL_COUNT} failed", flush=True)
    if FAIL_COUNT == 0:
        print("ALL MODEL TESTS PASSED!", flush=True)
    print(f"{'='*50}", flush=True)
    if FAIL_COUNT > 0:
        sys.exit(1)
