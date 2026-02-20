"""
Training loop tests (local CPU mode).

Usage:
    JAX_PLATFORMS=cpu JAX_NUM_CPU_DEVICES=4 python3 test_training_loop.py
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
    import optax
    import numpy as np
    import signal
    import time
    from einops import rearrange, repeat, reduce

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
    MAX_TEMPORAL_LEN, SPATIAL_COMPRESSION_RATE, UPSAMPLE_RATE = 16, 4, 2
    BATCH, TEMPORAL_LEN, LR = num_devices, 8, 1e-3

    def test_pass(name):
        global PASS_COUNT
        PASS_COUNT += 1
        print(f"  PASS: {name}", flush=True)

    def test_fail(name, err):
        global FAIL_COUNT
        FAIL_COUNT += 1
        print(f"  FAIL: {name} - {err}", flush=True)

    # ── Build model + optimizer ──
    from rl_model import VideoVAE
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

    optimizer = nnx.Optimizer(model, optax.chain(optax.clip_by_global_norm(1.0), optax.adam(LR)))
    gdef_opt, opt_state = nnx.split(optimizer)
    opt_state = jax.device_put(opt_state, replicated_sharding)
    optimizer = nnx.merge(gdef_opt, opt_state)

    hparams = {"gamma1": 0.2, "gamma2": 0.001, "gamma3": 0.0, "gamma4": 0.05,
               "max_compression_rate": 2, "magnify_negatives_rate": 100, "rl_loss_weight": 0.01}

    # ── Loss function ──
    def per_sample_mean(x):
        return jnp.mean(x, axis=tuple(range(1, x.ndim)))

    def magnify_negatives(x, rate):
        return jnp.where(x < 0, x * rate, x)

    def dummy_perceptual(params, x, target):
        return jnp.zeros(x.shape[0])

    def loss_fn(model, video, mask, original_mask, rngs, hparams, ploss_fn, vgg_params, train=True):
        recon, compressed, selection, sel_mask, logvar, mean = model(video, mask, rngs, train=train)
        out_mask = repeat(original_mask, "b t -> (b 2) t")
        seq_len = jnp.clip(reduce(out_mask, "b t -> b 1", "sum"), 1.0, None)
        vm = rearrange(out_mask, "b t -> b t 1 1 1")
        video = repeat(video, "b ... -> (b 2) ...")
        sl = rearrange(seq_len, "b 1 -> b 1 1 1 1")

        mae = per_sample_mean(reduce(jnp.abs((video - recon) * vm), "b t h w c -> b 1 h w c", "sum") / sl)
        mse = per_sample_mean(reduce(jnp.square((video - recon) * vm), "b t h w c -> b 1 h w c", "sum") / sl)
        ploss = ploss_fn(vgg_params, recon, video)

        ksm = rearrange(out_mask, "b t -> b t 1 1")
        density = reduce(sel_mask * ksm, "b t 1 1 -> b 1", "sum") / seq_len
        sel_loss = per_sample_mean(jnp.square(magnify_negatives(density - 1/hparams["max_compression_rate"], hparams["magnify_negatives_rate"])))

        sl_kl = rearrange(seq_len, "b 1 -> b 1 1 1")
        kl = per_sample_mean(0.5 * (jnp.exp(logvar) - 1 - logvar + jnp.square(mean)) * ksm / sl_kl)

        psl = mse + hparams["gamma3"] * ploss + hparams["gamma1"] * sel_loss + hparams["gamma2"] * kl + hparams["gamma4"] * mae

        pairs = rearrange(psl, "(b p) -> b p", p=2)
        m_ = rearrange(per_sample_mean(pairs), "b -> b 1")
        s_ = rearrange(jnp.std(pairs, axis=1) + 1e-6, "b -> b 1")
        disadv = (pairs - m_) / s_

        acts = rearrange(sel_mask, "(b p) t 1 1 -> b p t", p=2)
        sel2 = rearrange(selection, "(b p) t 1 1 -> b p t", p=2)
        rp = jnp.clip(jnp.abs(sel2 + acts - 1), 1e-6, 1 - 1e-6)
        probs = rp / jax.lax.stop_gradient(rp)
        rl_m = rearrange(out_mask, "(b p) t -> b p t", p=2)
        probs = jnp.where(rl_m, probs, 1.0)
        probs = reduce(probs, "b p t -> b p 1", "prod")
        rl_loss = probs * jax.lax.stop_gradient(rearrange(disadv, "b p -> b p 1"))

        loss = jnp.mean(psl) + jnp.mean(rl_loss) * hparams["rl_loss_weight"]
        return loss, {"MSE": jnp.mean(mse), "sel": jnp.mean(sel_loss), "kl": jnp.mean(kl),
                      "rl": jnp.mean(rl_loss), "mae": jnp.mean(mae), "density": density.mean()}

    def train_step(model, optimizer, video, mask, hparams, rngs, ploss_fn, vgg_params):
        om = mask.copy()
        mask = rearrange(mask, "b t -> b 1 1 t")
        (loss, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model, video, mask, om, rngs, hparams, ploss_fn, vgg_params)
        optimizer.update(grads)
        return loss, aux

    jit_train = nnx.jit(train_step, static_argnames=("ploss_fn",))

    def make_batch():
        v = jax.device_put(jnp.array(np.random.randn(BATCH, TEMPORAL_LEN, HEIGHT, WIDTH, CHANNELS).astype(np.float32) * 0.1), data_sharding)
        m = jax.device_put(jnp.ones((BATCH, TEMPORAL_LEN), dtype=jnp.bool_), NamedSharding(mesh, P('data', None)))
        return v, m

    # ── Test 1: Loss function ──
    print("\n[Test 1] Loss function produces valid scalar", flush=True)
    try:
        video, mask = make_batch()
        @nnx.jit
        def compute_loss(model, video, mask, hparams, rngs):
            om = mask.copy()
            return loss_fn(model, video, rearrange(mask, "b t -> b 1 1 t"), om, rngs, hparams, dummy_perceptual, None)
        loss, aux = compute_loss(model, video, mask, hparams, nnx.Rngs(1))
        lv = float(loss)
        assert np.isfinite(lv) and lv > 0, f"Bad loss: {lv}"
        for k in ["MSE", "sel", "kl", "rl", "mae", "density"]:
            assert np.isfinite(float(aux[k])), f"{k} not finite"
        test_pass(f"loss={lv:.4f}, MSE={float(aux['MSE']):.4f}, density={float(aux['density']):.4f}")
    except Exception as e:
        test_fail("loss function", e)

    # ── Test 2: Train step ──
    print("\n[Test 2] Train step executes correctly", flush=True)
    try:
        video, mask = make_batch()
        t0 = time.perf_counter()
        loss, aux = jit_train(model, optimizer, video, mask, hparams, nnx.Rngs(2), dummy_perceptual, None)
        dt = time.perf_counter() - t0
        assert np.isfinite(float(loss))
        test_pass(f"loss={float(loss):.4f}, time={dt:.2f}s (incl. compile)")
    except Exception as e:
        test_fail("train step", e)

    # ── Test 3: Loss decreases ──
    print("\n[Test 3] Loss decreases over training", flush=True)
    try:
        fv, fm = make_batch()
        losses = []
        for step in range(10):
            loss, aux = jit_train(model, optimizer, fv, fm, hparams, nnx.Rngs(step + 100), dummy_perceptual, None)
            losses.append(float(loss))
        first = np.mean(losses[:5])
        second = np.mean(losses[5:])
        print(f"  Losses: {[f'{l:.4f}' for l in losses]}", flush=True)
        assert second < first, f"Not decreasing: {first:.4f} -> {second:.4f}"
        test_pass(f"Loss decreased: {first:.4f} -> {second:.4f}")
    except Exception as e:
        test_fail("loss decrease", e)

    # ── Test 4: Gradient sanity ──
    print("\n[Test 4] Gradient sanity check", flush=True)
    try:
        video, mask = make_batch()
        def check_fn(model, video, mask, hparams, rngs):
            om = mask.copy()
            return loss_fn(model, video, rearrange(mask, "b t -> b 1 1 t"), om, rngs, hparams, dummy_perceptual, None)
        @nnx.jit
        def get_grads(model, video, mask, hparams, rngs):
            (loss, _), grads = nnx.value_and_grad(check_fn, has_aux=True)(model, video, mask, hparams, rngs)
            gl = jax.tree_util.tree_leaves(nnx.state(grads))
            gl = [g for g in gl if hasattr(g, 'shape') and g.size > 0]
            mg = jnp.max(jnp.stack([jnp.max(jnp.abs(g)) for g in gl]))
            nan = jnp.any(jnp.stack([jnp.any(jnp.isnan(g)) for g in gl]))
            return loss, mg, nan
        loss, mg, nan = get_grads(model, video, mask, hparams, nnx.Rngs(999))
        assert not bool(nan), "NaN in gradients"
        assert float(mg) > 0 and np.isfinite(float(mg))
        test_pass(f"max_grad={float(mg):.6f}, no NaN")
    except Exception as e:
        test_fail("gradient sanity", e)

    # ── Test 5: SIGTERM ──
    print("\n[Test 5] SIGTERM handling", flush=True)
    try:
        import distributed_train as dt
        assert not dt._SHOULD_STOP
        dt._signal_handler(signal.SIGTERM, None)
        assert dt._SHOULD_STOP
        dt._SHOULD_STOP = False
        dt._signal_handler(signal.SIGINT, None)
        assert dt._SHOULD_STOP
        assert signal.getsignal(signal.SIGTERM) == dt._signal_handler
        assert signal.getsignal(signal.SIGINT) == dt._signal_handler
        dt._SHOULD_STOP = False
        test_pass("SIGTERM/SIGINT handlers work")
    except Exception as e:
        test_fail("SIGTERM handling", e)

    # ── Test 6: Data-parallel sharding ──
    print("\n[Test 6] Data-parallel batch sharding", flush=True)
    try:
        tagged = np.arange(num_devices).reshape(num_devices, 1, 1, 1, 1).astype(np.float32)
        tagged = np.broadcast_to(tagged, (num_devices, TEMPORAL_LEN, HEIGHT, WIDTH, CHANNELS)).copy()
        sv = jax.device_put(jnp.array(tagged), data_sharding)
        assert len({s.device for s in sv.addressable_shards}) == num_devices
        full = np.array(sv)
        for d in range(num_devices):
            assert np.isclose(full[d, 0, 0, 0, 0], float(d))
        test_pass(f"Correctly sharded across {num_devices} devices")
    except Exception as e:
        test_fail("data-parallel sharding", e)

    # ── Test 7: VGG perceptual loss ──
    print("\n[Test 7] VGG perceptual loss", flush=True)
    try:
        from vgg_tests import load_vgg, get_adversarial_perceptual_loss_fn
        vgg_model, vgg_params = load_vgg()
        pfn = get_adversarial_perceptual_loss_fn(vgg_model)
        x = jnp.ones((2, TEMPORAL_LEN, HEIGHT, WIDTH, 3), dtype=jnp.bfloat16) * 0.5
        t = jnp.ones((2, TEMPORAL_LEN, HEIGHT, WIDTH, 3), dtype=jnp.bfloat16) * 0.3
        pl = jax.jit(pfn)(vgg_params, x, t)
        pv = float(jnp.mean(pl))
        assert np.isfinite(pv) and pv >= 0
        test_pass(f"perceptual_loss={pv:.4f}")
    except Exception as e:
        test_fail("VGG perceptual loss", e)

    # ── Summary ──
    print(f"\n{'='*50}", flush=True)
    print(f"TRAINING LOOP TESTS: {PASS_COUNT} passed, {FAIL_COUNT} failed", flush=True)
    if FAIL_COUNT == 0:
        print("ALL TRAINING LOOP TESTS PASSED!", flush=True)
    print(f"{'='*50}", flush=True)
    if FAIL_COUNT > 0:
        sys.exit(1)
