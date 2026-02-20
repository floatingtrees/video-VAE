"""
Distributed training for VideoVAE on TPU pods.

Supports single-host multi-device and multi-host TPU pod configurations.
Uses JAX SPMD data parallelism: model params replicated, batch sharded across devices.

Usage:
    python distributed_train.py [--run] [--model_path PATH] [--test]
"""

import os
import numpy as np
import time
import math
import shutil
import argparse

# NOTE: All JAX imports and jax.distributed.initialize() are inside __main__
# to avoid conflicts with Grain's multiprocessing workers which re-import modules.


# ---------------------------------------------------------------------------
# Hyperparameters (pure Python, no JAX dependency)
# ---------------------------------------------------------------------------
NUM_EPOCHS = 100
PER_DEVICE_BATCH_SIZE = 1
MAX_FRAMES = 32
RESIZE = (256, 256)
LEARNING_RATE = 2e-5
DECAY_STEPS = 1_000_000
GAMMA1 = 0.2
GAMMA2 = 0.001
GAMMA3 = 0.1
GAMMA4 = 0.05
MAGNIFY_NEGATIVES_RATE = 100
NEGATIVE_PENALTY_TRAINING_STEPS = 2000
RLLossWeight = 0.01
max_compression_rate = 2
DATA_DIR = os.path.expanduser("~/data/videos/videos")
VIDEO_SAVE_DIR = "outputs"
model_save_path = os.path.expanduser("~/video_vae_distributed_saves/")
SHUFFLE = True
NUM_WORKERS = 4
PREFETCH_SIZE = 16
DROP_REMAINDER = True
SEED = 0


if __name__ == "__main__":
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".99"

    import jax
    # Initialize distributed JAX BEFORE any device access.
    # On TPU pods this auto-detects coordinator, process id, and peer count.
    jax.distributed.initialize()

    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from flax import nnx
    import optax
    from einops import rearrange, repeat, reduce
    from jaxtyping import Float, Array
    from rl_model import VideoVAE
    from dataloader import create_batched_dataloader, batch_to_video
    from vgg_tests import get_adversarial_perceptual_loss_fn, load_vgg
    import orbax.checkpoint as ocp

    # -------------------------------------------------------------------
    # Distributed topology
    # -------------------------------------------------------------------
    num_devices = jax.device_count()
    local_devices = jax.local_device_count()
    process_index = jax.process_index()
    num_processes = jax.process_count()

    if process_index == 0:
        print(f"Distributed setup: {num_devices} total devices, "
              f"{local_devices} local, {num_processes} processes")

    mesh = jax.make_mesh((num_devices,), ('data',))
    replicated_sharding = NamedSharding(mesh, P())
    data_sharding = NamedSharding(mesh, P('data'))

    # -------------------------------------------------------------------
    # Parse args
    # -------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="Enable wandb logging")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--per_device_batch_size", type=int, default=PER_DEVICE_BATCH_SIZE)
    parser.add_argument("--max_frames", type=int, default=MAX_FRAMES)
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    args = parser.parse_args()

    PER_DEVICE_BATCH_SIZE = args.per_device_batch_size
    LOCAL_BATCH_SIZE = PER_DEVICE_BATCH_SIZE * local_devices
    GLOBAL_BATCH_SIZE = LOCAL_BATCH_SIZE * num_processes
    MAX_FRAMES = args.max_frames
    DATA_DIR = args.data_dir
    WARMUP_STEPS = int(20000 / math.sqrt(GLOBAL_BATCH_SIZE))

    TRAINING_RUN = args.run
    IS_TEST = args.test

    if TRAINING_RUN and process_index == 0:
        import wandb
        wandb.init(project="video-vae-distributed")

    # Reset checkpoint directory (process 0 only)
    if process_index == 0:
        if os.path.exists(model_save_path):
            shutil.rmtree(model_save_path)
        os.makedirs(model_save_path, exist_ok=True)
    jax.experimental.multihost_utils.sync_global_devices("reset_dir")

    if process_index == 0:
        print(f"Per-device batch: {PER_DEVICE_BATCH_SIZE}, "
              f"Local batch: {LOCAL_BATCH_SIZE}, "
              f"Global batch: {GLOBAL_BATCH_SIZE}")

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------
    def per_sample_mean(x):
        return jnp.mean(x, axis=tuple(range(1, x.ndim)))

    def magnify_negatives(x, magnification_rate):
        return jnp.where(x < 0, x * magnification_rate, x)

    def shard_batch(batch):
        sharded = {}
        for key, val in batch.items():
            ndim = val.ndim
            spec = P('data', *([None] * (ndim - 1)))
            s = NamedSharding(mesh, spec)
            sharded[key] = jax.make_array_from_process_local_data(s, val)
        return sharded

    # -------------------------------------------------------------------
    # Loss function (same logic as rl_nonadversarial.py)
    # -------------------------------------------------------------------
    def loss_fn(model, video, mask, original_mask, rngs,
                hparams, perceptual_loss_fn, vgg_params, train=True):

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

        raw_probs_masked = jnp.where(rl_mask, raw_probs, 1.0)
        raw_trajectory_probs = reduce(raw_probs_masked, "b p time -> b p 1", "prod")

        probs = reduce(probs, "b p time -> b p 1", "prod")
        disadvantages = rearrange(disadvantages, "b p -> b p 1")
        rl_loss = probs * jax.lax.stop_gradient(disadvantages)

        loss = jnp.mean(per_sample_loss) + jnp.mean(rl_loss) * hparams["rl_loss_weight"]

        return loss, {
            "MSE": jnp.mean(per_sample_error),
            "perceptual_loss": jnp.mean(perceptual_loss),
            "selection_loss": jnp.mean(selection_loss),
            "kl_loss": jnp.mean(kl_loss),
            "reconstruction": reconstruction,
            "kept_frame_density": kept_frame_density.mean(),
            "mean_trajectory_prob": jnp.mean(raw_trajectory_probs),
            "rl_loss": jnp.mean(rl_loss),
            "per_sample_MAE": jnp.mean(per_sample_MAE),
        }

    def train_step(model, optimizer, video, mask, hparams,
                   rngs, perceptual_loss_fn, vgg_params):
        original_mask = mask.copy()
        # FactoredAttention handles hw expansion internally,
        # so just reshape mask to (b, 1, 1, t) for the model
        mask = rearrange(mask, "b time -> b 1 1 time")
        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (loss, aux), grads = grad_fn(
            model, video, mask, original_mask, rngs, hparams,
            perceptual_loss_fn, vgg_params)
        optimizer.update(grads)
        return loss, aux

    def eval_step(model, video, mask, hparams,
                  rngs, perceptual_loss_fn, vgg_params):
        original_mask = mask.copy()
        mask = rearrange(mask, "b time -> b 1 1 time")
        loss, aux = loss_fn(
            model, video, mask, original_mask, rngs, hparams,
            perceptual_loss_fn, vgg_params, train=True)
        return loss, aux

    # -------------------------------------------------------------------
    # Checkpoint helpers
    # -------------------------------------------------------------------
    def save_checkpoint(model, optimizer, path):
        state = {"model": nnx.state(model), "optimizer": nnx.state(optimizer)}
        ocp.StandardCheckpointer().save(path, state)

    def load_checkpoint_fn(model, optimizer, path):
        abstract_state = {
            "model": jax.tree.map(ocp.utils.to_shape_dtype_struct, nnx.state(model)),
            "optimizer": jax.tree.map(ocp.utils.to_shape_dtype_struct, nnx.state(optimizer)),
        }
        restored = ocp.StandardCheckpointer().restore(path, abstract_state)
        nnx.update(model, restored["model"])
        nnx.update(optimizer, restored["optimizer"])

    # -------------------------------------------------------------------
    # Build model
    # -------------------------------------------------------------------
    height, width = RESIZE
    patch_size = 16
    model = VideoVAE(
        height=height, width=width, channels=3, patch_size=patch_size,
        encoder_depth=9, decoder_depth=12, mlp_dim=1536, num_heads=8,
        qkv_features=512, max_temporal_len=64,
        spatial_compression_rate=8, unembedding_upsample_rate=4,
        rngs=nnx.Rngs(2),
    )

    params_state = nnx.state(model, nnx.Param)
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(params_state))
    if process_index == 0:
        print(f"Model parameters: {num_params / 1e6:.1f}M")

    # -------------------------------------------------------------------
    # Optimizer
    # -------------------------------------------------------------------
    schedule_fn = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        decay_steps=DECAY_STEPS,
        end_value=LEARNING_RATE / 10,
    )
    optimizer_def = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=schedule_fn),
    )
    optimizer = nnx.Optimizer(model, optimizer_def)

    # -------------------------------------------------------------------
    # Replicate model + optimizer state across all devices
    # -------------------------------------------------------------------
    gdef, state = nnx.split(model)
    state = jax.device_put(state, replicated_sharding)
    model = nnx.merge(gdef, state)

    gdef_opt, opt_state = nnx.split(optimizer)
    opt_state = jax.device_put(opt_state, replicated_sharding)
    optimizer = nnx.merge(gdef_opt, opt_state)

    hparams = {
        "gamma1": GAMMA1,
        "gamma2": GAMMA2,
        "gamma3": GAMMA3,
        "gamma4": GAMMA4,
        "max_compression_rate": max_compression_rate,
        "magnify_negatives_rate": MAGNIFY_NEGATIVES_RATE,
        "rl_loss_weight": RLLossWeight,
    }

    if args.model_path is not None:
        load_checkpoint_fn(model, optimizer, args.model_path)
        hparams["max_compression_rate"] = 100000
        SEED = 42

    rngs = nnx.Rngs(3)

    # -------------------------------------------------------------------
    # VGG perceptual loss (replicated)
    # -------------------------------------------------------------------
    vgg_model, vgg_params = load_vgg()
    vgg_params = jax.device_put(vgg_params, replicated_sharding)
    perceptual_loss_fn = get_adversarial_perceptual_loss_fn(vgg_model)

    # -------------------------------------------------------------------
    # JIT compile
    # -------------------------------------------------------------------
    jit_train_step = nnx.jit(train_step, static_argnames=("perceptual_loss_fn",))
    jit_eval_step = nnx.jit(eval_step, static_argnames=("perceptual_loss_fn",))

    # -------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------
    global_step = 0
    start = time.perf_counter()

    if process_index == 0:
        os.makedirs(os.path.join(VIDEO_SAVE_DIR, "train"), exist_ok=True)
        os.makedirs(os.path.join(VIDEO_SAVE_DIR, "eval"), exist_ok=True)

    for epoch in range(NUM_EPOCHS):
        if process_index == 0:
            os.makedirs(os.path.join(VIDEO_SAVE_DIR, f"train/epoch{epoch}"), exist_ok=True)
            os.makedirs(os.path.join(VIDEO_SAVE_DIR, f"eval/epoch{epoch}"), exist_ok=True)

        max_frames_cap = 64
        min_batch_size = 1
        max_epoch_multiplier = min(
            int(math.log2(max(LOCAL_BATCH_SIZE / min_batch_size, 1))),
            int(math.log2(max(max_frames_cap / MAX_FRAMES, 1))) - 1
                if max_frames_cap > MAX_FRAMES else 0,
        )
        max_epoch_multiplier = max(max_epoch_multiplier, 0)
        epoch_multiplier = min(epoch, max_epoch_multiplier)
        effective_batch_size = max(LOCAL_BATCH_SIZE // (2 ** epoch_multiplier), 1)
        effective_max_frames = MAX_FRAMES * (2 ** epoch_multiplier)

        # Ensure divisible by local device count
        if effective_batch_size % local_devices != 0:
            effective_batch_size = max(
                ((effective_batch_size + local_devices - 1) // local_devices) * local_devices,
                local_devices,
            )

        if process_index == 0:
            print(f"\nEpoch {epoch}: effective_batch={effective_batch_size} "
                  f"(global={effective_batch_size * num_processes}), "
                  f"effective_max_frames={effective_max_frames}")

        train_dataloader = create_batched_dataloader(
            base_dir=DATA_DIR,
            batch_size=effective_batch_size,
            max_frames=effective_max_frames,
            resize=RESIZE,
            shuffle=SHUFFLE,
            num_workers=NUM_WORKERS,
            prefetch_size=PREFETCH_SIZE,
            drop_remainder=True,
            seed=SEED + epoch,
        )

        total_videos = 40000  # approximate
        steps_per_epoch = total_videos // (effective_batch_size * num_processes)

        for i, batch in enumerate(train_dataloader):
            if i > steps_per_epoch:
                break
            if i > NEGATIVE_PENALTY_TRAINING_STEPS:
                hparams["max_compression_rate"] = 10000

            # Shard batch across all devices
            global_batch = shard_batch(batch)
            video = global_batch["video"].astype(jnp.bfloat16)
            mask = global_batch["mask"].astype(jnp.bool_)

            loss, aux = jit_train_step(
                model, optimizer, video, mask, hparams, rngs,
                perceptual_loss_fn, vgg_params)

            global_step += 1

            # Logging (process 0 only)
            if process_index == 0 and i % 50 == 0:
                elapsed = time.perf_counter() - start
                print(f"  Step {i}: loss={float(loss):.4f} "
                      f"MSE={float(aux['MSE']):.4f} "
                      f"perceptual={float(aux['perceptual_loss']):.4f} "
                      f"sel={float(aux['selection_loss']):.4f} "
                      f"kl={float(aux['kl_loss']):.4f} "
                      f"density={float(aux['kept_frame_density']):.4f} "
                      f"rl={float(aux['rl_loss']):.4f} "
                      f"MAE={float(aux['per_sample_MAE']):.4f} "
                      f"time={elapsed:.1f}s")

            if process_index == 0 and i % 500 == 499:
                recon_local = np.array(aux["reconstruction"][:PER_DEVICE_BATCH_SIZE])
                recon_batch = {"video": recon_local, "mask": np.array(mask[:PER_DEVICE_BATCH_SIZE])}
                batch_to_video(recon_batch, os.path.join(
                    VIDEO_SAVE_DIR, f"train/epoch{epoch}/video_{i}_latent.mp4"), fps=30.0)
                local_batch_np = {k: np.array(v[:PER_DEVICE_BATCH_SIZE])
                                  for k, v in global_batch.items()}
                batch_to_video(local_batch_np, os.path.join(
                    VIDEO_SAVE_DIR, f"train/epoch{epoch}/video_{i}_original.mp4"), fps=30.0)

            if TRAINING_RUN and process_index == 0:
                wandb.log({
                    "train_loss": float(loss),
                    "train_MSE": float(aux["MSE"]),
                    "train_perceptual_loss": float(aux["perceptual_loss"]),
                    "train_Selection Loss": float(aux["selection_loss"]),
                    "train_KL Loss": float(aux["kl_loss"]),
                    "train_time": time.perf_counter() - start,
                    "kept_frame_density": float(aux["kept_frame_density"]),
                    "mean_trajectory_prob": float(aux["mean_trajectory_prob"]),
                    "train_rl_loss": float(aux["rl_loss"]),
                    "train_per_sample_MAE": float(aux["per_sample_MAE"]),
                    "effective_batch_size": effective_batch_size * num_processes,
                    "effective_max_frames": effective_max_frames,
                    "global_step": global_step,
                })

        # Save checkpoint (process 0 only)
        if process_index == 0:
            save_checkpoint(model, optimizer, f"{model_save_path}/checkpoint_{epoch}")
            print(f"Saved checkpoint for epoch {epoch}")
        jax.experimental.multihost_utils.sync_global_devices(f"checkpoint_epoch_{epoch}")

    if process_index == 0:
        print("Training complete!")
