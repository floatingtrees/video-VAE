"""
Distributed training for VideoVAE on TPU pods.

Supports single-host multi-device and multi-host TPU pod configurations.
Uses JAX SPMD data parallelism: model params replicated, batch sharded across devices.

Designed for v6e-16 (4 workers x 4 chips = 16 devices).

Usage:
    python distributed_train.py [--model_path PATH] [--test]
"""

import os
import numpy as np
import time
import math
import shutil
import argparse
import signal
import sys

# NOTE: All JAX imports and jax.distributed.initialize() are inside __main__
# to avoid conflicts with Grain's multiprocessing workers which re-import modules.


# ---------------------------------------------------------------------------
# Hyperparameters (pure Python, no JAX dependency)
# ---------------------------------------------------------------------------
NUM_EPOCHS = 100
PER_DEVICE_BATCH_SIZE = 1
MAX_FRAMES = 32
RESIZE = (256, 256)
LEARNING_RATE = 6e-5
DECAY_STEPS = 1_000_000
GAMMA1 = 0.2
GAMMA2 = 0.01
GAMMA3 = 0.1
GAMMA4 = 0.05
MAGNIFY_NEGATIVES_RATE = 100
NEGATIVE_PENALTY_TRAINING_STEPS = 2000
RLLossWeight = 0.01
max_compression_rate = 2
DATA_DIR = os.path.expanduser("~/data/videos")
RUN_TIMESTAMP = int(time.time())
GCS_RUN_DIR = f"gs://tpus-487818-checkpoints/run{RUN_TIMESTAMP}"
VIDEO_SAVE_DIR = f"{GCS_RUN_DIR}/images"
model_save_path = f"{GCS_RUN_DIR}/perceptual_loss_model"
SHUFFLE = True
NUM_WORKERS = 4
PREFETCH_SIZE = 16
DROP_REMAINDER = True
SEED = 0


# ---------------------------------------------------------------------------
# SIGTERM / SIGINT handling for spot instances
# ---------------------------------------------------------------------------
_SHOULD_STOP = False

def _signal_handler(signum, frame):
    global _SHOULD_STOP
    sig_name = signal.Signals(signum).name
    print(f"\n[Worker] Received {sig_name} - will stop after current step and save checkpoint.", flush=True)
    _SHOULD_STOP = True

signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)


if __name__ == "__main__":
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".99"

    print(f"[{os.uname().nodename}] Starting distributed_train.py...", flush=True)
    import jax
    #jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
    print(f"[{os.uname().nodename}] JAX imported, initializing distributed...", flush=True)
    # Initialize distributed JAX BEFORE any device access.
    # On TPU pods this auto-detects coordinator, process id, and peer count.
    jax.distributed.initialize()
    print(f"[{os.uname().nodename}] Distributed initialized! Process {jax.process_index()}/{jax.process_count()}", flush=True)

    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from flax import nnx
    import optax
    from einops import rearrange, repeat, reduce
    from jaxtyping import Float, Array
    from rl_model import VideoVAE
    from dataloader import create_batched_dataloader, batch_to_video, VideoDataSource
    from vgg_tests import get_adversarial_perceptual_loss_fn, load_vgg
    import orbax.checkpoint as ocp
    os.environ.setdefault("WANDB_API_KEY", "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")
    import wandb

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

    IS_TEST = args.test

    if process_index == 0:
        print(f"Checkpoint save path: {model_save_path}")
    if process_index == 0:
        print(f"Per-device batch: {PER_DEVICE_BATCH_SIZE}, "
              f"Local batch: {LOCAL_BATCH_SIZE}, "
              f"Global batch: {GLOBAL_BATCH_SIZE}")
        print(f"Data dir: {DATA_DIR}")
        print(f"Num devices: {num_devices}, Num processes: {num_processes}, "
              f"Local devices: {local_devices}")

    # -------------------------------------------------------------------
    # Wandb (process 0 only)
    # -------------------------------------------------------------------
    if process_index == 0:
        wandb.init(
            project="distributed-video-vae",
            config={
                "num_epochs": NUM_EPOCHS,
                "per_device_batch_size": PER_DEVICE_BATCH_SIZE,
                "local_batch_size": LOCAL_BATCH_SIZE,
                "global_batch_size": GLOBAL_BATCH_SIZE,
                "max_frames": MAX_FRAMES,
                "learning_rate": LEARNING_RATE,
                "decay_steps": DECAY_STEPS,
                "warmup_steps": WARMUP_STEPS,
                "gamma1_selection": GAMMA1,
                "gamma2_kl": GAMMA2,
                "gamma3_perceptual": GAMMA3,
                "gamma4_mae": GAMMA4,
                "rl_loss_weight": RLLossWeight,
                "max_compression_rate": max_compression_rate,
                "magnify_negatives_rate": MAGNIFY_NEGATIVES_RATE,
                "num_devices": num_devices,
                "num_processes": num_processes,
                "resize": RESIZE,
                "seed": SEED,
            },
        )

    # Don't use sync_global_devices here — wandb.init() can trigger internal
    # JAX collectives on process 0 that cross with the barrier on other workers.
    # A simple sleep lets wandb finish; workers hard-sync at the first real
    # JAX collective in training anyway.
    if process_index != 0:
        import time
        time.sleep(10)
    if process_index == 0:
        print("Wandb init complete, proceeding.")

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

        reconstruction, compressed, selection, selection_mask, variance, mean = \
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
        selection_loss = per_sample_mean(jnp.abs(
            magnify_negatives(density_diff, hparams["magnify_negatives_rate"])))

        sl_kl = rearrange(sequence_lengths, "b 1 -> b 1 1 1")
        kl_loss = per_sample_mean(
            0.5 * (variance - 1 - jnp.log(variance) + jnp.square(mean)) * ksm / sl_kl)

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


        rl_mask = rearrange(output_mask, "(b p) time -> b p time", p=2)

        # Inverse frequency weighting: balance gradient signal between kept/dropped frames
        rl_mask_f = rl_mask.astype(actions.dtype)
        num_kept = jax.lax.stop_gradient(reduce(actions * rl_mask_f, "b p time -> b p 1", "sum"))
        num_dropped = jax.lax.stop_gradient(reduce((1 - actions) * rl_mask_f, "b p time -> b p 1", "sum"))
        num_kept = jnp.clip(num_kept, 1.0)
        num_dropped = jnp.clip(num_dropped, 1.0)
        target_kept = jax.lax.stop_gradient(
            reduce(rl_mask_f, "b p time -> b p 1", "sum")) / 2 # 16 for length 32, etc
        frame_weight = jnp.where(actions > 0.5, target_kept / num_kept, target_kept / num_dropped)


        raw_probs_masked = jnp.where(rl_mask, raw_probs, 1.0)
        raw_trajectory_probs = reduce(raw_probs_masked, "b p time -> b p 1", "prod")

        disadvantages = rearrange(disadvantages, "b p -> b p 1")

        log_probs = jnp.log(raw_probs) - jax.lax.stop_gradient(jnp.log(raw_probs)) 
        weighted_log_probs = log_probs * jax.lax.stop_gradient(frame_weight)
        weighted_log_probs = jnp.where(rl_mask, weighted_log_probs, 0.0) 
        trajectory_score = reduce(weighted_log_probs, "b p time -> b p 1", "sum") 
        rl_loss = trajectory_score * jax.lax.stop_gradient(disadvantages) 

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
        # Convert to numpy to bypass orbax's JaxArrayHandler which has a
        # set_mesh context manager bug in orbax 0.11.33 + JAX 0.6.2.
        state = jax.tree.map(lambda x: np.array(x), state)
        ckptr = ocp.StandardCheckpointer()
        ckptr.save(path, state)
        ckptr.wait_until_finished()

    def load_checkpoint_fn(model, optimizer, path):
        abstract_state = {
            "model": jax.tree.map(ocp.utils.to_shape_dtype_struct, nnx.state(model)),
            "optimizer": jax.tree.map(ocp.utils.to_shape_dtype_struct, nnx.state(optimizer)),
        }
        # Use the handler directly to bypass Checkpointer's completeness
        # check, which fails when loading a single-process checkpoint in a
        # multi-process environment.
        if process_index == 0:
            from etils import epath
            handler = ocp.StandardCheckpointHandler()
            restored = handler.restore(
                epath.Path(path),
                args=ocp.args.StandardRestore(abstract_state),
            )
        else:
            restored = jax.tree.map(lambda x: np.zeros(x.shape, dtype=x.dtype), abstract_state)
        # Broadcast from process 0 to all
        restored = jax.experimental.multihost_utils.broadcast_one_to_all(restored)
        nnx.update(model, restored["model"])
        nnx.update(optimizer, restored["optimizer"])

    # -------------------------------------------------------------------
    # Build model
    # -------------------------------------------------------------------
    height, width = RESIZE
    patch_size = 16
    model = VideoVAE(
        height=height, width=width, channels=3, patch_size=patch_size,
        encoder_depth=1 if IS_TEST else 9, decoder_depth=1 if IS_TEST else 12, mlp_dim=1536, num_heads=8,
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
    # -------------------------------------------------------------------
    # Replicate model state across all devices, then create optimizer
    # -------------------------------------------------------------------
    gdef, state = nnx.split(model)
    state = jax.device_put(state, replicated_sharding)
    model = nnx.merge(gdef, state)

    optimizer = nnx.Optimizer(model, optimizer_def)
    print(f"OPTIMIZER: {optimizer.model is model}")

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

    LOCAL_TMP_VIDEO_DIR = "/tmp/video_vae_videos"
    if process_index == 0:
        os.makedirs(LOCAL_TMP_VIDEO_DIR, exist_ok=True)

    def save_video_to_gcs(batch_data, gcs_path, fps=30.0):
        """Save video locally then upload to GCS."""
        import subprocess
        local_path = os.path.join(LOCAL_TMP_VIDEO_DIR, os.path.basename(gcs_path))
        batch_to_video(batch_data, local_path, fps=fps)
        subprocess.run(["gcloud", "storage", "cp", local_path, gcs_path, "--quiet"], check=True)
        os.remove(local_path)

    for epoch in range(NUM_EPOCHS):
        if _SHOULD_STOP:
            if process_index == 0:
                print("SIGTERM received before epoch start, saving and exiting.")
            save_checkpoint(model, optimizer, f"{model_save_path}/checkpoint_sigterm")
            break

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

        total_videos = len(VideoDataSource(DATA_DIR))
        steps_per_epoch = total_videos // (effective_batch_size * num_processes)

        for i, batch in enumerate(train_dataloader):
            if i % 50 == 0:
                params = nnx.state(model, nnx.Param)
                param_norm = sum(float(jnp.linalg.norm(x)) for x in jax.tree_util.tree_leaves(params))
                if process_index == 0:
                    print(f"  param_norm={param_norm:.4f}")


            if _SHOULD_STOP:
                if process_index == 0:
                    print(f"SIGTERM received at step {i}, saving checkpoint and exiting.")
                save_checkpoint(model, optimizer,
                                f"{model_save_path}/checkpoint_sigterm_e{epoch}_s{i}")
                break

            if i > steps_per_epoch:
                break
            if IS_TEST and i >= 5:
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

            if i % 500 == 0:
                print(f"  [worker {process_index}] heartbeat step={i} global_step={global_step}", flush=True)

            # Logging (process 0 only)
            if process_index == 0 and i % 50 == 0:
                elapsed = time.perf_counter() - start
                log_dict = {
                    "loss": float(loss),
                    "MSE": float(aux["MSE"]),
                    "perceptual_loss": float(aux["perceptual_loss"]),
                    "selection_loss": float(aux["selection_loss"]),
                    "kl_loss": float(aux["kl_loss"]),
                    "kept_frame_density": float(aux["kept_frame_density"]),
                    "mean_trajectory_prob": float(aux["mean_trajectory_prob"]),
                    "rl_loss": float(aux["rl_loss"]),
                    "MAE": float(aux["per_sample_MAE"]),
                    "epoch": epoch,
                    "step_in_epoch": i,
                    "global_step": global_step,
                    "elapsed_time": elapsed,
                    "effective_batch_size": effective_batch_size,
                    "effective_max_frames": effective_max_frames,
                    "learning_rate": float(schedule_fn(global_step)),
                }
                wandb.log(log_dict, step=global_step)
                print(f"  Step {i}: loss={log_dict['loss']:.4f} "
                      f"MSE={log_dict['MSE']:.4f} "
                      f"perceptual={log_dict['perceptual_loss']:.4f} "
                      f"sel={log_dict['selection_loss']:.4f} "
                      f"kl={log_dict['kl_loss']:.4f} "
                      f"density={log_dict['kept_frame_density']:.4f} "
                      f"rl={log_dict['rl_loss']:.4f} "
                      f"MAE={log_dict['MAE']:.4f} "
                      f"lr={log_dict['learning_rate']:.2e} "
                      f"time={elapsed:.1f}s "
                      f"global_step={global_step}", flush=True)

            if i % (3 if IS_TEST else 500) == (2 if IS_TEST else 499):
                # All workers materialize arrays to match any implicit collectives
                # (np.array on sharded JAX arrays can trigger all-gathers)
                recon_local = np.array(aux["reconstruction"][:PER_DEVICE_BATCH_SIZE])
                mask_local = np.array(mask[:PER_DEVICE_BATCH_SIZE])
                batch_local = {k: np.array(v[:PER_DEVICE_BATCH_SIZE])
                               for k, v in global_batch.items()}
                if process_index == 0:
                    try:
                        recon_batch = {"video": recon_local, "mask": mask_local}
                        save_video_to_gcs(recon_batch,
                            f"{VIDEO_SAVE_DIR}/video_e{epoch}_s{i}_latent.mp4", fps=30.0)
                        save_video_to_gcs(batch_local,
                            f"{VIDEO_SAVE_DIR}/video_e{epoch}_s{i}_original.mp4", fps=30.0)
                        print(f"  Saved videos at step {i}", flush=True)
                    except Exception as e:
                        print(f"  WARNING: Video save failed at step {i}: {e}", flush=True)
                # Barrier so no worker races ahead during process 0's I/O
                jax.experimental.multihost_utils.sync_global_devices(f"video_save_e{epoch}_s{i}")

            if global_step % (3 if IS_TEST else 10000) == 0:
                save_checkpoint(model, optimizer,
                                f"{model_save_path}/checkpoint_step_{global_step}")
                if process_index == 0:
                    print(f"Saved checkpoint at global_step {global_step}", flush=True)

        if _SHOULD_STOP:
            break

        # Save checkpoint (orbax coordinates internally, all processes must call)
        save_checkpoint(model, optimizer, f"{model_save_path}/checkpoint_{epoch}")
        if process_index == 0:
            print(f"Saved checkpoint for epoch {epoch}")

    if process_index == 0:
        wandb.finish()
        print("Training complete!")
