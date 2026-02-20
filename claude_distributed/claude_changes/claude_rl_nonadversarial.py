import os
import sys
import math
import shutil

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".99"

import jax

# ── Distributed initialization ────────────────────────────────────────────────
# Must happen before ANY jax.devices() / device operations.
#
# Modes (set by claude_distributed_run.sh via env vars):
#   Single device      : no special env vars → skip init
#   Local multi-process: JAX_COORDINATOR_ADDRESS + JAX_NUM_PROCESSES + JAX_PROCESS_ID
#   TPU cluster        : TPU_NAME or CLOUD_TPU_TASK_ID already in env → auto-init
#
# Reference: https://docs.jax.dev/en/latest/multi_process.html
# ─────────────────────────────────────────────────────────────────────────────
_is_tpu        = bool(os.environ.get("TPU_NAME") or os.environ.get("CLOUD_TPU_TASK_ID"))
_num_processes = int(os.environ.get("JAX_NUM_PROCESSES", "1"))
_coordinator   = os.environ.get("JAX_COORDINATOR_ADDRESS", "")
_process_id    = int(os.environ.get("JAX_PROCESS_ID", "0"))

if _is_tpu:
    jax.distributed.initialize()
elif _num_processes > 1 and _coordinator:
    jax.distributed.initialize(
        coordinator_address=_coordinator,
        num_processes=_num_processes,
        process_id=_process_id,
    )
# else: single device — no initialization needed

from jax.sharding import NamedSharding, PartitionSpec as P
import numpy as np

# PYTHONPATH is set by claude_distributed_run.sh to include the parent
# (claude_distributed/) directory, so these resolve correctly.
from dataloader import create_batched_dataloader, batch_to_video
from rl_model import VideoVAE
from flax import nnx
import jax.numpy as jnp
import optax
import wandb
import time
from jaxtyping import Float, Array
from einops import rearrange, repeat, reduce
from vgg_tests import get_adversarial_perceptual_loss_fn, load_vgg
import orbax.checkpoint as ocp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Inlined from model_loader.py (avoids its broken top-level `from model import`) ──
def load_checkpoint(model, optimizer, path):
    abstract_state = {
        "model":     jax.tree.map(ocp.utils.to_shape_dtype_struct, nnx.state(model)),
        "optimizer": jax.tree.map(ocp.utils.to_shape_dtype_struct, nnx.state(optimizer)),
    }
    restored = ocp.StandardCheckpointer().restore(path, abstract_state)
    nnx.update(model,     restored["model"])
    nnx.update(optimizer, restored["optimizer"])


def save_checkpoint(model, optimizer, path):
    state = {
        "model":     nnx.state(model),
        "optimizer": nnx.state(optimizer),
    }
    ocp.StandardCheckpointer().save(path, state)


def reset_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"Deleted: {path}")
    os.makedirs(path, exist_ok=True)
    print(f"Created/Reloaded: {path}")


# ── Loss hyperparameter constants (unchanged from original) ──────────────────
GAMMA1 = 0.2    # selection loss weight
GAMMA2 = 0.001  # KL weight
GAMMA3 = 0.1    # perceptual loss weight
GAMMA4 = 0.05   # MAE loss weight
MAGNIFY_NEGATIVES_RATE = 100
NEGATIVE_PENALTY_TRAINING_STEPS = 2000
RLLossWeight = 0.01
max_compression_rate = 2

SHUFFLE = True
NUM_WORKERS = 4
PREFETCH_SIZE = 16
DROP_REMAINDER = True
SEED = 0
DECAY_STEPS = 1_000_000
NUM_EPOCHS = 100


def per_sample_mean(x: Float[Array, "b ..."]):
    return jnp.mean(x, axis=tuple(range(1, x.ndim)))


def magnify_negatives(x, magnification_rate: float):
    return jnp.where(x < 0, x * magnification_rate, x)


def loss_fn(model: nnx.Module, video: Float[Array, "b time height width channels"],
    mask: Float[Array, "b 1 1 time"], original_mask: Float[Array, "b time"],
    rngs: nnx.Rngs, hparams: dict, perceptual_loss_fn, vgg_params, train: bool = True):
    reconstruction, compressed_representation, selection, selection_mask, logvar, mean = model(video, mask, rngs, train=train)
    output_mask = repeat(original_mask, "b time-> (b 2) time")
    sequence_lengths = reduce(output_mask, "b time -> b 1", "sum")
    sequence_lengths: Float[Array, "b"] = jnp.clip(sequence_lengths, 1.0, None)

    video_shaped_mask = rearrange(output_mask, "b time -> b time 1 1 1")
    video = repeat(video, "b ... -> (b 2) ...")

    sequence_lengths_reshaped = rearrange(sequence_lengths, "b 1 -> b 1 1 1 1")

    masked_abs_error = jnp.abs((video - reconstruction) * video_shaped_mask)
    sequence_lengths_reshaped = rearrange(sequence_lengths, "b 1 -> b 1 1 1 1")
    frame_reduced_MAE = reduce(masked_abs_error, "b time h w c -> b 1 h w c", "sum") / sequence_lengths_reshaped
    per_sample_MAE = per_sample_mean(frame_reduced_MAE)

    masked_squared_error = jnp.square((video - reconstruction) * video_shaped_mask)
    frame_reduced_error = reduce(masked_squared_error, "b time h w c -> b 1 h w c", "sum") / sequence_lengths_reshaped
    per_sample_error = per_sample_mean(frame_reduced_error)

    perceptual_loss: Float[Array, "b"] = perceptual_loss_fn(vgg_params, reconstruction, video)

    kl_and_selection_mask = rearrange(output_mask, "b time -> b time 1 1")

    selection_sum = reduce(selection_mask * kl_and_selection_mask, "b time 1 1 -> b 1", "sum")

    kept_frame_density = selection_sum / sequence_lengths
    density_compression_difference = kept_frame_density - (1 / hparams["max_compression_rate"])
    selection_loss = per_sample_mean(jnp.square(magnify_negatives(density_compression_difference, hparams["magnify_negatives_rate"])))

    sequence_lengths_reshaped_kl = rearrange(sequence_lengths, "b 1 -> b 1 1 1")
    kl_loss = 0.5 * (jnp.exp(logvar) - 1 - logvar + jnp.square(mean)) * kl_and_selection_mask / sequence_lengths_reshaped_kl
    kl_loss = per_sample_mean(kl_loss)

    per_sample_loss = (per_sample_error
        + hparams["gamma3"] * perceptual_loss
        + hparams["gamma1"] * selection_loss
        + hparams["gamma2"] * kl_loss
        + hparams["gamma4"] * per_sample_MAE)
    pairs = rearrange(per_sample_loss, "(b p) -> b p", p=2)
    means = rearrange(per_sample_mean(pairs), "b -> b 1")
    stds = rearrange(jnp.std(pairs, axis=1) + 1e-6, "b -> b 1")
    disadvantages = (pairs - means) / stds
    actions = rearrange(selection_mask, "(b p) time 1 1 -> b p time", p=2)
    selection = rearrange(selection, "(b p) time 1 1 -> b p time", p=2)
    raw_probs = jnp.clip(jnp.abs(selection + actions - 1), 1e-6, 1.0 - 1e-6)
    probs: Float[Array, "b p time"] = raw_probs / jax.lax.stop_gradient(raw_probs)
    rl_mask = rearrange(output_mask, "(b p) time -> b p time", p=2)
    probs = jnp.where(rl_mask, probs, 1.0)

    raw_probs_masked = jnp.where(rl_mask, raw_probs, 1.0)
    raw_trajectory_probs = reduce(raw_probs_masked, "b p time -> b p 1", "prod")

    probs = reduce(probs, "b p time -> b p 1", "prod")
    disadvantages = rearrange(disadvantages, "b p -> b p 1")
    rl_loss = probs * jax.lax.stop_gradient(disadvantages)
    loss = jnp.mean(per_sample_loss) + jnp.mean(rl_loss) * hparams["rl_loss_weight"]

    return loss, {
        "MSE":               jnp.mean(per_sample_error),
        "perceptual_loss":   jnp.mean(perceptual_loss),
        "selection_loss":    jnp.mean(selection_loss),
        "kl_loss":           jnp.mean(kl_loss),
        "reconstruction":    reconstruction,
        "kept_frame_density":     kept_frame_density.mean(),
        "mean_trajectory_prob":   jnp.mean(raw_trajectory_probs),
        "rl_loss":           jnp.mean(rl_loss),
        "per_sample_MAE":    jnp.mean(per_sample_MAE),
    }


def train_step(model, optimizer, video, mask, hparams: dict, hw: int, rngs: nnx.Rngs, perceptual_loss_fn, vgg_params):
    original_mask = mask.copy()
    # FactoredAttention already expands (b, 1, 1, time) -> (b*hw, 1, 1, time) internally.
    # Do NOT pre-expand here — that would cause a double expansion to (b*hw*hw, ...).
    mask = rearrange(mask, "b time -> b 1 1 time")
    # argnums=nnx.DiffState(0, nnx.Param) must match wrt=nnx.Param in nnx.Optimizer
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True, argnums=nnx.DiffState(0, nnx.Param))
    (loss, aux), grads = grad_fn(
        model, video, mask, original_mask, rngs, hparams, perceptual_loss_fn, vgg_params)
    optimizer.update(model, grads)
    return loss, aux


def eval_step(model, video, mask, hparams: dict, hw: int, rngs: nnx.Rngs, perceptual_loss_fn, vgg_params):
    original_mask = mask.copy()
    mask = rearrange(mask, "b time -> b 1 1 time")
    loss, aux = loss_fn(
        model, video, mask, original_mask, rngs, hparams, perceptual_loss_fn, vgg_params, train=True)
    return loss, aux


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run",        action="store_true", help="Enable wandb logging and use production save paths")
    parser.add_argument("--small",      action="store_true", help="Use small model + small data config for local testing")
    parser.add_argument("--model_path", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--train_dir",  type=str, default="/Users/jonathanzhou/Desktop/videos", help="Training video directory")
    parser.add_argument("--eval_dir",   type=str, default=None, help="Eval video directory (omit to skip eval)")
    args = parser.parse_args()

    TRAINING_RUN = args.run
    is_process_zero = jax.process_index() == 0

    if is_process_zero:
        print(f"JAX: {jax.process_count()} process(es), {jax.device_count()} total device(s)")
        print(f"Local devices: {jax.local_devices()}")

    # ── Hyperparameters ───────────────────────────────────────────────────────
    if args.small:
        # Small config — fast iteration on a single CPU/GPU
        height, width          = 64, 64
        patch_size             = 8
        encoder_depth          = 1
        decoder_depth          = 1
        mlp_dim                = 64
        num_heads              = 2
        qkv_features           = 32
        max_temporal_len       = 32   # covers up to 32 frames (epoch scaling)
        spatial_compression_rate    = 2
        unembedding_upsample_rate   = 2
        BATCH_SIZE             = 2
        MAX_FRAMES             = 8
        RESIZE                 = (64, 64)
        LEARNING_RATE          = 2e-4
    else:
        # Full production config
        height, width          = 256, 256
        patch_size             = 16
        encoder_depth          = 9
        decoder_depth          = 12
        mlp_dim                = 1536
        num_heads              = 8
        qkv_features           = 512
        max_temporal_len       = 64
        spatial_compression_rate    = 8
        unembedding_upsample_rate   = 4
        BATCH_SIZE             = 2
        MAX_FRAMES             = 32
        RESIZE                 = (256, 256)
        LEARNING_RATE          = 2e-5

    # int() required: optax schedule expects integer warmup_steps
    WARMUP_STEPS = int(20000 // math.sqrt(BATCH_SIZE))
    hw = height // patch_size * width // patch_size

    # ── Mesh for data-parallel training ──────────────────────────────────────
    # Per https://docs.jax.dev/en/latest/multi_process.html:
    #   - All devices are placed on a single 'batch' axis.
    #   - Model params use P() (no partition spec) = fully replicated on every device.
    #   - Input data is placed on the local device (not globally sharded) to
    #     avoid ShardingTypeErrors from batch-merging ops in loss_fn.
    #   - XLA inserts allreduce automatically for gradient aggregation.
    # Works identically for 1 device (single-process) or N devices (TPU/GPU cluster).
    mesh = jax.make_mesh((jax.device_count(),), ('batch',))
    replicate_sharding = NamedSharding(mesh, P())
    # ─────────────────────────────────────────────────────────────────────────

    # ── Save paths ────────────────────────────────────────────────────────────
    if TRAINING_RUN:
        model_save_path = '/mnt/t9/video_vae_saves_training/'
    else:
        model_save_path = os.path.join(SCRIPT_DIR, "checkpoints")

    VIDEO_SAVE_DIR = os.path.join(SCRIPT_DIR, "outputs")

    if TRAINING_RUN and is_process_zero:
        wandb.init(project="video-vae")

    if is_process_zero:
        reset_directory(model_save_path)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = VideoVAE(
        height=height, width=width, channels=3, patch_size=patch_size,
        encoder_depth=encoder_depth, decoder_depth=decoder_depth,
        mlp_dim=mlp_dim, num_heads=num_heads, qkv_features=qkv_features,
        max_temporal_len=max_temporal_len,
        spatial_compression_rate=spatial_compression_rate,
        unembedding_upsample_rate=unembedding_upsample_rate,
        rngs=nnx.Rngs(2),
    )

    num_params = sum(x.size for x in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param)))
    if is_process_zero:
        print(f"Trainable Parameters: {num_params / 1e6:.2f}M")

    # Replicate model params across all devices in the mesh.
    # With P() every device holds a full copy; XLA inserts allreduce for grads.
    graphdef, state = nnx.split(model)
    state = jax.device_put(state, replicate_sharding)
    model = nnx.merge(graphdef, state)

    # ── Optimizer ─────────────────────────────────────────────────────────────
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
    # nnx.Optimizer initialises Adam state from the (now-replicated) model
    # params, so optimizer state is automatically on the same sharding.
    # wrt=nnx.Param must match argnums=nnx.DiffState(0, nnx.Param) in train_step.
    optimizer = nnx.Optimizer(model, optimizer_def, wrt=nnx.Param)

    hparams = {
        "gamma1":               GAMMA1,
        "gamma2":               GAMMA2,
        "gamma3":               GAMMA3,
        "gamma4":               GAMMA4,
        "max_compression_rate": max_compression_rate,
        "magnify_negatives_rate": MAGNIFY_NEGATIVES_RATE,
        "rl_loss_weight":       RLLossWeight,
    }

    if args.model_path is not None:
        load_checkpoint(model, optimizer, args.model_path)
        hparams["max_compression_rate"] = 100000
        SEED = 42

    rngs = nnx.Rngs(3)

    # ── VGG for perceptual loss ───────────────────────────────────────────────
    vgg_model, vgg_params = load_vgg()
    perceptual_loss_fn = get_adversarial_perceptual_loss_fn(vgg_model)
    vgg_params = jax.device_put(vgg_params, replicate_sharding)

    jit_train_step = nnx.jit(train_step, static_argnames=("hw", "perceptual_loss_fn"))
    jit_eval_step  = nnx.jit(eval_step,  static_argnames=("hw", "perceptual_loss_fn"))

    global_step = 0
    start = time.perf_counter()

    if is_process_zero:
        os.makedirs(os.path.join(VIDEO_SAVE_DIR, "train"), exist_ok=True)
        if args.eval_dir:
            os.makedirs(os.path.join(VIDEO_SAVE_DIR, "eval"), exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(NUM_EPOCHS):
        if is_process_zero:
            os.makedirs(os.path.join(VIDEO_SAVE_DIR, f"train/epoch{epoch}"), exist_ok=True)

        max_frames = 64
        min_batch_size = 1
        max_epoch_multiplier = min(
            int(math.log2(BATCH_SIZE / min_batch_size)),
            int(math.log2(max_frames / MAX_FRAMES)) - 1
        )
        epoch_multiplier = min(epoch, max_epoch_multiplier)
        effective_batch_size = BATCH_SIZE // (2 ** epoch_multiplier)
        effective_max_frames = MAX_FRAMES * (2 ** epoch_multiplier)

        # create_batched_dataloader shards by process automatically via
        # grain.ShardOptions(shard_index=jax.process_index(), shard_count=jax.process_count())
        # so each process only loads its own slice of the dataset.
        train_dataloader = create_batched_dataloader(
            base_dir=args.train_dir,
            batch_size=effective_batch_size,
            max_frames=effective_max_frames,
            resize=RESIZE,
            shuffle=SHUFFLE,
            num_workers=NUM_WORKERS,
            prefetch_size=PREFETCH_SIZE,
            drop_remainder=DROP_REMAINDER,
            seed=SEED + epoch,
        )

        for i, batch in enumerate(train_dataloader):
            if i > 425948 // effective_batch_size:
                break
            if i > NEGATIVE_PENALTY_TRAINING_STEPS:
                hparams["max_compression_rate"] = 10000

            # Place this process's local numpy batch on the local device.
            # grain.ShardOptions already ensures each process gets a unique
            # data shard, so we don't need globally-sharded arrays here.
            # Globally-replicated model params cause XLA to allreduce grads
            # automatically across all processes (data-parallel training).
            device = jax.local_devices()[0]
            video = jax.device_put(np.array(batch["video"]), device).astype(jnp.bfloat16)
            mask  = jax.device_put(np.array(batch["mask"]),  device).astype(jnp.bool_)

            loss, aux = jit_train_step(
                model, optimizer, video, mask, hparams, hw, rngs, perceptual_loss_fn, vgg_params)

            global_step += 1

            if i % 500 == 499 and is_process_zero:
                recon_batch = {
                    "video": np.array(aux["reconstruction"])[:effective_batch_size],
                    "mask":  np.array(mask)[:effective_batch_size],
                }
                batch_to_video(recon_batch, os.path.join(VIDEO_SAVE_DIR, f"train/epoch{epoch}/video_{i}_latent.mp4"),   fps=30.0)
                batch_to_video(batch,       os.path.join(VIDEO_SAVE_DIR, f"train/epoch{epoch}/video_{i}_original.mp4"), fps=30.0)

            if TRAINING_RUN and is_process_zero:
                wandb.log({
                    "train_loss":              float(loss),
                    "train_MSE":               float(aux["MSE"]),
                    "train_perceptual_loss":   float(aux["perceptual_loss"]),
                    "train_Selection Loss":    float(aux["selection_loss"]),
                    "train_KL Loss":           float(aux["kl_loss"]),
                    "train_time":              time.perf_counter() - start,
                    "kept_frame_density":      float(aux["kept_frame_density"]),
                    "mean_trajectory_prob":    float(aux["mean_trajectory_prob"]),
                    "train_rl_loss":           float(aux["rl_loss"]),
                    "train_per_sample_MAE":    float(aux["per_sample_MAE"]),
                    "effective_batch_size":    effective_batch_size,
                    "effective_max_frames":    effective_max_frames,
                    "global_step":             global_step,
                })
            elif is_process_zero:
                print(
                    f"Epoch {epoch}, Step {i}: "
                    f"loss={float(loss):.4f}  MSE={float(aux['MSE']):.4f}  "
                    f"perceptual={float(aux['perceptual_loss']):.4f}  "
                    f"sel={float(aux['selection_loss']):.4f}  "
                    f"kl={float(aux['kl_loss']):.4f}  "
                    f"density={float(aux['kept_frame_density']):.4f}  "
                    f"rl={float(aux['rl_loss']):.4f}  "
                    f"MAE={float(aux['per_sample_MAE']):.4f}  "
                    f"t={time.perf_counter()-start:.1f}s"
                )

        if is_process_zero:
            save_checkpoint(model, optimizer, f"{model_save_path}/checkpoint_{epoch}")

        # ── Optional eval loop ────────────────────────────────────────────────
        if args.eval_dir is None:
            continue

        if is_process_zero:
            os.makedirs(os.path.join(VIDEO_SAVE_DIR, f"eval/epoch{epoch}"), exist_ok=True)

        eval_dataloader = create_batched_dataloader(
            base_dir=args.eval_dir,
            batch_size=effective_batch_size,
            max_frames=effective_max_frames,
            resize=RESIZE,
            shuffle=SHUFFLE,
            num_workers=NUM_WORKERS,
            prefetch_size=PREFETCH_SIZE,
            drop_remainder=DROP_REMAINDER,
            seed=SEED + epoch,
        )

        for i, batch in enumerate(eval_dataloader):
            device = jax.local_devices()[0]
            video = jax.device_put(np.array(batch["video"]), device).astype(jnp.bfloat16)
            mask  = jax.device_put(np.array(batch["mask"]),  device).astype(jnp.bool_)

            loss, aux = jit_eval_step(
                model, video, mask, hparams, hw, rngs, perceptual_loss_fn, vgg_params)

            if i % 100 == 0 and is_process_zero:
                recon_batch = {
                    "video": np.array(aux["reconstruction"])[:effective_batch_size],
                    "mask":  np.array(mask)[:effective_batch_size],
                }
                batch_to_video(recon_batch, os.path.join(VIDEO_SAVE_DIR, f"eval/epoch{epoch}/video_{i}_latent.mp4"),   fps=30.0)
                batch_to_video(batch,       os.path.join(VIDEO_SAVE_DIR, f"eval/epoch{epoch}/video_{i}_original.mp4"), fps=30.0)

            if TRAINING_RUN and is_process_zero:
                wandb.log({
                    "eval_loss":             float(loss),
                    "eval_MSE":              float(aux["MSE"]),
                    "eval_perceptual_loss":  float(aux["perceptual_loss"]),
                    "eval_Selection Loss":   float(aux["selection_loss"]),
                    "eval_KL Loss":          float(aux["kl_loss"]),
                    "eval_time":             time.perf_counter() - start,
                    "eval_per_sample_MAE":   float(aux["per_sample_MAE"]),
                    "kept_frame_density":    float(aux["kept_frame_density"]),
                    "effective_batch_size":  effective_batch_size,
                    "effective_max_frames":  effective_max_frames,
                })
            elif is_process_zero:
                print(
                    f"EVAL Epoch {epoch}, Step {i}: "
                    f"loss={float(loss):.4f}  MSE={float(aux['MSE']):.4f}  "
                    f"MAE={float(aux['per_sample_MAE']):.4f}"
                )
