import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".99"

from dataloader import create_batched_dataloader, batch_to_video
from model import VideoVAE
from flax import nnx
import jax
import jax.numpy as jnp
import optax
import wandb
import time
from jaxtyping import jaxtyped, Float, Array
from vgg_tests import load_vgg, get_perceptual_loss_fn
from einops import rearrange, repeat, reduce
from model_loader import load_checkpoint

import orbax.checkpoint as ocp
import shutil

model_save_path = '/mnt/t9/video_vae_saves/'


def reset_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"Deleted: {path}")
    os.makedirs(path, exist_ok=True)
    print(f"Created/Reloaded: {path}")


def save_checkpoint(model, optimizer, path):
    state = {
        "model": nnx.state(model),
        "optimizer": nnx.state(optimizer),
    }
    ocp.StandardCheckpointer().save(path, state)                                                        


def loss_fn(model: nnx.Module, perceptual_loss_fn, vgg_params, video: Float[Array, "b time height width channels"],
    mask: Float[Array, "b 1 1 time"], original_mask: Float[Array, "b time"],
    rngs: nnx.Rngs, hparams: dict):
    reconstruction, compressed_representation, selection, logvar, mean = model(video, mask, rngs)
    sequence_lengths = reduce(original_mask, "b time -> b 1", "sum")

    video_shaped_mask = rearrange(original_mask, "b time -> b time 1 1 1")
    masked_squared_error = jnp.square((video - reconstruction) * video_shaped_mask)
    sequence_lengths_reshaped = rearrange(sequence_lengths, "b 1 -> b 1 1 1 1")
    frame_reduced_error = reduce(masked_squared_error, "b time h w c -> b 1 h w c", "sum") / sequence_lengths_reshaped
    MSE = jnp.mean(frame_reduced_error)

    kl_and_selection_mask = rearrange(original_mask, "b time -> b time 1 1")

    selection_sum = reduce(selection * kl_and_selection_mask, "b time 1 1 -> b 1", "sum")
    kept_frame_density = selection_sum / sequence_lengths
    selection_loss = jnp.mean(jnp.square(kept_frame_density - (1 / hparams["max_compression_rate"])))

    sequence_lengths_reshaped = rearrange(sequence_lengths, "b 1 -> b 1 1 1")
    kl_loss = 0.5 * (jnp.exp(logvar) - 1 - logvar + jnp.square(mean)) * kl_and_selection_mask / sequence_lengths_reshaped
    kl_loss = jnp.mean(kl_loss)

    perceptual_loss = perceptual_loss_fn(vgg_params, reconstruction, video)

    loss = MSE + hparams["gamma1"] * selection_loss + hparams["gamma2"] * kl_loss + hparams["gamma3"] * perceptual_loss
    return loss, (MSE, selection_loss, kl_loss, perceptual_loss, reconstruction)


def train_step(model, optimizer, perceptual_loss_fn, vgg_params, video, mask, hparams: dict, hw: int, rngs: nnx.Rngs):
    original_mask = mask.copy()
    mask = rearrange(mask, "b time -> b 1 1 time")
    mask = repeat(mask, "b 1 1 time -> b hw 1 1 time", hw=hw)
    mask = rearrange(mask, "b hw 1 1 time -> (b hw) 1 1 time")
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, (MSE, selection_loss, kl_loss, perceptual_loss, reconstruction)), grads = grad_fn(
        model, perceptual_loss_fn, vgg_params, video, mask, original_mask, rngs, hparams)
    optimizer.update(grads)
    return loss, MSE, selection_loss, kl_loss, perceptual_loss, reconstruction

def eval_step(model, perceptual_loss_fn, vgg_params, video, mask, hparams: dict, hw: int, rngs: nnx.Rngs):
    original_mask = mask.copy()
    mask = rearrange(mask, "b time -> b 1 1 time")
    mask = repeat(mask, "b 1 1 time -> b hw 1 1 time", hw=hw)
    mask = rearrange(mask, "b hw 1 1 time -> (b hw) 1 1 time")
    loss, (MSE, selection_loss, kl_loss, perceptual_loss, reconstruction) = loss_fn(
        model, perceptual_loss_fn, vgg_params, video, mask, original_mask, rngs, hparams)
    return loss, MSE, selection_loss, kl_loss, perceptual_loss, reconstruction



NUM_EPOCHS = 100
BATCH_SIZE = 16
MAX_FRAMES = 8
RESIZE = (256, 256)
SHUFFLE = True
NUM_WORKERS = 4
PREFETCH_SIZE = 16
DROP_REMAINDER = True
SEED = 42
WARMUP_STEPS = 2500
DECAY_STEPS = 100_000
GAMMA1 = 0.005
GAMMA2 = 0.001
GAMMA3 = 0.1
LEARNING_RATE = 2e-5
VIDEO_SAVE_DIR = "outputs"
max_compression_rate = 1.2

hparams = {
        "gamma1": GAMMA1,
        "gamma2": GAMMA2,
        "gamma3": GAMMA3,
        "max_compression_rate": max_compression_rate,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Check for run flag.")
    parser.add_argument("--run", action="store_true", help="Set the run flag to True")
    parser.add_argument("--model_path", type=str, default=None, help="Path to load model/optimizer checkpoint from")
    args = parser.parse_args()
    is_running = args.run
    TRAINING_RUN = is_running
    if TRAINING_RUN:
        model_save_path = '/mnt/t9/video_vae_saves_training/'
        wandb.init(project="video-vae")
    reset_directory(model_save_path)

    train_dataloader = create_batched_dataloader(
        batch_size=BATCH_SIZE,
        max_frames=MAX_FRAMES,
        resize=RESIZE,
        shuffle=SHUFFLE,
        num_workers=NUM_WORKERS,
        prefetch_size=PREFETCH_SIZE,
        drop_remainder=DROP_REMAINDER,
        seed=SEED
    )

    test_dataloader = create_batched_dataloader(
        base_dir = "/mnt/t9/videos_eval",
        batch_size=BATCH_SIZE,
        max_frames=MAX_FRAMES,
        resize=RESIZE,
        shuffle=SHUFFLE,
        num_workers=NUM_WORKERS,
        prefetch_size=PREFETCH_SIZE,
        drop_remainder=DROP_REMAINDER,
        seed=SEED
    )
    height, width = (256, 256)
    patch_size = 16
    hw = height // patch_size * width // patch_size
    model = VideoVAE(height=height, width=width, channels=3, patch_size=patch_size, 
        depth=6, mlp_dim=1024, num_heads=8, qkv_features=128,
        max_temporal_len=MAX_FRAMES, spatial_compression_rate=4, rngs = nnx.Rngs(1))

    params = nnx.state(model, nnx.Param)
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Trainable Parameters: {num_params / 10**6} Million")

    

        
    schedule_fn= optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    decay_steps=DECAY_STEPS,
    end_value=LEARNING_RATE / 10,
    )
    optimizer_def = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=schedule_fn)
    )

    optimizer = nnx.Optimizer(model, optimizer_def)

    if args.model_path is not None:
        load_checkpoint(model, optimizer, args.model_path)
        hparams["max_compression_rate"] = 100000
        SEED = 42

    vgg, vgg_params = load_vgg()
    perceptual_loss_fn = get_perceptual_loss_fn(vgg)
    rngs = nnx.Rngs(0)
    jit_train_step = nnx.jit(train_step, static_argnames=("hw", "perceptual_loss_fn"))
    jit_eval_step = nnx.jit(eval_step, static_argnames=("hw", "perceptual_loss_fn"))
    device = jax.devices()[0]
    start = time.perf_counter()

    for epoch in range(NUM_EPOCHS):
        os.makedirs(os.path.join(VIDEO_SAVE_DIR, f"train/epoch{epoch}"), exist_ok=True)
        os.makedirs(os.path.join(VIDEO_SAVE_DIR, f"eval/epoch{epoch}"), exist_ok=True)
        for i, batch in enumerate(train_dataloader):
            max_compression_rate += 1e-5
            video = jax.device_put(batch["video"], device)
            mask = jax.device_put(batch["mask"], device)
            
            loss, MSE, selection_loss, kl_loss, perceptual_loss, reconstruction = jit_train_step(
                model, optimizer, perceptual_loss_fn, vgg_params, video, mask, hparams, hw, rngs)
            if i % 1000 == 999:
                recon_batch = {
                    "video": reconstruction,
                    "mask": mask  # or mask.squeeze(), depending on shape
                }
                batch_to_video(recon_batch, os.path.join(VIDEO_SAVE_DIR, f"train/epoch{epoch}/video_{i}_reconstruction.mp4"), fps=30.0)
                batch_to_video(batch, os.path.join(VIDEO_SAVE_DIR, f"train/epoch{epoch}/video_{i}_original.mp4"), fps=30.0)
            if TRAINING_RUN:
                wandb.log({
                    "train_loss": loss,
                    "train_MSE": MSE,
                    "train_Selection Loss": selection_loss,
                    "train_KL Loss": kl_loss,
                    "train_Perceptual Loss": perceptual_loss,
                    "train_time": time.perf_counter() - start
                })
            else:
                print(f"Epoch {epoch}, Step {i}: Loss = {loss:.4f}, MSE = {MSE:.4f}, Selection Loss = {selection_loss:.4f}, KL Loss = {kl_loss:.4f}, Perceptual Loss = {perceptual_loss:.4f}, time = {time.perf_counter() - start:.4f}")
        save_checkpoint(model, optimizer, f"{model_save_path}/checkpoint_{epoch}")

        for i, batch in enumerate(test_dataloader):
            video = jax.device_put(batch["video"], device)
            mask = jax.device_put(batch["mask"], device)
            loss, MSE, selection_loss, kl_loss, perceptual_loss, reconstruction = jit_eval_step(
                model, perceptual_loss_fn, vgg_params, video, mask, hparams, hw, rngs)
            if i % 100 == 0:
                recon_batch = {
                    "video": reconstruction,
                    "mask": mask  # or mask.squeeze(), depending on shape
                }
                batch_to_video(recon_batch, os.path.join(VIDEO_SAVE_DIR, f"eval/epoch{epoch}/video_{i}_reconstruction.mp4"), fps=30.0)
                batch_to_video(batch, os.path.join(VIDEO_SAVE_DIR, f"eval/epoch{epoch}/video_{i}_original.mp4"), fps=30.0)
            if TRAINING_RUN:
                wandb.log({
                    "eval_loss": loss,
                    "eval_MSE": MSE,
                    "eval_Selection Loss": selection_loss,
                    "eval_KL Loss": kl_loss,
                    "eval_Perceptual Loss": perceptual_loss,
                    "eval_time": time.perf_counter() - start
                })
            else:
                print(f"VALIDATION Epoch {epoch}, Step {i}: Loss = {loss:.4f}, MSE = {MSE:.4f}, Selection Loss = {selection_loss:.4f}, KL Loss = {kl_loss:.4f}, Perceptual Loss = {perceptual_loss:.4f}")
            
