from dataloader import create_batched_dataloader, batch_to_video
from model import VideoVAE
from flax import nnx
import jax
import jax.numpy as jnp
import optax
import wandb
import time
from jaxtyping import jaxtyped, Float, Array
import os
from einops import rearrange, repeat, reduce

#wandb.init(project="video-vae")

NUM_EPOCHS = 100
BATCH_SIZE = 1
MAX_FRAMES = 128
RESIZE = (256, 256)
SHUFFLE = True
NUM_WORKERS = 0
PREFETCH_SIZE = 16
DROP_REMAINDER = True
SEED = 42
WARMUP_STEPS = 5000
DECAY_STEPS = 100_000
GAMMA1 = 0.05
GAMMA2 = 0.1
VIDEO_SAVE_DIR = "outputs"
max_compression_rate = 4

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
    depth=1, mlp_dim=128, num_heads=8, qkv_features=128,
    max_temporal_len=MAX_FRAMES, spatial_compression_rate=4, rngs = nnx.Rngs(0))

schedule_fn= optax.warmup_cosine_decay_schedule(
  init_value=0.0,
  peak_value=1.0,
  warmup_steps=WARMUP_STEPS,
  decay_steps=DECAY_STEPS,
  end_value=0.0,
)
optimizer_def = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=schedule_fn)
)

optimizer = nnx.Optimizer(model, optimizer_def)
#step_count = optimizer.opt_state.count

def loss_fn(model: nnx.Module, video: Float[Array, "b time height width channels"], 
    mask: Float[Array, "b 1 1 time"], gamma1: float, gamma2: float, max_compression_rate: float, 
    original_mask: Float[Array, "b time"],
    rngs: nnx.Rngs):
    reconstruction, compressed_representation, selection, logvar = model(video, mask, rngs)
    MSE = jnp.mean(jnp.square(video - reconstruction))
    sequence_lengths = reduce(mask, "b 1 1 time -> b 1", "sum")
    selection_sum = reduce(selection, "b time 1 1 -> b time", "sum")
    kept_frame_density = selection_sum / sequence_lengths
    selection_loss = jnp.mean(jnp.square(kept_frame_density - 1 / max_compression_rate))
    original_mask = rearrange(original_mask, "b time -> b time 1 1")
    kl_loss = 0.5 * (jnp.exp(logvar) - 1 - logvar) * original_mask / sequence_lengths
    kl_loss = jnp.mean(kl_loss)
    loss = MSE + gamma1 * selection_loss + gamma2 * kl_loss
    return loss, (MSE, selection_loss, kl_loss, reconstruction)


def train_step(model, optimizer, video, mask, gamma1, gamma2, max_compression_rate, hw, rngs: nnx.Rngs):
    original_mask = mask.copy()
    mask = rearrange(mask, "b time -> b 1 1 time")
    mask = repeat(mask, "b 1 1 time -> b hw 1 1 time", hw = hw)
    mask = rearrange(mask, "b hw 1 1 time -> (b hw) 1 1 time")
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, (MSE, selection_loss, kl_loss, reconstruction)), grads = grad_fn(model, video, mask, gamma1, gamma2, 
    max_compression_rate, original_mask, rngs)
    optimizer.update(grads)
    return loss, MSE, selection_loss, kl_loss, reconstruction

def eval_step(model, video, mask, gamma1, gamma2, max_compression_rate, hw, rngs: nnx.Rngs):
    original_mask = mask.copy()
    mask = rearrange(mask, "b time -> b 1 1 time")
    mask = repeat(mask, "b 1 1 time -> b hw 1 1 time", hw = hw)
    mask = rearrange(mask, "b hw 1 1 time -> (b hw) 1 1 time")
    loss, (MSE, selection_loss, kl_loss, reconstruction) = loss_fn(model, video, mask, gamma1, gamma2, 
    max_compression_rate, original_mask, rngs)
    return loss, MSE, selection_loss, kl_loss, reconstruction

rngs = nnx.Rngs(0)
jit_train_step = train_step #nnx.jit(train_step)
jit_eval_step = nnx.jit(eval_step)
device = jax.devices()[0]

for epoch in range(NUM_EPOCHS):
    os.makedirs(os.path.join(VIDEO_SAVE_DIR, f"train/epoch{epoch}"), exist_ok=True)
    os.makedirs(os.path.join(VIDEO_SAVE_DIR, f"eval/epoch{epoch}"), exist_ok=True)
    for i, batch in enumerate(train_dataloader):
        
        video = jax.device_put(batch["video"], device)
        mask = jax.device_put(batch["mask"], device)
        
        loss, MSE, selection_loss, kl_loss, reconstruction = jit_train_step(model, optimizer, video, mask, GAMMA1, GAMMA2, max_compression_rate, hw, rngs)
        if i % 1000 == 0:
            recon_batch = {
                "video": reconstruction,
                "mask": mask  # or mask.squeeze(), depending on shape
            }
            batch_to_video(recon_batch, os.path.join(VIDEO_SAVE_DIR, f"train/epoch{epoch}/video{i}.mp4"), fps=30.0)
        print(f"Epoch {epoch}, Step {i}: Loss = {loss:.4f}, MSE = {MSE:.4f}, Selection Loss = {selection_loss:.4f}, KL Loss = {kl_loss:.4f}")
        exit()
    for i, batch in enumerate(test_dataloader):
        video = jax.device_put(batch["video"], device)
        mask = jax.device_put(batch["mask"], device)
        loss, MSE, selection_loss, kl_loss, reconstruction = jit_eval_step(model, video, mask, GAMMA1, GAMMA2, max_compression_rate, hw, rngs)
        if i % 100 == 0:
            recon_batch = {
                "video": reconstruction,
                "mask": mask  # or mask.squeeze(), depending on shape
            }
            batch_to_video(recon_batch, os.path.join(VIDEO_SAVE_DIR, f"eval/epoch{epoch}/video{i}.mp4"), fps=30.0)
        
        print(f"VALIDATION Epoch {epoch}, Step {i}: Loss = {loss:.4f}, MSE = {MSE:.4f}, Selection Loss = {selection_loss:.4f}, KL Loss = {kl_loss:.4f}")
        
