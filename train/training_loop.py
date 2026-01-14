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
from einops import rearrange, repeat, reduce



import orbax.checkpoint as ocp
model_save_path = '/mnt/t9/video_vae_saves/'
import shutil
import os

def reset_directory(path):
    # 1. Check if it exists and delete it
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"Deleted: {path}")
    
    # 2. Recreate the empty directory
    os.makedirs(path, exist_ok=True)
    print(f"Created/Reloaded: {path}")



NUM_EPOCHS = 100
BATCH_SIZE = 64
MAX_FRAMES = 8
RESIZE = (256, 256)
SHUFFLE = True
NUM_WORKERS = 4
PREFETCH_SIZE = 16
DROP_REMAINDER = True
SEED = 0

DECAY_STEPS = 100_000
GAMMA1 = 0.005 # If too low, the encoder used to drop all frames, but STE gating function should prevent that now
GAMMA2 = 0.01
LEARNING_RATE = 1e-4
WARMUP_STEPS = 20000 // BATCH_SIZE
VIDEO_SAVE_DIR = "outputs"
max_compression_rate = 1.2



def save_checkpoint(model, optimizer, path):                                                                       
      state = {                                                                                                      
          "model": nnx.state(model),                                                                                 
          "optimizer": nnx.state(optimizer)                                                                          
      }                                                                                                              
      ocp.StandardCheckpointer().save(path, state)                                                        
                                                      
def print_max_grad(grads):
    """
    Calculates and prints the maximum absolute value found in the entire gradient tree.
    """
    # 1. Flatten the PyTree into a list of arrays
    leaves = jax.tree_util.tree_leaves(grads)
    
    # 2. Compute max(abs(x)) for each leaf
    # We use jnp.max to handle both arrays and scalars
    max_per_leaf = [jnp.max(jnp.abs(leaf)) for leaf in leaves]
    
    # 3. Compute the global max across all leaves
    global_max = jnp.max(jnp.array(max_per_leaf))
    
    # 4. Print
    # If inside JIT, use jax.debug.print. If outside, standard print works.
    jax.debug.print("📈 Max Gradient Value: {x:.6f}", x=global_max)
    
    return global_max

def loss_fn(model: nnx.Module, video: Float[Array, "b time height width channels"], 
    mask: Float[Array, "b 1 1 time"], gamma1: float, gamma2: float, max_compression_rate: float, 
    original_mask: Float[Array, "b time"],
    rngs: nnx.Rngs):
    reconstruction, compressed_representation, selection, logvar, mean = model(video, mask, rngs)
    sequence_lengths = reduce(original_mask, "b time -> b 1", "sum")
    sequence_lengths = jnp.clip(sequence_lengths, 1.0, None)
    
    


    video_shaped_mask = rearrange(original_mask, "b time -> b time 1 1 1")
    masked_squared_error = jnp.square((video - reconstruction) * video_shaped_mask)
    sequence_lengths_reshaped = rearrange(sequence_lengths, "b 1 -> b 1 1 1 1")
    
    frame_reduced_error = reduce(masked_squared_error, "b time h w c -> b 1 h w c", "sum") / sequence_lengths_reshaped
    MSE = jnp.mean(frame_reduced_error)
    

    kl_and_selection_mask = rearrange(original_mask, "b time -> b time 1 1")

    selection_sum = reduce(selection * kl_and_selection_mask, "b time 1 1 -> b 1", "sum")
    kept_frame_density = selection_sum / sequence_lengths
    # This is kind of weird 
    # The idea is that we want to keep the frame density as close to 1 / max_compression_rate as possible
    # Alternatively, we can lower bound kept_frame_density - (1 / max_compression_rate) 
    # But this runs into starvation risks if the encoder takes too long to reduce MSE
    selection_loss = jnp.mean(jnp.square(kept_frame_density - (1 / max_compression_rate)))


    
    sequence_lengths_reshaped = rearrange(sequence_lengths, "b 1 -> b 1 1 1")
    kl_loss = 0.5 * (jnp.exp(logvar) - 1 - logvar + jnp.square(mean)) * kl_and_selection_mask / sequence_lengths_reshaped
    kl_loss = jnp.mean(kl_loss)
    loss = MSE + gamma1 * selection_loss + gamma2 * kl_loss      
    #jax.debug.print("Input range: [{min:.3f}, {max:.3f}]", min=video.min(), max=video.max())
    #jax.debug.print("Recon range: [{min:.3f}, {max:.3f}]", min=reconstruction.min(), max=reconstruction.max())
    #jax.debug.print("log_var range: [{min:.3f}, {max:.3f}]", min=logvar.min(), max=logvar.max())
    #jax.debug.print("mean range: [{min:.3f}, {max:.3f}]", min=mean.min(), max=mean.max())       
                       
    return loss, (MSE, selection_loss, kl_loss, reconstruction)


def find_first_nan_grad(grads):
    def check(path, leaf):
        if jnp.isnan(leaf).any():
            print(f"❌ NaN Gradient in: {jax.tree_util.keystr(path)}")
            # Print range to see if it's overflowing
            print(f"   Range: [{jnp.min(leaf)}, {jnp.max(leaf)}]")
    
    jax.tree_util.tree_map_with_path(check, grads)

def train_step(model, optimizer, video, mask, gamma1, gamma2, max_compression_rate, hw, rngs: nnx.Rngs):
    original_mask = mask.copy()
    mask = rearrange(mask, "b time -> b 1 1 time")
    mask = repeat(mask, "b 1 1 time -> b hw 1 1 time", hw = hw)
    mask = rearrange(mask, "b hw 1 1 time -> (b hw) 1 1 time")
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, (MSE, selection_loss, kl_loss, reconstruction)), grads = grad_fn(model, video, mask, gamma1, gamma2, 
    max_compression_rate, original_mask, rngs)
    #jax.debug.print("Max Gradient Value: {x:.6f}", x=print_max_grad(grads)) 
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







if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Check for run flag.")
    parser.add_argument("--run", action="store_true", help="Set the run flag to True")
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
        depth=6, mlp_dim=1536, num_heads=8, qkv_features=256,
        max_temporal_len=MAX_FRAMES, spatial_compression_rate=4, rngs = nnx.Rngs(2))

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
    #step_count = optimizer.opt_state.count

    rngs = nnx.Rngs(3)
    jit_train_step = nnx.jit(train_step, static_argnames = ("hw"))
    jit_eval_step = nnx.jit(eval_step, static_argnames = ("hw"))
    device = jax.devices()[0]
    start = time.perf_counter()

    for epoch in range(NUM_EPOCHS):
        os.makedirs(os.path.join(VIDEO_SAVE_DIR, f"train/epoch{epoch}"), exist_ok=True)
        os.makedirs(os.path.join(VIDEO_SAVE_DIR, f"eval/epoch{epoch}"), exist_ok=True)
        

        for i, batch in enumerate(train_dataloader):
            
            max_compression_rate += 1e-5
            video = jax.device_put(batch["video"], device)
            mask = jax.device_put(batch["mask"], device)
            mask = mask.astype(jnp.bool)
            
            loss, MSE, selection_loss, kl_loss, reconstruction = jit_train_step(model, optimizer, video, mask, GAMMA1, GAMMA2, max_compression_rate, hw, rngs)                                  
            if i % 1000 == 999:
                recon_batch = {
                    "video": reconstruction,
                    "mask": mask  # or mask.squeeze(), depending on shape
                }
                batch_to_video(recon_batch, os.path.join(VIDEO_SAVE_DIR, f"train/epoch{epoch}/video_latent_{i}.mp4"), fps=30.0)
                batch_to_video(batch, os.path.join(VIDEO_SAVE_DIR, f"train/epoch{epoch}/video_original_{i}.mp4"), fps=30.0)
            if TRAINING_RUN:
                wandb.log({
                    "train_loss": loss,
                    "train_MSE": MSE,
                    "train_Selection Loss": selection_loss,
                    "train_KL Loss": kl_loss,
                    "train_time": time.perf_counter() - start
                })
            else:
                print(f"Epoch {epoch}, Step {i}: Loss = {loss:.4f}, MSE = {MSE:.4f}, Selection Loss = {selection_loss:.4f}, KL Loss = {kl_loss:.4f}, time = {time.perf_counter() - start:.4f}")
        save_checkpoint(model, optimizer, f"{model_save_path}/checkpoint_{epoch}")
        for i, batch in enumerate(test_dataloader):
            video = jax.device_put(batch["video"], device)
            mask = jax.device_put(batch["mask"], device)
            mask = mask.astype(jnp.bool)
            loss, MSE, selection_loss, kl_loss, reconstruction = jit_eval_step(model, video, mask, GAMMA1, GAMMA2, max_compression_rate, hw, rngs)
            if i % 100 == 0:
                recon_batch = {
                    "video": reconstruction,
                    "mask": mask  # or mask.squeeze(), depending on shape
                }
                batch_to_video(recon_batch, os.path.join(VIDEO_SAVE_DIR, f"eval/epoch{epoch}/video_latent_{i}.mp4"), fps=30.0)
                batch_to_video(batch, os.path.join(VIDEO_SAVE_DIR, f"eval/epoch{epoch}/video_original_{i}.mp4"), fps=30.0)
            if TRAINING_RUN:
                wandb.log({
                    "eval_loss": loss,
                    "eval_MSE": MSE,
                    "eval_Selection Loss": selection_loss,
                    "eval_KL Loss": kl_loss,
                    "eval_time": time.perf_counter() - start
                })
            else:
                print(f"VALIDATION Epoch {epoch}, Step {i}: Loss = {loss:.4f}, MSE = {MSE:.4f}, Selection Loss = {selection_loss:.4f}, KL Loss = {kl_loss:.4f}")
            
