import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".99"

from dataloader import create_batched_dataloader, batch_to_video
from model import VideoVAE
from classifier import Classifier
from flax import nnx
import jax
import jax.numpy as jnp
import optax
import wandb
import time
from jaxtyping import jaxtyped, Float, Array
from einops import rearrange, repeat, reduce
from model_loader import load_checkpoint
from vgg_tests import get_perceptual_loss_fn, load_vgg


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
BATCH_SIZE = 8
MAX_FRAMES = 16
RESIZE = (256, 256)
SHUFFLE = True
NUM_WORKERS = 4
PREFETCH_SIZE = 16
DROP_REMAINDER = True
SEED = 0
import math
DECAY_STEPS = 1_000_000
GAMMA1 = 0.05 # If too low, the encoder used to drop all frames, but STE gating function should prevent that now
GAMMA2 = 0.001
GAMMA3 = 0.1
GAMMA4 = 0.01  # Adversarial loss weight
ADVERSARIAL_START_STEPS = 0
LEARNING_RATE = 5e-5
WARMUP_STEPS = 20000 // math.sqrt(BATCH_SIZE)
VIDEO_SAVE_DIR = "outputs"
MAGNIFY_NEGATIVES_RATE = 100
NEGATIVE_PENALTY_TRAINING_STEPS = 2000
max_compression_rate = 2


def save_checkpoint(model, optimizer, path):                                                                       
      state = {                                                                                                      
          "model": nnx.state(model),                                                                                 
          "optimizer": nnx.state(optimizer),     
      }                                                                                                              
      ocp.StandardCheckpointer().save(path, state)                                                        


def magnify_negatives(x, magnification_rate: float):
    # Logic: If x < 0, return x * 10. Otherwise, return x.
    return jnp.where(x < 0, x * magnification_rate, x)

                               
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
    mask: Float[Array, "b 1 1 time"], original_mask: Float[Array, "b time"],
    rngs: nnx.Rngs, hparams: dict, perceptual_loss_fn, vgg_params,
    discriminator: nnx.Module = None, use_adversarial: bool = False, train: bool = True):
    reconstruction, compressed_representation, selection, logvar, mean = model(video, mask, rngs, train=train)
    sequence_lengths = reduce(original_mask, "b time -> b 1", "sum")
    sequence_lengths = jnp.clip(sequence_lengths, 1.0, None)

    video_shaped_mask = rearrange(original_mask, "b time -> b time 1 1 1")
    masked_squared_error = jnp.square((video - reconstruction) * video_shaped_mask)
    sequence_lengths_reshaped = rearrange(sequence_lengths, "b 1 -> b 1 1 1 1")
    frame_reduced_error = reduce(masked_squared_error, "b time h w c -> b 1 h w c", "sum") / sequence_lengths_reshaped
    MSE = jnp.mean(frame_reduced_error)

    perceptual_loss = perceptual_loss_fn(vgg_params, reconstruction, video)

    kl_and_selection_mask = rearrange(original_mask, "b time -> b time 1 1")

    selection_sum = reduce(selection * kl_and_selection_mask, "b time 1 1 -> b 1", "sum")

    # Kept frame density high -> lots of kept frames
    kept_frame_density = selection_sum / sequence_lengths
    # This is kind of weird
    # The idea is that we want to keep the frame density as close to 1 / max_compression_rate as possible
    # Alternatively, we can lower bound kept_frame_density - (1 / max_compression_rate)
    # But this runs into starvation risks if the encoder takes too long to reduce MSE

    density_compression_difference = kept_frame_density - (1 / hparams["max_compression_rate"])
    # We want kept_frame_density > max_compression_rate to prevent dropping all frames, so we magnify negatives
    selection_loss = jnp.mean(jnp.square(magnify_negatives(density_compression_difference, hparams["magnify_negatives_rate"])))

    sequence_lengths_reshaped = rearrange(sequence_lengths, "b 1 -> b 1 1 1")
    sequence_lengths_reshaped_kl = rearrange(sequence_lengths, "b 1 -> b 1 1 1")
    kl_loss = 0.5 * (jnp.exp(logvar) - 1 - logvar + jnp.square(mean)) * kl_and_selection_mask / sequence_lengths_reshaped_kl
    kl_loss = jnp.mean(kl_loss)

    # Adversarial loss (generator wants discriminator to output high values for reconstructions)
    if use_adversarial and discriminator is not None:
        # mask is already expanded to (b*hw, 1, 1, time) by train_step
        fake_logits = discriminator(reconstruction, mask, train=False)
        # Non-saturating GAN loss: -log(sigmoid(D(fake))) = softplus(-D(fake))
        adversarial_loss = jnp.mean(jax.nn.softplus(-fake_logits))
    else:
        adversarial_loss = 0.0

    loss = MSE + hparams["gamma3"] * perceptual_loss + hparams["gamma1"] * selection_loss + hparams["gamma2"] * kl_loss + hparams["gamma4"] * adversarial_loss

    return loss, (MSE, perceptual_loss, selection_loss, kl_loss, adversarial_loss, reconstruction, kept_frame_density.mean())

def discriminator_loss_fn(discriminator: nnx.Module, real_video: Float[Array, "b time height width channels"],
    fake_video: Float[Array, "b time height width channels"], mask: Float[Array, "b 1 1 time"], train: bool = True):
    # Standard GAN discriminator loss
    real_logits = discriminator(real_video, mask, train=train)
    fake_logits = discriminator(fake_video, mask, train=train)
    # D wants real -> 1, fake -> 0
    # Loss: -log(sigmoid(D(real))) - log(1 - sigmoid(D(fake))) = softplus(-D(real)) + softplus(D(fake))
    real_loss = jnp.mean(jax.nn.softplus(-real_logits))
    fake_loss = jnp.mean(jax.nn.softplus(fake_logits))
    # Accuracy: real should have positive logits, fake should have negative logits
    real_accuracy = jnp.mean(real_logits > 0)
    fake_accuracy = jnp.mean(fake_logits < 0)
    accuracy = (real_accuracy + fake_accuracy) / 2
    return real_loss + fake_loss, (real_loss, fake_loss, accuracy)

def discriminator_train_step(discriminator, disc_optimizer, real_video, fake_video, mask, hw: int):
    disc_mask = rearrange(mask, "b time -> b 1 1 time")
    disc_mask = repeat(disc_mask, "b 1 1 time -> b hw 1 1 time", hw=hw)
    disc_mask = rearrange(disc_mask, "b hw 1 1 time -> (b hw) 1 1 time")
    grad_fn = nnx.value_and_grad(discriminator_loss_fn, has_aux=True)
    (disc_loss, (real_loss, fake_loss, accuracy)), grads = grad_fn(discriminator, real_video, fake_video, disc_mask)
    disc_optimizer.update(grads)
    return disc_loss, real_loss, fake_loss, accuracy

def train_step(model, optimizer, video, mask, hparams: dict, hw: int, rngs: nnx.Rngs, perceptual_loss_fn, vgg_params,
    discriminator: nnx.Module = None, use_adversarial: bool = False):
    original_mask = mask.copy()
    mask = rearrange(mask, "b time -> b 1 1 time")
    mask = repeat(mask, "b 1 1 time -> b hw 1 1 time", hw=hw)
    mask = rearrange(mask, "b hw 1 1 time -> (b hw) 1 1 time")
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, (MSE, perceptual_loss, selection_loss, kl_loss, adversarial_loss, reconstruction, kept_frame_density)), grads = grad_fn(
        model, video, mask, original_mask, rngs, hparams, perceptual_loss_fn, vgg_params, discriminator, use_adversarial)
    optimizer.update(grads)

    return loss, MSE, perceptual_loss, selection_loss, kl_loss, adversarial_loss, reconstruction, kept_frame_density

def eval_step(model, video, mask, hparams: dict, hw: int, rngs: nnx.Rngs, perceptual_loss_fn, vgg_params,
    discriminator: nnx.Module = None, use_adversarial: bool = False):
    original_mask = mask.copy()
    mask = rearrange(mask, "b time -> b 1 1 time")
    mask = repeat(mask, "b 1 1 time -> b hw 1 1 time", hw=hw)
    mask = rearrange(mask, "b hw 1 1 time -> (b hw) 1 1 time")
    loss, (MSE, perceptual_loss, selection_loss, kl_loss, adversarial_loss, reconstruction, kept_frame_density) = loss_fn(
        model, video, mask, original_mask, rngs, hparams, perceptual_loss_fn, vgg_params, discriminator, use_adversarial, train=False)
    return loss, MSE, perceptual_loss, selection_loss, kl_loss, adversarial_loss, reconstruction, kept_frame_density







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

    height, width = (256, 256)
    patch_size = 16
    hw = height // patch_size * width // patch_size
    model = VideoVAE(height=height, width=width, channels=3, patch_size=patch_size,
        encoder_depth=9, decoder_depth=12, mlp_dim=1536, num_heads=8, qkv_features=512,
        max_temporal_len=64, spatial_compression_rate=8, unembedding_upsample_rate=4, rngs = nnx.Rngs(2))
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

    hparams = {
        "gamma1": GAMMA1,
        "gamma2": GAMMA2,
        "gamma3": GAMMA3,
        "gamma4": GAMMA4,
        "max_compression_rate": max_compression_rate,
        "magnify_negatives_rate": MAGNIFY_NEGATIVES_RATE,
    }

    if args.model_path is not None:
        load_checkpoint(model, optimizer, args.model_path)
        hparams["max_compression_rate"] = 100000
        SEED = 42

    rngs = nnx.Rngs(3)

    # Load VGG for perceptual loss
    vgg_model, vgg_params = load_vgg()
    perceptual_loss_fn = get_perceptual_loss_fn(vgg_model)

    # Create discriminator for adversarial loss
    discriminator = Classifier(channels=3, base_features=32, num_levels=4, rngs=nnx.Rngs(4),
        temporal_kernel=7, dtype=jnp.bfloat16, param_dtype=jnp.float32)
    disc_optimizer_def = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=LEARNING_RATE)
    )
    disc_optimizer = nnx.Optimizer(discriminator, disc_optimizer_def)

    jit_train_step = nnx.jit(train_step, static_argnames=("hw", "perceptual_loss_fn", "use_adversarial"))
    jit_eval_step = nnx.jit(eval_step, static_argnames=("hw", "perceptual_loss_fn", "use_adversarial"))
    jit_disc_train_step = nnx.jit(discriminator_train_step, static_argnames=("hw",))
    global_step = 0
    device = jax.devices()[0]
    start = time.perf_counter()
    os.makedirs(os.path.join(VIDEO_SAVE_DIR, f"train"), exist_ok=True)
    os.makedirs(os.path.join(VIDEO_SAVE_DIR, f"eval"), exist_ok=True)
    for epoch in range(NUM_EPOCHS):
        os.makedirs(os.path.join(VIDEO_SAVE_DIR, f"train/epoch{epoch}"), exist_ok=True)
        os.makedirs(os.path.join(VIDEO_SAVE_DIR, f"eval/epoch{epoch}"), exist_ok=True)
        
        max_frames = 64
        min_batch_size = 4
        max_epoch_multiplier = min(                                                                                        
            int(math.log2(BATCH_SIZE / min_batch_size)),      # from batch constraint: 2                                   
            int(math.log2(max_frames / MAX_FRAMES)) - 1       # from frame constraint (<64): 2                             
        )    
        epoch_multiplier = min(epoch, max_epoch_multiplier)
        effective_batch_size = BATCH_SIZE // (2 ** epoch_multiplier)
        effective_max_frames = MAX_FRAMES * (2 ** epoch_multiplier)
        train_dataloader = create_batched_dataloader(
                batch_size=effective_batch_size,
                max_frames=effective_max_frames,
                resize=RESIZE,
                shuffle=SHUFFLE,
                num_workers=NUM_WORKERS,
                prefetch_size=PREFETCH_SIZE,
                drop_remainder=DROP_REMAINDER,
                seed=SEED + epoch
            )

        test_dataloader = create_batched_dataloader(
                base_dir = "/mnt/t9/videos_eval",
                batch_size=effective_batch_size,
                max_frames=effective_max_frames,
                resize=RESIZE,
                shuffle=SHUFFLE,
                num_workers=NUM_WORKERS,
                prefetch_size=PREFETCH_SIZE,
                drop_remainder=DROP_REMAINDER,
                seed=SEED + epoch
            )

        compression_rate_increment = 1e-4 // (2 ** epoch_multiplier)
        for i, batch in enumerate(train_dataloader):


            if i > 425948 // effective_batch_size: # Dataloader doesn't natually terminate for some reason
                break
            if i > NEGATIVE_PENALTY_TRAINING_STEPS:
                hparams["max_compression_rate"] = 10000
            video = jax.device_put(batch["video"], device)
            mask = jax.device_put(batch["mask"], device)
            mask = mask.astype(jnp.bool)
            video = video.astype(jnp.bfloat16)

            use_adversarial = global_step >= ADVERSARIAL_START_STEPS

            loss, MSE, perceptual_loss, selection_loss, kl_loss, adversarial_loss, reconstruction, kept_frame_density = jit_train_step(
                model, optimizer, video, mask, hparams, hw, rngs, perceptual_loss_fn, vgg_params, discriminator, use_adversarial)

            # Always train discriminator (but only add to generator loss when use_adversarial)
            disc_loss, real_loss, fake_loss, disc_accuracy = jit_disc_train_step(
                discriminator, disc_optimizer, video, reconstruction, mask, hw)

            global_step += 1

            if i % 1000 == 999:
                recon_batch = {
                    "video": reconstruction,
                    "mask": mask  # or mask.squeeze(), depending on shape
                }
                batch_to_video(recon_batch, os.path.join(VIDEO_SAVE_DIR, f"train/epoch{epoch}/video_{i}_latent.mp4"), fps=30.0)
                batch_to_video(batch, os.path.join(VIDEO_SAVE_DIR, f"train/epoch{epoch}/video_{i}_original.mp4"), fps=30.0)
            if TRAINING_RUN:
                wandb.log({
                    "train_loss": loss,
                    "train_MSE": MSE,
                    "train_perceptual_loss": perceptual_loss,
                    "train_Selection Loss": selection_loss,
                    "train_KL Loss": kl_loss,
                    "train_adversarial_loss": adversarial_loss,
                    "train_disc_loss": disc_loss,
                    "train_disc_accuracy": disc_accuracy,
                    "train_time": time.perf_counter() - start,
                    "kept_frame_density": kept_frame_density,
                    "effective_batch_size": effective_batch_size,
                    "effective_max_frames": effective_max_frames,
                    "global_step": global_step,
                })
            else:
                print(f"Epoch {epoch}, Step {i}: Loss = {float(loss):.4f}, MSE = {float(MSE):.4f}, Perceptual Loss = {float(perceptual_loss):.4f}, Selection Loss = {float(selection_loss):.4f}, KL Loss = {float(kl_loss):.4f}, Adv Loss = {float(adversarial_loss):.4f}, Disc Loss = {float(disc_loss):.4f}, Disc Acc = {float(disc_accuracy):.4f}, time = {time.perf_counter() - start:.4f}, kept_frame_density = {float(kept_frame_density):.4f}")
        save_checkpoint(model, optimizer, discriminator, disc_optimizer, f"{model_save_path}/checkpoint_{epoch}")
        for i, batch in enumerate(test_dataloader):
            video = jax.device_put(batch["video"], device)
            mask = jax.device_put(batch["mask"], device)
            mask = mask.astype(jnp.bool)
            use_adversarial = global_step >= ADVERSARIAL_START_STEPS
            loss, MSE, perceptual_loss, selection_loss, kl_loss, adversarial_loss, reconstruction, kept_frame_density = jit_eval_step(
                model, video, mask, hparams, hw, rngs, perceptual_loss_fn, vgg_params, discriminator, use_adversarial)
            if i % 100 == 0:
                recon_batch = {
                    "video": reconstruction,
                    "mask": mask  # or mask.squeeze(), depending on shape
                }
                batch_to_video(recon_batch, os.path.join(VIDEO_SAVE_DIR, f"eval/epoch{epoch}/video_{i}_latent.mp4"), fps=30.0)
                batch_to_video(batch, os.path.join(VIDEO_SAVE_DIR, f"eval/epoch{epoch}/video_{i}_original.mp4"), fps=30.0)
            if TRAINING_RUN:
                wandb.log({
                    "eval_loss": loss,
                    "eval_MSE": MSE,
                    "eval_perceptual_loss": perceptual_loss,
                    "eval_Selection Loss": selection_loss,
                    "eval_KL Loss": kl_loss,
                    "eval_adversarial_loss": adversarial_loss,
                    "eval_time": time.perf_counter() - start,
                    "kept_frame_density": kept_frame_density,
                    "effective_batch_size": effective_batch_size,
                    "effective_max_frames": effective_max_frames
                })
            else:
                print(f"VALIDATION Epoch {epoch}, Step {i}: Loss = {float(loss):.4f}, MSE = {float(MSE):.4f}, Perceptual Loss = {float(perceptual_loss):.4f}, Selection Loss = {float(selection_loss):.4f}, KL Loss = {float(kl_loss):.4f}, Adv Loss = {float(adversarial_loss):.4f}, effective_batch_size = {effective_batch_size}, effective_max_frames = {effective_max_frames}")

