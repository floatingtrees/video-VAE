import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'inference'))

import jax
import jax.numpy as jnp
from flax import nnx
import optax
import orbax.checkpoint as ocp
import numpy as np
from einops import rearrange

from autoencoder import VideoVAE
from diffusion_model import VideoDiT
from train_step import train_step, sample
from dataloader import create_batched_dataloader, batch_to_video

VAE_CHECKPOINT = "/mnt/t9/vae_longterm_saves/gcs2/checkpoint_step_130000"
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'inference', 'test_videos')
MAX_FRAMES = 32
BATCH_SIZE = 1
HEIGHT, WIDTH = 256, 256
NUM_EPOCHS = 500
NUM_SAMPLE_STEPS = 50
SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "test_samples")


def load_vae_checkpoint(model, path):
    schedule_fn = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=6e-5,
        warmup_steps=5000, decay_steps=1_000_000, end_value=6e-6,
    )
    optimizer_def = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=schedule_fn),
    )
    optimizer = nnx.Optimizer(model, optimizer_def)

    abstract_state = {
        "model": jax.tree.map(ocp.utils.to_shape_dtype_struct, nnx.state(model)),
        "optimizer": jax.tree.map(ocp.utils.to_shape_dtype_struct, nnx.state(optimizer)),
    }
    restored = ocp.StandardCheckpointer().restore(path, abstract_state)
    nnx.update(model, restored["model"])


def main():
    # Create and load VAE (frozen, same config as run_inference.py)
    print("Creating VAE...")
    vae = VideoVAE(
        height=HEIGHT, width=WIDTH, channels=3, patch_size=16,
        encoder_depth=9, decoder_depth=12, mlp_dim=1536, num_heads=8,
        qkv_features=512, max_temporal_len=64,
        spatial_compression_rate=8, unembedding_upsample_rate=4,
        rngs=nnx.Rngs(0),
    )
    print(f"Loading VAE checkpoint from {VAE_CHECKPOINT}...")
    load_vae_checkpoint(vae, VAE_CHECKPOINT)
    print("VAE loaded.")

    # Small DiT for testing (compressed shape: b, t, hw=256, c=96)
    dit = VideoDiT(
        hw=256,
        residual_dim=256,
        compressed_channel_dim=96,
        depth=4,
        mlp_dim=512,
        num_heads=4,
        qkv_features=256,
        max_temporal_len=64,
        rngs=nnx.Rngs(1),
    )
    params = nnx.state(dit, nnx.Param)
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"DiT parameters: {num_params / 1e6:.1f}M")

    optimizer = nnx.Optimizer(dit, optax.adam(1e-4))
    hparams = {"lambda1": 0.1}
    rngs = nnx.Rngs(sampling=42)

    # Load videos
    print(f"Loading videos from {DATA_DIR}...")
    dataloader = create_batched_dataloader(
        base_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        max_frames=MAX_FRAMES,
        resize=(HEIGHT, WIDTH),
        shuffle=True,
        num_workers=1,
        prefetch_size=2,
        drop_remainder=True,
    )

    # Preload all batches into memory so we can repeat over them
    batches = []
    for batch in dataloader:
        video = jnp.array(batch["video"]).astype(jnp.bfloat16)
        mask = jnp.array(batch["mask"]).astype(jnp.bool_)
        video_mask = rearrange(mask, "b time -> b 1 1 time")
        batches.append((video, video_mask))
    print(f"Loaded {len(batches)} batches.")

    os.makedirs(SAMPLES_DIR, exist_ok=True)

    # JIT the sample + decode pipeline
    @nnx.jit(static_argnums=(3,))
    def generate(dit, vae, noise, num_steps, compression_mask, selection_indices, video_mask, rngs):
        denoised, sel_pred = sample(dit, noise, compression_mask, num_steps)
        reconstruction = vae.decompress(denoised, video_mask, selection_indices, compression_mask, rngs, train=False)
        return reconstruction

    # Get ground truth masks from first batch for generation
    ref_video, ref_video_mask = batches[0]
    ref_compressed, ref_selection_indices, ref_compression_mask = vae.compress(ref_video, ref_video_mask, rngs)

    # Train loop
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_sel = 0.0
        for i, (video, video_mask) in enumerate(batches):
            loss, aux = train_step(dit, vae, optimizer, video, video_mask, hparams, rngs)
            epoch_loss += float(loss)
            epoch_mse += float(aux["MSE"])
            epoch_sel += float(aux["selection_loss"])
            if i > 1000:
                break
        n = len(batches)
        print(f"Epoch {epoch:3d} | loss={epoch_loss/n:.6f}  MSE={epoch_mse/n:.6f}  sel_loss={epoch_sel/n:.6f}")

        if epoch % 50 == 0:
            key = rngs.sampling()
            noise = jax.random.normal(key, ref_compressed.shape)
            reconstruction = generate(
                dit, vae, noise, NUM_SAMPLE_STEPS,
                ref_compression_mask, ref_selection_indices, ref_video_mask, rngs,
            )
            recon_batch = {
                "video": np.array(reconstruction),
                "mask": np.array(rearrange(ref_video_mask, "b 1 1 t -> b t")),
            }
            out_path = os.path.join(SAMPLES_DIR, f"video{epoch}.mp4")
            batch_to_video(recon_batch, out_path, fps=30.0, sample_idx=0)
            print(f"  Saved sample to {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
