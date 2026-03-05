import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'inference'))

from diffusion.test_train_step import BATCH_SIZE
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


NUM_EPOCHS = 100
PER_DEVICE_BATCH_SIZE = 1
MAX_FRAMES = 32
RESIZE = (256, 256)
LEARNING_RATE = 6e-5
DECAY_STEPS = 1_000_000
VAE_PATH = "/mnt/t9/vae_longterm_saves/gcs2/checkpoint_step_130000"
SHUFFLE = True
NUM_WORKERS = 4
PREFETCH_SIZE = 16



if __name__ == "__main__":
    import argparse
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".99"

    print(f"[{os.uname().nodename}] Starting distributed_train.py...", flush=True)
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    #jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
    print(f"[{os.uname().nodename}] JAX imported, initializing distributed...", flush=True)
    # Initialize distributed JAX BEFORE any device access.
    # On TPU pods this auto-detects coordinator, process id, and peer count.
    jax.distributed.initialize()
    print(f"[{os.uname().nodename}] Distributed initialized! Process {jax.process_index()}/{jax.process_count()}", flush=True)
    import orbax.checkpoint as ocp
    os.environ.setdefault("WANDB_API_KEY", "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")
    import wandb
    import math
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


    if process_index == 0:
        wandb.init(
            project="distributed-video-vae",
            config={
            },
        )
    if process_index != 0:
        import time
        time.sleep(10)
    if process_index == 0:
        print("Wandb init complete, proceeding.")

    def shard_batch(batch):
        sharded = {}
        for key, val in batch.items():
            ndim = val.ndim
            spec = P('data', *([None] * (ndim - 1)))
            s = NamedSharding(mesh, spec)
            sharded[key] = jax.make_array_from_process_local_data(s, val)
        return sharded

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

    height, width = RESIZE
    patch_size = 16
    VAE = VideoVAE(
        height=height, width=width, channels=3, patch_size=patch_size,
        encoder_depth=9, decoder_depth=12, mlp_dim=1536, num_heads=8,
        qkv_features=512, max_temporal_len=64,
        spatial_compression_rate=8, unembedding_upsample_rate=4,
        rngs=nnx.Rngs(2),
    )

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
    gdef, state = nnx.split(VAE)
    state = jax.device_put(state, replicated_sharding)
    VAE = nnx.merge(gdef, state)

    optimizer_discard = nnx.Optimizer(VAE, optimizer_def)
    print(f"OPTIMIZER: {optimizer_discard.model is VAE}")

    load_checkpoint_fn(VAE, optimizer_discard, VAE_PATH)
    if args.model_path is not None:
        SEED = hash(args.model_path) % (2**31)
        rngs = nnx.Rngs(3)

    DiT = VideoDiT(hw = 256, residual_dim=1024, compressed_channel_dim = 96, depth=24, mlp_dim = 2048, num_heads = 8, 
    qkv_features = 1024, max_temporal_len = 64, rngs = nnx.Rngs(0)) 
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

    gdef, state = nnx.split(DiT)
    state = jax.device_put(state, replicated_sharding)
    DiT = nnx.merge(gdef, state)
    optimizer = nnx.Optimizer(DiT, optimizer_def)
    print(f"OPTIMIZER_DiT: {optimizer.model is DiT}")



    @nnx.jit(static_argnums=(3,))
    def generate(dit, vae, noise, num_steps, compression_mask, selection_indices, video_mask, rngs):
        denoised, sel_pred = sample(dit, noise, compression_mask, num_steps)
        reconstruction = vae.decompress(denoised, video_mask, selection_indices, compression_mask, rngs, train=False)
        return reconstruction


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
        train_dataloader = create_batched_dataloader(
            base_dir=DATA_DIR,
            batch_size=BATCH_SIZE,
            max_frames=MAX_FRAMES,
            resize=RESIZE,
            shuffle=SHUFFLE,
            num_workers=NUM_WORKERS,
            prefetch_size=PREFETCH_SIZE,
            drop_remainder=True,
            seed=SEED + epoch,
        )