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
from dataloader import create_batched_dataloader, batch_to_video, VideoDataSource
from einops import rearrange, repeat
import time
from ema_step import ema_step

NUM_EPOCHS = 100
PER_DEVICE_BATCH_SIZE = 4
REPITITION_CONSTANT = 1
MAX_FRAMES = 32
RESIZE = (256, 256)
LEARNING_RATE = 6e-5
DECAY_STEPS = 1_500_000
VAE_PATH = "gs://tpus-487818-checkpoints/run1772595923/perceptual_loss_model/checkpoint_step_290000/"
SHUFFLE = True
NUM_WORKERS = 16
PREFETCH_SIZE = 32
WEIGHT_DECAY = 0.01
SEED = 32
hparams = {
    "lambda1": 0.01
}

@nnx.jit(static_argnums=(3,))
def generate(dit, vae, noise, num_steps, rngs):
    compression_mask = repeat(jnp.arange(noise.shape[1]), "t -> b t", b = noise.shape[0])
    compression_mask = (compression_mask <= 10).astype(jnp.bool)
    denoised, sel_pred = sample(dit, noise, compression_mask, num_steps)
    sel_indices = jnp.round(sel_pred).astype(jnp.int32)
    # First element is absolute index (>= 0), rest are gaps (>= 1)
    sel_indices = sel_indices.at[:, 0].set(jnp.maximum(sel_indices[:, 0], 0))
    sel_indices = sel_indices.at[:, 1:].set(jnp.maximum(sel_indices[:, 1:], 1))
    # Derive video_mask: pretend last kept frame is the last frame
    last_frame_pos = jnp.sum(sel_indices * compression_mask, axis=1)  # (b,)
    t = noise.shape[1]
    video_mask = (jnp.arange(t) <= last_frame_pos[:, None])  # (b, t)
    video_mask = rearrange(video_mask, "b t -> b 1 1 t")
    video_mask = jnp.ones(video_mask.shape, dtype = bool)
    reconstruction = vae.decompress(denoised, video_mask, sel_indices, compression_mask, rngs, train=True)
    return reconstruction, video_mask


RUN_TIMESTAMP = int(time.time())
GCS_RUN_DIR = f"gs://tpus-487818-checkpoints/diffusion_run{RUN_TIMESTAMP}"
model_save_path = f"{GCS_RUN_DIR}/model"
VIDEO_SAVE_DIR = f"{GCS_RUN_DIR}/images"
DATA_DIR = os.path.expanduser("~/data/videos")
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
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--per_device_batch_size", type=int, default=PER_DEVICE_BATCH_SIZE)
    parser.add_argument("--max_frames", type=int, default=MAX_FRAMES)
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--switch_to_uniform", action="store_true")
    args = parser.parse_args()

    PER_DEVICE_BATCH_SIZE = args.per_device_batch_size
    LOCAL_BATCH_SIZE = PER_DEVICE_BATCH_SIZE * local_devices
    GLOBAL_BATCH_SIZE = LOCAL_BATCH_SIZE * num_processes
    MAX_FRAMES = args.max_frames
    DATA_DIR = args.data_dir
    if args.switch_to_uniform:
        hparams["noise_alpha"] = 0.9
    else:
        hparams["noise_alpha"] = 1
    WARMUP_STEPS = int(200000 / math.sqrt(GLOBAL_BATCH_SIZE))

    
    GCS_BUCKET = "tpus-487818-training-data"
    GCS_MOUNT_POINT = os.path.expanduser("~/data")
    total_videos = len(VideoDataSource(DATA_DIR))
    train_dataloader = create_batched_dataloader(
            base_dir=DATA_DIR,
            batch_size=LOCAL_BATCH_SIZE // REPITITION_CONSTANT,
            max_frames=MAX_FRAMES,
            resize=RESIZE,
            shuffle=SHUFFLE,
            num_workers=NUM_WORKERS,
            prefetch_size=PREFETCH_SIZE,
            drop_remainder=True,
            seed=SEED,
            gcs_bucket=GCS_BUCKET,
            gcs_mount_point=GCS_MOUNT_POINT,
        )

    if process_index == 0:
        wandb.init(
            project="distributed-video-dit",
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

    LOCAL_CHECKPOINT_DIR = "/tmp/checkpoints"

    def save_checkpoint(model, optimizer, gcs_path):
        import subprocess, shutil
        local_path = os.path.join(LOCAL_CHECKPOINT_DIR, os.path.basename(gcs_path))
        state = {"model": nnx.state(model), "optimizer": nnx.state(optimizer)}
        # Convert to numpy to bypass orbax's JaxArrayHandler which has a
        # set_mesh context manager bug in orbax 0.11.33 + JAX 0.6.2.
        state = jax.tree.map(lambda x: np.array(x), state)
        ckptr = ocp.StandardCheckpointer()
        ckptr.save(local_path, state)
        ckptr.wait_until_finished()
        if process_index == 0:
            subprocess.run(
                ["gcloud", "storage", "cp", "-r", local_path, gcs_path, "--quiet"],
                check=True,
            )
        shutil.rmtree(local_path, ignore_errors=True)
        jax.experimental.multihost_utils.sync_global_devices(f"checkpoint_save_{os.path.basename(gcs_path)}")

    def save_model(model, gcs_path):
        import subprocess, shutil
        local_path = os.path.join(LOCAL_CHECKPOINT_DIR, os.path.basename(gcs_path))
        state = {"model": nnx.state(model)}
        state = jax.tree.map(lambda x: np.array(x), state)
        ckptr = ocp.StandardCheckpointer()
        ckptr.save(local_path, state)
        ckptr.wait_until_finished()
        if process_index == 0:
            subprocess.run(
                ["gcloud", "storage", "cp", "-r", local_path, gcs_path, "--quiet"],
                check=True,
            )
        shutil.rmtree(local_path, ignore_errors=True)
        jax.experimental.multihost_utils.sync_global_devices(f"model_save_{os.path.basename(gcs_path)}")

    def load_model(model, path):
        abstract_state = {
            "model": jax.tree.map(ocp.utils.to_shape_dtype_struct, nnx.state(model)),
        }
        if process_index == 0:
            from etils import epath
            handler = ocp.StandardCheckpointHandler()
            restored = handler.restore(
                epath.Path(path),
                args=ocp.args.StandardRestore(abstract_state),
            )
        else:
            restored = jax.tree.map(lambda x: np.zeros(x.shape, dtype=x.dtype), abstract_state)
        restored = jax.experimental.multihost_utils.broadcast_one_to_all(restored)
        nnx.update(model, restored["model"])

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
    del optimizer_discard
    import gc 
    gc.collect()
    if process_index == 0:
        print("Sleeping after gc")
    time.sleep(5)

    DiT = VideoDiT(hw = 256, residual_dim=1024, compressed_channel_dim = 96, depth=30, mlp_dim = 2048, num_heads = 8, 
    qkv_features = 1024, max_temporal_len = 64, rngs = nnx.Rngs(0)) 

    params_state = nnx.state(DiT, nnx.Param)
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(params_state))
    if process_index == 0:
        print(f"DiT parameters: {num_params:,}")


    schedule_fn = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        decay_steps=DECAY_STEPS,
        end_value=LEARNING_RATE / 10,
    )
    optimizer_def = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule_fn, weight_decay = WEIGHT_DECAY),
    )



    gdef, state = nnx.split(DiT)
    state = jax.device_put(state, replicated_sharding)
    DiT = nnx.merge(gdef, state)
    optimizer = nnx.Optimizer(DiT, optimizer_def)
    print(f"OPTIMIZER_DiT: {optimizer.model is DiT}")

    if args.model_path is not None:
        load_checkpoint_fn(DiT, optimizer, args.model_path)
        SEED = (hash(args.model_path)  + process_index * 10912785)% (2**31)
        rngs = nnx.Rngs(SEED)
    else:
        rngs = nnx.Rngs(process_index)

    if args.reset:
        optimizer = nnx.Optimizer(DiT, optimizer_def)
        print(f"OPTIMIZER_DiT2: {optimizer.model is DiT}")


    
    master_weights= VideoDiT(hw = 256, residual_dim=1024, compressed_channel_dim = 96, depth=30, mlp_dim = 2048, num_heads = 8, 
    qkv_features = 1024, max_temporal_len = 64, rngs = nnx.Rngs(0)) 
    gdef, state = nnx.split(master_weights)
    state = jax.device_put(state, replicated_sharding)
    master_weights = nnx.merge(gdef, state)
    if args.model_path is not None:
        master_path = f"{args.model_path}_master"
        from etils import epath
        if epath.Path(master_path).exists():
            load_model(master_weights, master_path)
        else:
            if process_index == 0:
                print(f"Master checkpoint not found at {master_path}, copying from DiT")
            ema_step(master_weights, DiT, 0.0)

    cached_ema_step = nnx.cached_partial(ema_step, master_weights, DiT)

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

    ### Generate reference compressed tensor to adapt to tensor sharding on the fly

    key = rngs.sampling()
    REF_attn_mask = jnp.ones((2, 1, 1, MAX_FRAMES), dtype=bool)
    REF_input_image = jax.random.normal(key, (2, MAX_FRAMES, 256, 256, 3)) * 0.02
    REF_compressed, REF_selection_indices, REF_compression_mask = VAE.compress(REF_input_image, REF_attn_mask, rngs = nnx.Rngs(0))

    ### 
    
    start = time.perf_counter()
    global_step = 0
    for epoch in range(NUM_EPOCHS):
        
        
        steps_per_epoch = total_videos // (LOCAL_BATCH_SIZE // REPITITION_CONSTANT * num_processes)

        
        for i, batch in enumerate(train_dataloader):
            if i > steps_per_epoch:
                break

            global_step += 1
            if global_step % (10000) == 0:
                save_checkpoint(DiT, optimizer,
                                f"{model_save_path}/checkpoint_step_{global_step}")
                save_model(master_weights, f"{model_save_path}/checkpoint_step_{global_step}_master")
                if process_index == 0:
                    print(f"Saved checkpoint at global_step {global_step}", flush=True)

            
            if i % 200 == 1:
                params = nnx.state(DiT, nnx.Param)
                param_norm = sum(float(jnp.linalg.norm(x)) for x in jax.tree_util.tree_leaves(params))
                if process_index == 0:
                    print(f"  param_norm={param_norm:.4f}")

                params = nnx.state(master_weights, nnx.Param)
                param_norm = sum(float(jnp.linalg.norm(x)) for x in jax.tree_util.tree_leaves(params))
                if process_index == 0:
                    print(f"  master_norm={param_norm:.4f}")

            
            # Shard batch across all devices

            global_batch = shard_batch(batch)
            video = global_batch["video"].astype(jnp.bfloat16)
            mask = global_batch["mask"].astype(jnp.bool_)
            video = repeat(video, "b t h w c -> (b r) t h w c", r=REPITITION_CONSTANT)
            mask = repeat(mask, "b t -> (b r) t", r=REPITITION_CONSTANT)
            video_mask = rearrange(mask, "b time -> b 1 1 time")
            hparams["noise_alpha"] = min(hparams["noise_alpha"] + 1e-5, 1)
            hparams["noise_alpha"] = 0
            loss, aux = train_step(DiT, VAE, optimizer, video, video_mask, hparams, rngs = rngs)
            if i % 10 == 1:
                cached_ema_step(0.9999 ** 10)
            

            if i % 1000 == 0:
                print(f"  [worker {process_index}] heartbeat step={i} global_step={global_step}", flush=True)

            # Logging (process 0 only)
            if process_index == 0 and i % 50 == 0:
                elapsed = time.perf_counter() - start
                log_dict = {
                    "loss": float(loss),
                    "MSE": float(aux["MSE"]),
                    "selection_loss": float(aux["selection_loss"]),
                    "epoch": epoch,
                    "step_in_epoch": i,
                    "global_step": global_step,
                    "elapsed_time": elapsed,
                    "learning_rate": float(schedule_fn(global_step)),
                }
                wandb.log(log_dict, step=global_step)
                print(f"  Step {i}: loss={log_dict['loss']:.4f} "
                      f"MSE={log_dict['MSE']:.4f} "
                      f"sel={log_dict['selection_loss']:.4f} "
                      f"lr={log_dict['learning_rate']:.2e} "
                      f"time={elapsed:.1f}s "
                      f"global_step={global_step}", flush=True)

            if i % 500 == 0: # Frontload the generate compilation
                # All workers materialize arrays to match any implicit collectives
                # (np.array on sharded JAX arrays can trigger all-gathers)
                key = rngs.sampling()
                noise = jax.random.normal(key, REF_compressed.shape)
                reconstruction, video_mask = generate(master_weights, VAE, noise, 100, rngs)
                recon_local = np.array(reconstruction[:PER_DEVICE_BATCH_SIZE])
                mask_local = np.array(rearrange(video_mask[:PER_DEVICE_BATCH_SIZE], "b 1 1 t -> b t"))
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

            


        # Save checkpoint (orbax coordinates internally, all processes must call)
        save_checkpoint(DiT, optimizer, f"{model_save_path}/checkpoint_step_{epoch}")
        save_model(master_weights, f"{model_save_path}/checkpoint_step_{epoch}_master")
        if process_index == 0:
            print(f"Saved checkpoint for epoch {epoch}")
