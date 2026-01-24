
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
NUM_EPOCHS = 100
BATCH_SIZE = 4
MAX_FRAMES = 128
RESIZE = (256, 256)
SHUFFLE = True
NUM_WORKERS = 4
PREFETCH_SIZE = 16
DROP_REMAINDER = True
SEED = 42
WARMUP_STEPS = 5000
DECAY_STEPS = 100_000
GAMMA1 = 0.05
GAMMA2 = 0.001
LEARNING_RATE = 1e-4
VIDEO_SAVE_DIR = "outputs"
max_compression_rate = 1.2






def load_checkpoint(model, optimizer, path):                                                                       
      abstract_state = {                                                                                             
          "model": jax.tree.map(ocp.utils.to_shape_dtype_struct, nnx.state(model)),                                  
          "optimizer": jax.tree.map(ocp.utils.to_shape_dtype_struct, nnx.state(optimizer))                           
      }                                                                                                              
      restored = ocp.StandardCheckpointer().restore(path, abstract_state, partial_restore=True)                                            
      nnx.update(model, restored["model"])                                                                           
      nnx.update(optimizer, restored["optimizer"])   

if __name__ == "__main__":
    schedule_fn= optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        decay_steps=DECAY_STEPS,
        end_value=LEARNING_RATE / 10,
        )


    model_save_path = '/mnt/t9/video_vae_saves/'
    model = VideoVAE(height=256, width=256, channels=3, patch_size=16, 
            depth=6, mlp_dim=1024, num_heads=8, qkv_features=128,
            max_temporal_len=MAX_FRAMES, spatial_compression_rate=4, rngs = nnx.Rngs(1))
    optimizer_def = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate=schedule_fn)
        )
    optimizer = nnx.Optimizer(model, optimizer_def)

    load_checkpoint(model, optimizer, f"{model_save_path}/checkpoint_{0}")