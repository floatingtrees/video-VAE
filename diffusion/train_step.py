from bz2 import compress
from quopri import encodestring
from timeit import default_timer
import jax
#jax.config.update("jax_numpy_rank_promotion", "warn")
import jax.numpy as jnp
from flax import nnx
from beartype import beartype
from jaxtyping import jaxtyped, Float, Array, Int, Bool
from layers import PatchEmbedding, FactoredAttention, GumbelSigmoidSTE, PatchUnEmbedding
from einops import rearrange
from unet import UNet
from einops import repeat, reduce, rearrange
from shift_indices import shift_indices_to_left, convert_to_indices
from autoencoder import VideoVAE
from diffusion_model import VideoDiT


def all_but_first_2_dimension_mean(x):
        return jnp.mean(x, axis=tuple(range(2, x.ndim)))


def loss_fn(DiT, compressed: Float[Array, "b t hw c"], selection_indices: Float[Array, "b t"], compression_mask: Int[Array, "b t"], hparams, rngs):
    key = rngs.sampling()
    timestep_logits = jax.random.normal(key, compressed.shape[0])
    timestep = 1/(1 + jnp.exp(timestep_logits))
    timestep = rearrange(timestep, "b -> b 1")
    key = rngs.sampling()
    noise = jax.random.normal(key, compressed.shape)
    timestep_reshaped = rearrange(timestep, "b 1 -> b 1 1 1")
    interpolation = (1 - timestep_reshaped) * noise + timestep_reshaped * compressed
    movement_vector_prediction, selection_prediction = DiT(interpolation, compression_mask, timestep)
    num_nonzero_timesteps = jnp.maximum(reduce(compression_mask, "b t -> b 1", "sum"), 1)

    movement_vector_target = compressed - noise
    per_frame_error = all_but_first_2_dimension_mean(jnp.square(movement_vector_target - movement_vector_prediction)) * compression_mask
    MSE = jnp.mean(reduce(per_frame_error, "b t -> b 1", "sum") / num_nonzero_timesteps)

    
    batchwise_selection_loss = reduce(jnp.square(selection_prediction - selection_indices) * compression_mask, "b t -> b 1", "sum") / num_nonzero_timesteps
    selection_loss = jnp.mean(batchwise_selection_loss)

    loss = MSE + hparams["lambda1"] * selection_loss


    #jax.debug.print("sel_pred: {} sel_target: {} mask: {}", selection_prediction, selection_indices, compression_mask)
    return loss, {"selection_loss": selection_loss, "MSE": MSE}



def sample(DiT, noise, compression_mask, num_steps):
    dt = 1.0 / num_steps
    b = noise.shape[0]
    init_sel = jnp.zeros((b, compression_mask.shape[1]), dtype=noise.dtype)

    def body_fn(i, carry):
        x, _ = carry
        t = jnp.full((b, 1), i / num_steps)
        velocity, selection_prediction = DiT(x, compression_mask, t)
        return (x + velocity * dt, selection_prediction)

    x, selection_prediction = jax.lax.fori_loop(0, num_steps, body_fn, (noise, init_sel))
    return x, selection_prediction


@nnx.jit(static_argnums=(7,))
def train_step(DiT, VAE, optimizer, video, video_mask, hparams, rngs, deterministic_compress=False):
    sequence_length = video.shape[1]
    compressed, selection_indices, compression_mask = VAE.compress(video, video_mask, rngs, train=not deterministic_compress)
    compressed = compressed[:, :sequence_length // 2, ...]
    selection_indices = selection_indices[:, :sequence_length// 2]
    compression_mask = compression_mask[:, :sequence_length // 2]
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, aux), grads = grad_fn(DiT, compressed, selection_indices, compression_mask, hparams, rngs)
    optimizer.update(grads)
    return loss, aux


if __name__ == "__main__":
    seed = 42
    key = jax.random.key(seed)
    try:
        gpu_device = jax.devices('gpu')[0] # 'cuda' works too, but 'gpu' is the generic backend name
    except RuntimeError:
        raise RuntimeError("No GPU found! Is JAX installed with CUDA support?")
    import optax
    temporal_length = 5
    input_image = jax.random.normal(key, (2, temporal_length, 256, 256, 3)) * 0.02
    VAE = VideoVAE(height=256, width=256, channels=3, patch_size=16,
    encoder_depth=9, decoder_depth=12, mlp_dim=1536, num_heads=8, qkv_features=512,
    max_temporal_len=temporal_length, spatial_compression_rate=8, unembedding_upsample_rate=4, rngs = nnx.Rngs(0), 
    dtype = jnp.bfloat16, param_dtype=jnp.float32)
    attn_mask = jnp.ones((2, 1, 1, temporal_length), dtype=bool)
    DiT = VideoDiT(hw = 256, residual_dim=1024, compressed_channel_dim = 96, depth=2, mlp_dim = 2048, num_heads = 8, 
    qkv_features = 1024, max_temporal_len = 64, rngs = nnx.Rngs(0)) 
    schedule_fn = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=0.1,
        warmup_steps=10,
        decay_steps=10000,
        end_value=0.1 / 10,
    )
    optimizer_def = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=schedule_fn),
    )
    optimizer = nnx.Optimizer(DiT, optimizer_def)
    train_step(DiT, VAE, optimizer, input_image, jnp.ones((2, 1, 1, temporal_length), dtype=bool), {"lambda1": 0.1}, rngs = nnx.Rngs(0))