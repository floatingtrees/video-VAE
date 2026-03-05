from bz2 import compress
import jax
#jax.config.update("jax_numpy_rank_promotion", "warn")
import jax.numpy as jnp
from flax import nnx
from beartype import beartype
from jaxtyping import jaxtyped, Float, Array, Int, Bool
from layers import PatchEmbedding, FactoredAttention, GumbelSigmoidSTE, PatchUnEmbedding
from einops import rearrange


# AE outputs (2, t, hw, 96), hw = 256
class VideoDiT(nnx.Module):
    def __init__(self, hw, residual_dim, compressed_channel_dim, depth,
    mlp_dim, num_heads, qkv_features, max_temporal_len, 
    rngs: nnx.Rngs, dtype: jnp.dtype = jnp.bfloat16, param_dtype: jnp.dtype = jnp.float32):
        super().__init__()

        layers = []
        self.timestep_proj = nnx.Linear(1, residual_dim, kernel_init=nnx.initializers.zeros, bias_init=nnx.initializers.zeros,rngs=rngs)
        self.last_dim = residual_dim
        self.up_proj = nnx.Linear(compressed_channel_dim, residual_dim, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        for _ in range(depth):
            layers.append(FactoredAttention(mlp_dim = mlp_dim,
                in_features = self.last_dim,
                num_heads = num_heads,
                qkv_features = qkv_features,
                max_temporal_len = max_temporal_len,
                max_spatial_len = hw,
                rngs = rngs,
                dtype=dtype,
                param_dtype=param_dtype
            ))
        self.down_proj = nnx.Linear(residual_dim, compressed_channel_dim, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        self.spacing_pred1 = nnx.Linear(residual_dim, 1, dtype = dtype, param_dtype=param_dtype, rngs=rngs)
        self.spacing_pred2 = nnx.Linear(hw, 1, dtype = dtype, param_dtype=param_dtype, rngs=rngs)
        self.layers = layers

    def __call__(self, compressed: Float[Array, "b t hw d"], compression_mask: Bool[Array, "b t"], time: Float[Array, "b 1"]):
        compression_mask = rearrange(compression_mask, "b t -> b 1 1 t")
        timestep_weights = rearrange(self.timestep_proj(time), "b d -> b 1 1 d")
        x = self.up_proj(compressed) + timestep_weights
        for layer in self.layers:
            x = layer(x, compression_mask)
        latent_prediction = self.down_proj(x)
        spacing_reduce1 = rearrange(self.spacing_pred1(x), "b t hw 1 -> b t hw")
        spacing_reduce2 = rearrange(self.spacing_pred2(spacing_reduce1), "b t 1 -> b t")
        return latent_prediction, spacing_reduce2

if __name__ == "__main__":
    #jax.config.update("jax_enable_x64", True)
    seed = 42
    key = jax.random.key(seed)
    try:
        gpu_device = jax.devices('gpu')[0] # 'cuda' works too, but 'gpu' is the generic backend name
    except RuntimeError:
        raise RuntimeError("No GPU found! Is JAX installed with CUDA support?")
    temporal_length = 9
    compression_mask = jnp.ones((2, temporal_length), dtype=bool)
    compression_mask = compression_mask.at[:, 5:].set(False)
    input_image = jax.random.normal(key, (2, temporal_length, 256, 96)) * 0.02
    timestep = jax.random.uniform(key, shape=(2, 1), minval=0.0, maxval=1.0)

    DiT = VideoDiT(hw = 256, residual_dim=1024, compressed_channel_dim = 96, depth=24, mlp_dim = 2048, num_heads = 8, 
    qkv_features = 1024, max_temporal_len = 64, rngs = nnx.Rngs(0)) 


    params_state = nnx.state(DiT, nnx.Param)
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(params_state))
    print(num_params)
    output, spacing_preds = DiT(input_image, compression_mask, timestep)
    @nnx.jit
    def forward(model, compressed: Float[Array, "b t hw d"], compression_mask: Bool[Array, "b t"], time: Float[Array, "b 1"]):
        return model(compressed, compression_mask, time)

    output, spacing_preds = forward(DiT, input_image, compression_mask, timestep)
    output_masked, spacing_preds_masked = forward(DiT, input_image[:, :5], compression_mask[:, :5], timestep)


    print(compression_mask[:, :5])
    print(jnp.max(jnp.abs(output[:, :5] - output_masked)))
    print(jnp.allclose(output[:, :5], output_masked))
    print(jnp.allclose(spacing_preds[:, :5], spacing_preds_masked))

