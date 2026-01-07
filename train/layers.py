import jax
import jax.numpy as jnp
from flax import nnx
from beartype import beartype
from jaxtyping import jaxtyped, Float, Array
from einops import rearrange

class PatchEmbedding(nnx.Module):
    def __init__(self, height, width, channels, patch_size, rngs: nnx.Rngs):
        super().__init__()
        self.patch_size = patch_size
        
        self.linear = nnx.Linear(patch_size * patch_size * channels, patch_size * patch_size * channels, rngs = rngs)
        self.norm = nnx.LayerNorm(patch_size * patch_size * channels, rngs = rngs)
        
    @jaxtyped(typechecker=beartype)
    def __call__(self, x: Float[Array, "b time height width channels"]):
        x = rearrange(x, "b t (h p1) (w p2) c -> b t (h w) (p1 p2 c)",
            p1 = self.patch_size, p2 = self.patch_size)
        x = self.linear(x)
        x = self.norm(x)
        return x

import jax
import jax.numpy as jnp
from flax import nnx
from beartype import beartype
from jaxtyping import Array, Float

def create_sinusoidal_embeddings(
    seq_len: int,
    embed_dim: int,
    max_timescale: float = 10000.0
) -> Array:
    position = jnp.arange(seq_len, dtype=jnp.float32)[:, jnp.newaxis]

    div_term = jnp.exp(
        jnp.arange(0, embed_dim, 2, dtype=jnp.float32) * -(jnp.log(max_timescale) / embed_dim)
    )

    scaled_time = position * div_term[jnp.newaxis, :]
    
    sin_part = jnp.sin(scaled_time)
    cos_part = jnp.cos(scaled_time)

    pe = jnp.stack([sin_part, cos_part], axis=-1)
    
    pe = pe.reshape(seq_len, embed_dim)

    return pe[jnp.newaxis, :, :]

class SinusoidalPositionalEncoding(nnx.Module):
    def __init__(self, max_len: int, embed_dim: int):
        self.pe_table = create_sinusoidal_embeddings(max_len, embed_dim)

    def __call__(self, x: Float[Array, "batch seq dim"]) -> Float[Array, "batch seq dim"]:
        seq_len = x.shape[1]
        return x + self.pe_table[:, :seq_len, :]

class Attention(nnx.Module):
    def __init__(self, in_features, inner_dim, num_heads, qkv_features, max_len, rngs: nnx.Rngs):
        super().__init__()
        self.MHA = nnx.MultiHeadAttention(num_heads = num_heads, 
            in_features = in_features, qkv_features = qkv_features, 
            rngs = rngs, decode = False)
        self.PE = SinusoidalPositionalEncoding(max_len = max_len, embed_dim = in_features)
        
    def __call__(self, x: Float[Array, "a seq dim"]):
        x = self.PE(x)
        attn_output = self.MHA(x)
        return attn_output

class MLP(nnx.Module):
    def __init__(self, in_features, mlp_dim, rngs: nnx.Rngs):
        super().__init__()
        self.norm = nnx.LayerNorm(in_features, rngs = rngs)
        self.linear1 = nnx.Linear(in_features, mlp_dim, rngs = rngs)
        
        self.linear2 = nnx.Linear(mlp_dim, in_features, rngs = rngs)
        
    def __call__(self, x: Float[Array, "b time hw channels"]):
        x = self.norm(x)
        x = self.linear1(x)
        x = nnx.silu(x)
        x = self.linear2(x)
        return x

class FactoredAttention(nnx.Module):
    def __init__(self, mlp_dim, in_features, inner_dim, num_heads, qkv_features, max_temporal_len, max_spatial_len, rngs: nnx.Rngs):
        super().__init__()
        self.SpatialAttention = Attention(in_features, inner_dim, num_heads, qkv_features, max_spatial_len, rngs)
        self.SpatialMLP = MLP(in_features, mlp_dim, rngs)
        self.TemporalAttention = Attention(in_features, inner_dim, num_heads, qkv_features, max_temporal_len, rngs)
        self.TemporalMLP = MLP(in_features, mlp_dim, rngs)

    def __call__(self, x: Float[Array, "b time hw channels"]):
        print(x.shape)
        b, t, hw, c = x.shape
        temporal_x = rearrange(x, "b t hw c -> (b hw) t c")
        temporal_attn_output = self.TemporalAttention(temporal_x)
        temporal_x = temporal_x + temporal_attn_output
        temporal_x = temporal_x + self.TemporalMLP(temporal_x)

        original_shape_x = rearrange(temporal_x, "(b hw) t c -> b t hw c", b = b, hw = hw)

        spatial_x = rearrange(original_shape_x, "b t hw c -> (b t) hw c")
        spatial_attn_output = self.SpatialAttention(spatial_x)
        spatial_x = spatial_x + spatial_attn_output
        spatial_x = spatial_x + self.SpatialMLP(spatial_x)
        
        original_shape_x = rearrange(spatial_x, "(b t) hw c -> b t hw c", b = b, t = t)
        return original_shape_x


@jax.custom_vjp
def gumbel_softmax_ste(
    logits: Array,
    temperature: float,
    gumbel_noise: Array,
) -> Array:
    """
    Forward pass: Compute soft probabilities, then discretize (Hard).
    """
    # 1. Compute Softmax (Continuous)
    # shape: (..., num_classes)
    y_soft = jax.nn.softmax((logits + gumbel_noise) / temperature)

    # 2. Compute Hard Sample (Discrete)
    # We use argmax + one_hot to get the discrete output
    k = y_soft.shape[-1]
    y_hard_indices = jnp.argmax(y_soft, axis=-1)
    y_hard = jax.nn.one_hot(y_hard_indices, k)

    # 3. Save state for backward pass
    # We save 'y_soft' because the gradient depends on the continuous distribution,
    # not the hard one-hot vector.
    return y_hard


def _gumbel_ste_fwd(logits, temperature, gumbel_noise):
    """
    Forward pass: Compute soft probabilities, then discretize (Hard).
    """
    # 1. Compute Softmax (Continuous)
    # shape: (..., num_classes)
    y_soft = jax.nn.softmax((logits + gumbel_noise) / temperature)

    # 2. Compute Hard Sample (Discrete)
    # We use argmax + one_hot to get the discrete output
    k = y_soft.shape[-1]
    y_hard_indices = jnp.argmax(y_soft, axis=-1)
    y_hard = jax.nn.one_hot(y_hard_indices, k)

    # 3. Save state for backward pass
    # We save 'y_soft' because the gradient depends on the continuous distribution,
    # not the hard one-hot vector.
    return y_hard, (y_soft, temperature)


def _gumbel_ste_bwd(residuals, grad_output):
    """
    Backward pass: Manually compute gradients for Softmax w.r.t logits.
    """
    y_soft, temperature = residuals

    # The Gradient Math (Softmax Jacobian-Vector Product):
    # We want dL/dLogits. We have dL/dOutput (grad_output).
    # Since this is STE, we pretend Output = y_soft during backward.
    # d(Softmax)_ij = y_i * (delta_ij - y_j)
    # VJP = (grad_output - sum(grad_output * y_soft)) * y_soft

    # 1. Compute sum(grad_output * y_soft) across classes
    # shape: (..., 1)
    dot = jnp.sum(grad_output * y_soft, axis=-1, keepdims=True)

    # 2. Compute VJP
    # shape: (..., num_classes)
    grad_softmax = (grad_output - dot) * y_soft

    # 3. Adjust for temperature (Chain rule: inner derivative is 1/temp)
    grad_logits = grad_softmax / temperature

    # Return gradients for (logits, temperature, gumbel_noise)
    # We return None for temp and noise as we aren't optimizing them here.
    return grad_logits, None, None


# Register the custom VJP
gumbel_softmax_ste.defvjp(_gumbel_ste_fwd, _gumbel_ste_bwd)

# --- 2. The NNX Layer ---

class GumbelSoftmaxSTE(nnx.Module):
    """
    Straight-Through Gumbel-Softmax Layer.
    Compatible with Flax NNX.
    """
    @beartype
    def __init__(self, temperature: float = 1.0):
        # Temperature is a static configuration here, but could be a param if desired.
        self.temperature = temperature

    @beartype
    def __call__(
        self,
        logits: Float[Array, "... num_classes"],
        rngs: nnx.Rngs
    ) -> Float[Array, "... num_classes"]:
        """
        Args:
            logits: Unnormalized log-probabilities.
            rngs: NNX Rngs object with 'sampling' key.

        Returns:
            One-hot encoded hard samples (forward), with softmax gradients (backward).
        """
        # 1. Sample Gumbel Noise
        # Using a stable method to generate Gumbel(0, 1)
        # u ~ Uniform(0, 1)
        key = rngs.sampling()
        u = jax.random.uniform(key, logits.shape)
        
        # Limit u to avoid log(0) issues
        epsilon = 1e-20
        gumbel_noise = -jnp.log(-jnp.log(u + epsilon) + epsilon)

        # 2. Apply Custom Op
        return gumbel_softmax_ste(logits, self.temperature, gumbel_noise)