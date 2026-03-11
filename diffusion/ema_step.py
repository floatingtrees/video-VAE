import jax
import jax.numpy as jnp
from flax import nnx
from diffusion_model import VideoDiT

@nnx.jit
def ema_step(ema_model, model, moving_average_coef):
    ema_params = nnx.state(ema_model, nnx.Param)
    model_params = nnx.state(model, nnx.Param)
    updated_params = jax.tree.map(
        lambda ema, p: ema * moving_average_coef + (1 - moving_average_coef) * p, ema_params, model_params
    )
    nnx.update(ema_model, updated_params)

if __name__ == "__main__":
    rngs1 = nnx.Rngs(0)
    rngs2 = nnx.Rngs(0)

    model = VideoDiT(
        hw=256, residual_dim=1024, compressed_channel_dim=96, depth=24,
        mlp_dim=2048, num_heads=8, qkv_features=1024, max_temporal_len=64,
        rngs=rngs1
    )

    ema_model = VideoDiT(
        hw=256, residual_dim=1024, compressed_channel_dim=96, depth=24,
        mlp_dim=2048, num_heads=8, qkv_features=1024, max_temporal_len=64,
        rngs=rngs2
    )

    

    # Test: params should start identical (same seed), then diverge after ema step with different weights
    ema_params_before = nnx.state(ema_model, nnx.Param)
    model_params = nnx.state(model, nnx.Param)

    # Verify they start equal
    diffs = jax.tree.map(lambda a, b: jnp.max(jnp.abs(a - b)), ema_params_before, model_params)
    max_diff = max(jax.tree.leaves(diffs))
    print(f"Max param diff before ema_step: {max_diff}")

    # Manually perturb model params so ema_step has an effect
    perturbed = jax.tree.map(lambda p: p + 0.1, model_params)
    nnx.update(model, perturbed)

    ema_step(ema_model, model, 0.9999)

    ema_params_after = nnx.state(ema_model, nnx.Param)
    diffs_after = jax.tree.map(lambda a, b: jnp.max(jnp.abs(a - b)), ema_params_after, ema_params_before)
    max_diff_after = max(jax.tree.leaves(diffs_after))
    print(f"Max param diff after ema_step: {max_diff_after}")
