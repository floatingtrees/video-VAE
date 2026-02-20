import jax
import jax.numpy as jnp
import flaxmodels as fm
from einops import rearrange
from flax import nnx
from functools import partial

def load_vgg(pretrained='imagenet', normalize=True):
    """Load VGG16 model for perceptual loss extraction.

    Args:
        pretrained: Use 'imagenet' for pretrained weights, None for random init.
        normalize: Whether to normalize inputs with ImageNet mean/std.

    Returns:
        Tuple of (model, params) where model returns activations dict.
    """
    model = fm.VGG16(
        output='activations',
        pretrained=pretrained,
        normalize=normalize,
        include_head=False, 
        dtype=jnp.bfloat16
    )

    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, 224, 224, 3))
    params = model.init(rng, dummy_input)

    # Convert params to bf16
    params = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), params)

    return model, params


PERCEPTUAL_LAYERS = ('relu1_1', 'relu1_2', 'relu2_1')

def get_adversarial_perceptual_loss_fn(model):
    """Create a perceptual loss function with model captured in closure.

    Args:
        model: VGG model instance.

    Returns:
        A function (params, x, target) -> loss that can be jit-compiled without static_argnums.
    """
    def perceptual_loss(params, x, target):
        b, t, h, w, c = x.shape
        x_flat = rearrange(x, 'b t h w c -> (b t) h w c').astype(jnp.bfloat16)
        target_flat = rearrange(target, 'b t h w c -> (b t) h w c').astype(jnp.bfloat16)

        def vgg_forward(params, inp):
            return model.apply(params, inp)

        checkpointed_forward = jax.checkpoint(vgg_forward)
        x_feats = checkpointed_forward(params, x_flat)
        target_feats = checkpointed_forward(params, target_flat)

        loss_weird = (
            jnp.mean((x_feats['relu1_1'] - target_feats['relu1_1']) ** 2, axis=tuple(range(1, x_feats['relu1_1'].ndim))) +
            jnp.mean((x_feats['relu1_2'] - target_feats['relu1_2']) ** 2, axis=tuple(range(1, x_feats['relu1_2'].ndim))) +
            jnp.mean((x_feats['relu2_1'] - target_feats['relu2_1']) ** 2, axis=tuple(range(1, x_feats['relu2_1'].ndim)))
        )
        loss_unflattened = rearrange(loss_weird, "(b t) -> b t", b=b, t=t)
        loss = jnp.mean(loss_unflattened, axis = -1)
        return loss

    return perceptual_loss

def get_perceptual_loss_fn(model):
    """Create a perceptual loss function with model captured in closure.

    Args:
        model: VGG model instance.

    Returns:
        A function (params, x, target) -> loss that can be jit-compiled without static_argnums.
    """
    def perceptual_loss(params, x, target):
        x_flat = rearrange(x, 'b t h w c -> (b t) h w c').astype(jnp.bfloat16)
        target_flat = rearrange(target, 'b t h w c -> (b t) h w c').astype(jnp.bfloat16)

        def vgg_forward(params, inp):
            return model.apply(params, inp)

        checkpointed_forward = jax.checkpoint(vgg_forward)
        x_feats = checkpointed_forward(params, x_flat)
        target_feats = checkpointed_forward(params, target_flat)

        loss = (
            jnp.mean((x_feats['relu1_1'] - target_feats['relu1_1']) ** 2) +
            jnp.mean((x_feats['relu1_2'] - target_feats['relu1_2']) ** 2) +
            jnp.mean((x_feats['relu2_1'] - target_feats['relu2_1']) ** 2)
        )
        return loss

    return perceptual_loss


def get_perceptual_loss(model, params, x, target):
    """Compute perceptual loss using VGG features from first 3 layers.

    Args:
        model: VGG model instance (output='activations').
        params: Model parameters.
        x: Predicted images with shape (b, t, h, w, c).
        target: Target images with shape (b, t, h, w, c).

    Returns:
        Scalar perceptual loss value.
    """
    x_flat = rearrange(x, 'b t h w c -> (b t) h w c').astype(jnp.bfloat16)
    target_flat = rearrange(target, 'b t h w c -> (b t) h w c').astype(jnp.bfloat16)

    def vgg_forward(params, x):
        return model.apply(params, x)

    checkpointed_forward = jax.checkpoint(vgg_forward)
    x_feats = checkpointed_forward(params, x_flat)
    target_feats = checkpointed_forward(params, target_flat)

    # Cast activations to bf16 (flaxmodels VGG outputs float32)
    #x_feats = jax.tree_util.tree_map(lambda a: a.astype(jnp.bfloat16), x_feats)
    #target_feats = jax.tree_util.tree_map(lambda a: a.astype(jnp.bfloat16), target_feats)

    loss = (
        jnp.mean((x_feats['relu1_1'] - target_feats['relu1_1']) ** 2) +
        jnp.mean((x_feats['relu1_2'] - target_feats['relu1_2']) ** 2) +
        jnp.mean((x_feats['relu2_1'] - target_feats['relu2_1']) ** 2)
    )

    return loss


if __name__ == "__main__":
    print("Running shape tests...")

    # Test reshape with einops
    b, t, h, w, c = 2, 4, 64, 64, 3
    dummy = jnp.ones((b, t, h, w, c))
    flat = rearrange(dummy, 'b t h w c -> (b t) h w c')
    assert flat.shape == (b * t, h, w, c), f"Expected {(b * t, h, w, c)}, got {flat.shape}"
    print(f"  Reshape test passed: {dummy.shape} -> {flat.shape}")

    # Test model loading and forward pass
    print("Loading VGG model...")
    model, params = load_vgg(pretrained='imagenet', normalize=True)

    # Verify params are in bf16
    param_leaves = jax.tree_util.tree_leaves(params)
    print(f"  Params dtype: {param_leaves[0].dtype}")
    assert param_leaves[0].dtype == jnp.bfloat16, "Params should be bf16"

    # Test with different spatial sizes
    for test_h, test_w in [(64, 64), (128, 128), (224, 224)]:
        x = jnp.ones((b, t, test_h, test_w, c))
        x_flat = rearrange(x, 'b t h w c -> (b t) h w c').astype(jnp.bfloat16)
        feats = model.apply(params, x_flat)
        # Cast activations to bf16 (flaxmodels VGG outputs float32)
        feats = jax.tree_util.tree_map(lambda a: a.astype(jnp.bfloat16), feats)

        print(f"  Input shape: {x.shape}")
        print(f"  Flattened shape: {x_flat.shape} (dtype: {x_flat.dtype})")
        for layer_name in PERCEPTUAL_LAYERS:
            feat = feats[layer_name]
            print(f"    {layer_name}: {feat.shape} (dtype: {feat.dtype})")

    # Test perceptual loss function
    print("Testing perceptual loss...")
    x = jnp.ones((b, t, 64, 64, c))
    target = jnp.ones((b, t, 64, 64, c)) * 0.5
    loss = get_perceptual_loss(model, params, x, target)
    print(f"  Loss value: {loss}")
    assert loss.shape == (), f"Expected scalar loss, got shape {loss.shape}"

    # Test with jax.jit
    print("Testing with jax.jit...")
    jit_perceptual_loss = jax.jit(get_perceptual_loss, static_argnums=(0,))
    loss_jit = jit_perceptual_loss(model, params, x, target)
    print(f"  JIT loss value: {loss_jit}")
    print(jnp.max(jnp.abs(loss - loss_jit)))

    # Test backprop with jit-compiled grad function
    print("Testing backprop with jit...")

    def loss_for_grad(x, target, model, params):
        return get_perceptual_loss(model, params, x, target)

    grad_fn = nnx.jit(jax.grad(loss_for_grad, argnums=0), static_argnums=(2,))

    # Test with increasing batch sizes to find max
    test_b, test_t, test_h, test_w, test_c = 32, 8, 256, 256, 3
    print(f"  Testing backprop with shape ({test_b}, {test_t}, {test_h}, {test_w}, {test_c})...")
    x_large = jnp.ones((test_b, test_t, test_h, test_w, test_c), dtype=jnp.bfloat16)
    target_large = jnp.ones((test_b, test_t, test_h, test_w, test_c), dtype=jnp.bfloat16) * 0.5

    grads = grad_fn(x_large, target_large, model, params)
    print(f"  Gradient shape: {grads.shape}")
    print(f"  Gradient dtype: {grads.dtype}")
    print(f"  Gradient max: {jnp.max(jnp.abs(grads))}")

    print("All tests passed!")