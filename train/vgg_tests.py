import jax
import jax.numpy as jnp
import flaxmodels as fm
from einops import rearrange


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
        include_head=False
    )

    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, 224, 224, 3))
    params = model.init(rng, dummy_input)

    return model, params


def get_perceptual_loss(model, params, x, target, layer_weights=None):
    """Compute perceptual loss using VGG features from first 3 layers.

    Args:
        model: VGG model instance (output='activations').
        params: Model parameters.
        x: Predicted images with shape (b, t, h, w, c).
        target: Target images with shape (b, t, h, w, c).
        layer_weights: Optional dict mapping layer names to weights.
                      Defaults to equal weighting of first 3 relu layers.

    Returns:
        Scalar perceptual loss value.
    """
    if layer_weights is None:
        layer_weights = {
            'relu1_1': 1.0,
            'relu1_2': 1.0,
            'relu2_1': 1.0,
        }

    x_flat = rearrange(x, 'b t h w c -> (b t) h w c').astype(jnp.bfloat16)
    target_flat = rearrange(target, 'b t h w c -> (b t) h w c').astype(jnp.bfloat16)

    x_feats = model.apply(params, x_flat)
    target_feats = model.apply(params, target_flat)

    loss = 0.0
    for layer_name, weight in layer_weights.items():
        x_f = x_feats[layer_name]
        t_f = target_feats[layer_name]
        layer_loss = jnp.mean((x_f - t_f) ** 2)
        loss = loss + weight * layer_loss

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

    # Verify params are in fp32
    param_leaves = jax.tree_util.tree_leaves(params)
    print(f"  Params dtype: {param_leaves[0].dtype}")
    assert param_leaves[0].dtype == jnp.float32, "Params should be fp32"

    # Test with different spatial sizes
    for test_h, test_w in [(64, 64), (128, 128), (224, 224)]:
        x = jnp.ones((b, t, test_h, test_w, c))
        x_flat = rearrange(x, 'b t h w c -> (b t) h w c').astype(jnp.bfloat16)
        feats = model.apply(params, x_flat)

        print(f"  Input shape: {x.shape}")
        print(f"  Flattened shape: {x_flat.shape} (dtype: {x_flat.dtype})")
        for layer_name in ['relu1_1', 'relu1_2', 'relu2_1']:
            feat = feats[layer_name]
            print(f"    {layer_name}: {feat.shape} (dtype: {feat.dtype})")
            assert feat.dtype == jnp.bfloat16, f"Activations should be bf16, got {feat.dtype}"

    # Test perceptual loss function
    print("Testing perceptual loss...")
    x = jnp.ones((b, t, 64, 64, c))
    target = jnp.ones((b, t, 64, 64, c)) * 0.5
    loss = get_perceptual_loss(model, params, x, target)
    print(f"  Loss value: {loss}")
    assert loss.shape == (), f"Expected scalar loss, got shape {loss.shape}"

    print("All tests passed!")