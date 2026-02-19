import sys
import time
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from flax import nnx
import optax
from rl_model import VideoVAE

# Multi-process parameters from sys.argv
proc_id = int(sys.argv[1])
num_procs = int(sys.argv[2])

# Initialize JAX distributed
jax.distributed.initialize('localhost:10000', num_procs, proc_id)

num_devices = jax.device_count()
print(f"Process {proc_id}: {jax.local_device_count()} local devices, {num_devices} total devices")

# Create a mesh for data parallelism across all devices
mesh = jax.make_mesh((num_devices,), ('data',)) 

# Tiny model dimensions for testing
HEIGHT = 32
WIDTH = 32
CHANNELS = 3
PATCH_SIZE = 8
ENCODER_DEPTH = 1
DECODER_DEPTH = 1
MLP_DIM = 64
NUM_HEADS = 2
QKV_FEATURES = 32
TEMPORAL_LEN = 4
SPATIAL_COMPRESSION_RATE = 2
UPSAMPLE_RATE = 2
BATCH_SIZE = 2
LR = 1e-4
NUM_STEPS = 5

# Build model
rngs = nnx.Rngs(42)
model = VideoVAE(
    height=HEIGHT, width=WIDTH, channels=CHANNELS, patch_size=PATCH_SIZE,
    encoder_depth=ENCODER_DEPTH, decoder_depth=DECODER_DEPTH,
    mlp_dim=MLP_DIM, num_heads=NUM_HEADS, qkv_features=QKV_FEATURES,
    max_temporal_len=TEMPORAL_LEN,
    spatial_compression_rate=SPATIAL_COMPRESSION_RATE,
    unembedding_upsample_rate=UPSAMPLE_RATE,
    rngs=rngs, dtype=jnp.float32, param_dtype=jnp.float32,
)

# Count params
params_state = nnx.state(model, nnx.Param)
num_params = sum(x.size for x in jax.tree_util.tree_leaves(params_state))
if proc_id == 0:
    print(f"Model parameters: {num_params:,}")

# Replicate model state across all devices
replicate_sharding = NamedSharding(mesh, P())
graphdef, state = nnx.split(model)
state = jax.device_put(state, replicate_sharding)
model = nnx.merge(graphdef, state)

# Create optimizer (pure functional optax)
tx = optax.adam(LR)
graphdef, params, rest = nnx.split(model, nnx.Param, ...)
opt_state = tx.init(params)
opt_state = jax.device_put(opt_state, replicate_sharding)


def compute_loss(params, rest, graphdef, x, mask, rng_key):
    model = nnx.merge(graphdef, params, rest)
    rngs = nnx.Rngs(sampling=rng_key)
    reconstruction, compressed, selection, selection_mask, log_variance, mean = model(x, mask, rngs, train=True)
    # Reconstruction is shape (b*2, t, h, w, c) due to repeat in VideoVAE
    x_repeated = jnp.concatenate([x, x], axis=0)
    recon_loss = jnp.mean((reconstruction - x_repeated) ** 2)
    # KL divergence
    kl_loss = -0.5 * jnp.mean(1 + log_variance - mean ** 2 - jnp.exp(log_variance))
    # Selection sparsity reward (RL-style: encourage fewer tokens selected)
    selection_cost = jnp.mean(selection)
    total_loss = recon_loss + 0.001 * kl_loss + 0.01 * selection_cost
    return total_loss, (recon_loss, kl_loss, selection_cost)


@jax.jit
def train_step(params, rest, graphdef, opt_state, x, mask, rng_key):
    grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
    (loss, aux), grads = grad_fn(params, rest, graphdef, x, mask, rng_key)
    updates, new_opt_state = tx.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return loss, aux, new_params, new_opt_state


# Training loop - each process works on its own data shard
key = jax.random.key(0)
for step in range(NUM_STEPS):
    key, data_key, step_key = jax.random.split(key, 3)
    # Each process uses same key -> same data (replicated across devices)
    x = jax.random.normal(data_key, (BATCH_SIZE, TEMPORAL_LEN, HEIGHT, WIDTH, CHANNELS)) * 0.02
    mask = jnp.ones((BATCH_SIZE, 1, 1, TEMPORAL_LEN), dtype=jnp.bool_)

    # Replicate data across all devices
    x = jax.device_put(x, replicate_sharding)
    mask = jax.device_put(mask, replicate_sharding)

    start = time.perf_counter()
    loss, (recon_loss, kl_loss, sel_cost), params, opt_state = train_step(
        params, rest, graphdef, opt_state, x, mask, step_key
    )
    loss.block_until_ready()
    elapsed = time.perf_counter() - start

    if proc_id == 0:
        print(f"Step {step}: loss={float(loss):.4f} recon={float(recon_loss):.4f} "
              f"kl={float(kl_loss):.4f} sel={float(sel_cost):.4f} time={elapsed:.3f}s")

if proc_id == 0:
    print("Training complete!")
