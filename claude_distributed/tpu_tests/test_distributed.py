"""
Distributed smoke-test using only random tensors — no video data required.

Single device / local:
    python3 tpu_tests/test_distributed.py --small

TPU pod (run on every host — jax.distributed.initialize() auto-configures):
    python3 tpu_tests/test_distributed.py
    python3 tpu_tests/test_distributed.py --small

Local multi-process (2 simulated processes):
    JAX_COORDINATOR_ADDRESS=localhost:8476 JAX_NUM_PROCESSES=2 JAX_PROCESS_ID=0 python3 tpu_tests/test_distributed.py &
    JAX_COORDINATOR_ADDRESS=localhost:8476 JAX_NUM_PROCESSES=2 JAX_PROCESS_ID=1 python3 tpu_tests/test_distributed.py &
    wait
"""

import os
import sys
import time
import argparse
import inspect

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".99")

# flax 0.10.4 passes abstracted_axes to jax.jit which some JAX versions removed.
# Patch it out before any flax/nnx imports so nnx.jit works correctly.
# This is a no-op on JAX versions that do support the argument.
import jax as _jax
_jax_jit_orig = _jax.jit
def _jax_jit_patched(fn, **kwargs):
    kwargs.pop('abstracted_axes', None)
    return _jax_jit_orig(fn, **kwargs)
if 'abstracted_axes' not in inspect.signature(_jax.jit).parameters:
    _jax.jit = _jax_jit_patched

import jax

# ── Distributed init ──────────────────────────────────────────────────────────
# TPU pod: jax.distributed.initialize() reads CLOUD_TPU_TASK_ID / TPU_NAME
# automatically and configures all hosts.
_is_tpu        = bool(os.environ.get("TPU_NAME") or os.environ.get("CLOUD_TPU_TASK_ID"))
_num_processes = int(os.environ.get("JAX_NUM_PROCESSES", "1"))
_coordinator   = os.environ.get("JAX_COORDINATOR_ADDRESS", "")
_process_id    = int(os.environ.get("JAX_PROCESS_ID", "0"))

if _is_tpu:
    jax.distributed.initialize()
elif _num_processes > 1 and _coordinator:
    jax.distributed.initialize(
        coordinator_address=_coordinator,
        num_processes=_num_processes,
        process_id=_process_id,
    )

# ── Now safe to use JAX ───────────────────────────────────────────────────────
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from flax import nnx
import optax
from einops import rearrange, repeat, reduce
from jaxtyping import Float, Array

# Resolve claude_changes/ (model) and claude_distributed/ (root) on sys.path
_HERE   = os.path.dirname(os.path.abspath(__file__))
_ROOT   = os.path.dirname(_HERE)                    # claude_distributed/
_CLAUDE = os.path.join(_ROOT, "claude_changes")     # claude_changes/
for _p in (_ROOT, _CLAUDE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from claude_rl_model import VideoVAE

# ── Version-safe optimizer.update ────────────────────────────────────────────
# flax 0.10.x: optimizer.update(grads)
# flax 0.11.x+: optimizer.update(model, grads)
_update_params = list(inspect.signature(nnx.Optimizer.update).parameters)
if 'model' in _update_params or len([p for p in _update_params if p not in ('self', 'kwargs')]) >= 2:
    def _opt_update(opt, model, grads): opt.update(model, grads)
else:
    def _opt_update(opt, model, grads): opt.update(grads)

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--small", action="store_true",
                    help="Use a tiny model (faster, lower memory)")
args = parser.parse_args()

rank  = jax.process_index()
nproc = jax.process_count()
ndev  = jax.device_count()
nldev = jax.local_device_count()

if rank == 0:
    print(f"JAX: {nproc} process(es), {ndev} total device(s), "
          f"{nldev} local device(s) per process")
    print(f"Local devices: {jax.local_devices()}")

# ── Hyperparameters ───────────────────────────────────────────────────────────
if args.small:
    HEIGHT, WIDTH           = 32, 32
    PATCH_SIZE              = 8
    ENCODER_DEPTH           = 1
    DECODER_DEPTH           = 1
    MLP_DIM                 = 32
    NUM_HEADS               = 2
    QKV_FEATURES            = 16
    MAX_TEMPORAL_LEN        = 4
    SPATIAL_COMPRESSION     = 2
    UPSAMPLE_RATE           = 2
    BATCH_SIZE_PER_PROCESS  = 2   # samples per host process; global = × nproc
    TEMPORAL_LEN            = 4
    LEARNING_RATE           = 1e-4
    NUM_STEPS               = 5
else:
    HEIGHT, WIDTH           = 64, 64
    PATCH_SIZE              = 8
    ENCODER_DEPTH           = 2
    DECODER_DEPTH           = 2
    MLP_DIM                 = 128
    NUM_HEADS               = 4
    QKV_FEATURES            = 64
    MAX_TEMPORAL_LEN        = 8
    SPATIAL_COMPRESSION     = 2
    UPSAMPLE_RATE           = 2
    BATCH_SIZE_PER_PROCESS  = 2
    TEMPORAL_LEN            = 8
    LEARNING_RATE           = 1e-4
    NUM_STEPS               = 5

# On a multi-host TPU pod each process places its data on local_devices()[0].
# XLA all-reduces gradients across hosts automatically, so every chip
# contributes gradient signal even though only one local device is active
# per host.  This matches claude_rl_nonadversarial.py exactly.
LOCAL_BATCH  = BATCH_SIZE_PER_PROCESS           # samples this process handles
GLOBAL_BATCH = BATCH_SIZE_PER_PROCESS * nproc   # effective global batch

# Loss weights (mirrors claude_rl_nonadversarial.py)
GAMMA1               = 0.2
GAMMA2               = 0.001
GAMMA4               = 0.05
RL_LOSS_WEIGHT       = 0.01
MAX_COMPRESSION_RATE = 2
MAGNIFY_NEG_RATE     = 100

# ── Mesh ──────────────────────────────────────────────────────────────────────
mesh               = jax.make_mesh((ndev,), ('batch',))
replicate_sharding = NamedSharding(mesh, P())   # model params fully replicated
if rank == 0:
    print(f"Mesh: {mesh}")
    print(f"Batch: {BATCH_SIZE_PER_PROCESS} per process × {nproc} process(es) "
          f"= {GLOBAL_BATCH} global")

# ── Model ─────────────────────────────────────────────────────────────────────
model = VideoVAE(
    height=HEIGHT, width=WIDTH, channels=3, patch_size=PATCH_SIZE,
    encoder_depth=ENCODER_DEPTH, decoder_depth=DECODER_DEPTH,
    mlp_dim=MLP_DIM, num_heads=NUM_HEADS, qkv_features=QKV_FEATURES,
    max_temporal_len=MAX_TEMPORAL_LEN,
    spatial_compression_rate=SPATIAL_COMPRESSION,
    unembedding_upsample_rate=UPSAMPLE_RATE,
    rngs=nnx.Rngs(42),
    dtype=jnp.float32, param_dtype=jnp.float32,
)

num_params = sum(x.size for x in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param)))
if rank == 0:
    print(f"Parameters: {num_params:,}")

# Replicate model params across all devices
graphdef, state = nnx.split(model)
state = jax.device_put(state, replicate_sharding)
model = nnx.merge(graphdef, state)

# ── Optimizer ─────────────────────────────────────────────────────────────────
schedule_fn = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=LEARNING_RATE,
    warmup_steps=100,
    decay_steps=10_000,
    end_value=LEARNING_RATE / 10,
)
optimizer_def = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=schedule_fn),
)
optimizer = nnx.Optimizer(model, optimizer_def, wrt=nnx.Param)

# ── Loss (no VGG — pure random-tensor test) ───────────────────────────────────
def per_sample_mean(x: Float[Array, "b ..."]):
    return jnp.mean(x, axis=tuple(range(1, x.ndim)))


def magnify_negatives(x, rate: float):
    return jnp.where(x < 0, x * rate, x)


def loss_fn(model: nnx.Module,
            video:         Float[Array, "b time h w c"],
            mask:          Float[Array, "b 1 1 time"],
            original_mask: Float[Array, "b time"],
            rngs: nnx.Rngs,
            train: bool = True):
    reconstruction, compressed, selection, selection_mask, logvar, mean = \
        model(video, mask, rngs, train=train)

    output_mask       = repeat(original_mask, "b time -> (b 2) time")
    sequence_lengths  = reduce(output_mask, "b time -> b 1", "sum")
    sequence_lengths  = jnp.clip(sequence_lengths, 1.0, None)

    video_shaped_mask = rearrange(output_mask, "b time -> b time 1 1 1")
    video_doubled     = repeat(video, "b ... -> (b 2) ...")
    seq_len_bthwc     = rearrange(sequence_lengths, "b 1 -> b 1 1 1 1")

    # MSE
    masked_sq = jnp.square((video_doubled - reconstruction) * video_shaped_mask)
    frame_mse = reduce(masked_sq, "b time h w c -> b 1 h w c", "sum") / seq_len_bthwc
    per_mse   = per_sample_mean(frame_mse)

    # MAE
    masked_abs = jnp.abs((video_doubled - reconstruction) * video_shaped_mask)
    frame_mae  = reduce(masked_abs, "b time h w c -> b 1 h w c", "sum") / seq_len_bthwc
    per_mae    = per_sample_mean(frame_mae)

    # KL
    kl_mask    = rearrange(output_mask, "b time -> b time 1 1")
    seq_len_kl = rearrange(sequence_lengths, "b 1 -> b 1 1 1")
    kl_loss    = 0.5 * (jnp.exp(logvar) - 1 - logvar + jnp.square(mean)) \
                 * kl_mask / seq_len_kl
    per_kl     = per_sample_mean(kl_loss)

    # Selection sparsity
    sel_sum      = reduce(selection_mask * kl_mask, "b time 1 1 -> b 1", "sum")
    kept_density = sel_sum / sequence_lengths
    density_diff = kept_density - (1.0 / MAX_COMPRESSION_RATE)
    sel_loss     = per_sample_mean(
        jnp.square(magnify_negatives(density_diff, MAGNIFY_NEG_RATE)))

    per_sample_loss = per_mse + GAMMA1 * sel_loss + GAMMA2 * per_kl + GAMMA4 * per_mae

    # RL loss
    pairs         = rearrange(per_sample_loss, "(b p) -> b p", p=2)
    means_bp      = rearrange(per_sample_mean(pairs), "b -> b 1")
    stds_bp       = rearrange(jnp.std(pairs, axis=1) + 1e-6, "b -> b 1")
    disadvantages = (pairs - means_bp) / stds_bp
    actions       = rearrange(selection_mask, "(b p) time 1 1 -> b p time", p=2)
    selection2    = rearrange(selection,      "(b p) time 1 1 -> b p time", p=2)
    raw_probs     = jnp.clip(jnp.abs(selection2 + actions - 1), 1e-6, 1.0 - 1e-6)
    probs         = raw_probs / jax.lax.stop_gradient(raw_probs)
    rl_mask_bp    = rearrange(output_mask, "(b p) time -> b p time", p=2)
    probs         = jnp.where(rl_mask_bp, probs, 1.0)
    raw_probs_m   = jnp.where(rl_mask_bp, raw_probs, 1.0)
    raw_traj      = reduce(raw_probs_m, "b p time -> b p 1", "prod")
    probs         = reduce(probs, "b p time -> b p 1", "prod")
    disadvantages = rearrange(disadvantages, "b p -> b p 1")
    rl_loss       = probs * jax.lax.stop_gradient(disadvantages)

    loss = jnp.mean(per_sample_loss) + jnp.mean(rl_loss) * RL_LOSS_WEIGHT

    return loss, {
        "MSE":            jnp.mean(per_mse),
        "MAE":            jnp.mean(per_mae),
        "kl_loss":        jnp.mean(per_kl),
        "selection_loss": jnp.mean(sel_loss),
        "kept_density":   kept_density.mean(),
        "rl_loss":        jnp.mean(rl_loss),
        "mean_traj_prob": jnp.mean(raw_traj),
    }


def train_step(model, optimizer, video, mask, rngs):
    original_mask = mask
    mask_4d = rearrange(mask, "b time -> b 1 1 time")
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True, argnums=nnx.DiffState(0, nnx.Param))
    (loss, aux), grads = grad_fn(model, video, mask_4d, original_mask, rngs)
    _opt_update(optimizer, model, grads)
    return loss, aux


jit_train_step = nnx.jit(train_step)

# ── Training loop with random tensors ─────────────────────────────────────────
rngs = nnx.Rngs(3)
key  = jax.random.key(0)

if rank == 0:
    print(f"\nRunning {NUM_STEPS} steps "
          f"(local batch {LOCAL_BATCH}, global batch {GLOBAL_BATCH}, "
          f"video shape per process: {(LOCAL_BATCH, TEMPORAL_LEN, HEIGHT, WIDTH, 3)})...\n")

# Device each process places its local data on.
# Mirrors claude_rl_nonadversarial.py: one device per host contributes to
# the forward/backward pass; XLA all-reduces gradients across all hosts.
_local_device = jax.local_devices()[0]

for step in range(NUM_STEPS):
    key, data_key = jax.random.split(key)

    # Each process generates its own random batch and places it on its
    # local device.  Gradient all-reduce across hosts makes every chip
    # contribute signal even in a multi-host pod configuration.
    video = jax.random.normal(
        data_key, (LOCAL_BATCH, TEMPORAL_LEN, HEIGHT, WIDTH, 3), dtype=jnp.float32)
    mask  = jnp.ones((LOCAL_BATCH, TEMPORAL_LEN), dtype=jnp.bool_)

    video = jax.device_put(video, _local_device)
    mask  = jax.device_put(mask,  _local_device)

    t0 = time.perf_counter()
    loss, aux = jit_train_step(model, optimizer, video, mask, rngs)
    loss.block_until_ready()
    elapsed = time.perf_counter() - t0

    if rank == 0:
        print(
            f"Step {step}: loss={float(loss):.4f}  "
            f"MSE={float(aux['MSE']):.4f}  "
            f"MAE={float(aux['MAE']):.4f}  "
            f"kl={float(aux['kl_loss']):.4f}  "
            f"sel={float(aux['selection_loss']):.4f}  "
            f"density={float(aux['kept_density']):.4f}  "
            f"rl={float(aux['rl_loss']):.4f}  "
            f"t={elapsed:.3f}s"
        )

if rank == 0:
    print("\nPASSED")
