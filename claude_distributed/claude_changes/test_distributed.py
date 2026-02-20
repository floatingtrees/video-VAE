"""
Quick smoke-test for the distributed infrastructure.

Single device:
    python3 test_distributed.py

Local multi-process (2 simulated processes):
    LOCAL_PROCS=2 JAX_NUM_CPU_DEVICES=2 bash claude_distributed_run.sh --_test_only
    -- or run directly --
    JAX_COORDINATOR_ADDRESS=localhost:8476 JAX_NUM_PROCESSES=2 JAX_PROCESS_ID=0 python3 test_distributed.py &
    JAX_COORDINATOR_ADDRESS=localhost:8476 JAX_NUM_PROCESSES=2 JAX_PROCESS_ID=1 python3 test_distributed.py &
    wait
"""

import os
import sys

import jax

# ── Distributed init (mirrors claude_rl_nonadversarial.py) ───────────────────
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
import numpy as np
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

# Make sure dataloader is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataloader import prepare_batch_for_mesh

rank  = jax.process_index()
nproc = jax.process_count()

print(f"[proc {rank}/{nproc}] local_devices={jax.local_devices()}  "
      f"global_device_count={jax.device_count()}")

# ── Mesh ──────────────────────────────────────────────────────────────────────
mesh = jax.make_mesh((jax.device_count(),), ('batch',))
replicate_sharding = NamedSharding(mesh, P())
print(f"[proc {rank}/{nproc}] mesh={mesh}")

# ── Data sharding ─────────────────────────────────────────────────────────────
# Each process contributes a local batch; prepare_batch_for_mesh assembles the
# global array.  Shape check: global batch = local_batch * nproc.
local_batch_size = 2
local_batch = {
    "video": np.ones((local_batch_size, 4, 8, 8, 3), dtype=np.float32) * (rank + 1),
    "mask":  np.ones((local_batch_size, 4),           dtype=np.float32),
}

global_batch = prepare_batch_for_mesh(local_batch, mesh)
expected_global_batch = local_batch_size * nproc

assert global_batch["video"].shape == (expected_global_batch, 4, 8, 8, 3), \
    f"Unexpected video shape: {global_batch['video'].shape}"
assert global_batch["mask"].shape == (expected_global_batch, 4), \
    f"Unexpected mask shape: {global_batch['mask'].shape}"

# ── Replicated compute ────────────────────────────────────────────────────────
# All processes must execute the same ops in the same order.
result = jnp.sum(global_batch["video"])
result.block_until_ready()

if rank == 0:
    print(f"[proc 0] global video shape : {global_batch['video'].shape}")
    print(f"[proc 0] global mask  shape : {global_batch['mask'].shape}")
    print(f"[proc 0] sum(video)         : {float(result):.1f}")
    print("PASSED")
