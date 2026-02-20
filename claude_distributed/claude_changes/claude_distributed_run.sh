#!/bin/bash
# claude_distributed_run.sh
#
# Launches claude_rl_nonadversarial.py in one of three modes:
#
#   1. Single device (default):
#        ./claude_distributed_run.sh [training-args]
#
#   2. Local multi-process / multi-GPU on one machine (for testing):
#        LOCAL_PROCS=4 JAX_NUM_CPU_DEVICES=4 ./claude_distributed_run.sh [training-args]
#      Each process gets a unique JAX_PROCESS_ID and shares localhost as coordinator.
#
#   3. TPU cluster — run this script on EVERY TPU host, or set TPU_IPS to have
#      this script SSH-launch on all hosts automatically:
#        TPU_IPS="10.0.0.1 10.0.0.2 10.0.0.3 10.0.0.4" ./claude_distributed_run.sh [training-args]
#      jax.distributed.initialize() inside Python is called with no args; the
#      TPU runtime supplies the coordinator address automatically (see JAX docs:
#      https://docs.jax.dev/en/latest/multi_process.html).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

#source "$PARENT_DIR/../venv/bin/activate"

# Both the parent (rl_model, dataloader, …) and claude_changes directories
# must be on PYTHONPATH so imports resolve correctly.
export PYTHONPATH="$PARENT_DIR:$SCRIPT_DIR:${PYTHONPATH:-}"

TRAINING_SCRIPT="$SCRIPT_DIR/claude_rl_nonadversarial.py"

# ── Mode detection ────────────────────────────────────────────────────────────

if [ -n "${TPU_IPS:-}" ]; then
    # ── Mode 3: TPU cluster — SSH to every host in parallel ──────────────────
    # The Python script detects TPU_NAME / CLOUD_TPU_TASK_ID (set by TPU
    # runtime on each VM) and calls jax.distributed.initialize() with no args.
    IFS=' ' read -ra HOSTS <<< "$TPU_IPS"
    NUM_HOSTS="${#HOSTS[@]}"
    echo "TPU cluster mode: launching on $NUM_HOSTS host(s)"
    echo "Hosts: ${HOSTS[*]}"

    for ip in "${HOSTS[@]}"; do
        ssh -n -o StrictHostKeyChecking=no -o BatchMode=yes "$ip" \
            "PYTHONPATH=$PARENT_DIR:$SCRIPT_DIR \
             python3 $TRAINING_SCRIPT $*" &
    done

    wait
    echo "All TPU hosts completed."

elif [ -n "${LOCAL_PROCS:-}" ]; then
    # ── Mode 2: local multi-process (CPU simulation or single-node multi-GPU) ─
    # Spawns LOCAL_PROCS Python processes on localhost.
    # Process 0 acts as the JAX coordinator.
    N="${LOCAL_PROCS}"
    PORT="${LOCAL_COORDINATOR_PORT:-8476}"

    echo "Local multi-process mode: $N process(es) on localhost:$PORT"

    for i in $(seq 0 $((N - 1))); do
        JAX_COORDINATOR_ADDRESS="localhost:$PORT" \
        JAX_NUM_PROCESSES="$N" \
        JAX_PROCESS_ID="$i" \
        python3 "$TRAINING_SCRIPT" "$@" \
            > "/tmp/claude_train_proc${i}.out" 2>&1 &
    done

    wait

    # Print each process's output after all finish
    for i in $(seq 0 $((N - 1))); do
        echo "=================== process $i output ==================="
        cat "/tmp/claude_train_proc${i}.out"
        echo
    done

else
    # ── Mode 1: single device ─────────────────────────────────────────────────
    echo "Single-device mode"
    python3 "$TRAINING_SCRIPT" "$@"
fi
