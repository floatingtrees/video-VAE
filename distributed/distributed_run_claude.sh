#!/bin/bash
source /Users/jonathanzhou/code/video-VAE/venv/bin/activate
export JAX_NUM_CPU_DEVICES=2
num_processes=4

range=$(seq 0 $(($num_processes - 1)))

for i in $range; do
  python3 distributed_rl_model.py $i $num_processes > /tmp/distributed_rl_$i.out 2>&1 &
done

wait

for i in $range; do
  echo "=================== process $i output ==================="
  cat /tmp/distributed_rl_$i.out
  echo
done
