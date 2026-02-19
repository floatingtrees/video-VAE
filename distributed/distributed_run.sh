export JAX_NUM_CPU_DEVICES=2
num_processes=4

range=$(seq 0 $(($num_processes - 1)))

for i in $range; do
  python toy.py $i $num_processes > /tmp/toy_$i.out &
done

wait

for i in $range; do
  echo "=================== process $i output ==================="
  cat /tmp/toy_$i.out
  echo
done