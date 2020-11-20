for kernel in 6 3 2
do
  for cnn_unit in 32 50 100 200
  do
    for gru_unit in 32 50 100 200
    do
      python3 main.py \
      --data ../../MEMNN-RNN/data/traffic/traffic.txt \
      --CNN_kernel $kernel \
      --CNN_unit $cnn_unit \
      --GRU_unit $gru_unit \
      --input_length 24 \
      --horizon 3 \
      --output tra_AR_3_result.csv
    done
  done
done
