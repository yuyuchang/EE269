for kernel in 2 3 6
do
  for cnn_unit in 32 50 100 200
  do
    for gru_unit in 32 50 100 200
    do
      #python3.6 memdnn_split_attn.py \
      python3.6 memdnn.py \
      --CNN_kernel $kernel \
      --CNN_unit $cnn_unit \
      --GRU_unit $gru_unit \
      --input_length 24 \
      --horizon 12
    done
  done
done
