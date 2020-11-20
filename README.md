# EE269
The final project of EE269 Signal Processing for Machine Learning at Stanford

### usage

1. "cd src/electricity/"

2. Make directory "model" under "src/electricity/"

3. python3 memdnn_split_attn.py --data data/electricity.txt --CNN_kernel 6 --CNN_unit 100 --GRU_unit 100 --input_length 24 --horizon 3
