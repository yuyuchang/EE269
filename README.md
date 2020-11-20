# EE269
The final project of EE269 Signal Processing for Machine Learning at Stanford

### datasets
We can download the datasets from \
https://github.com/laiguokun/multivariate-time-series-data

### usage

1. Download the datasets from the link.

2. Create a directory "mkdir data" and move the downloaded datasets into "data/"

3. "cd src/electricity/"

4. Make a directory "model" under "src/electricity/"

5. python3 memdnn_split_attn.py --data data/electricity.txt --CNN_kernel 6 --CNN_unit 100 --GRU_unit 100 --input_length 24 --horizon 3
