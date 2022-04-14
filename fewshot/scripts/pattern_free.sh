# Please adapt the following parameters:
# 1) We set the "max_seq_length" for different datasets as follows:   
# For ("rte", "cb", "wic", "cr", "subj"), we set it to 256 and we set to 
# 128 for the rest ("mrpc", "qqp",  "qnli", "SST-2", "sst-5",  "mr",  "trec")

# We report the average/min/std performance across all data_seed and seeds:  
# 2) For data_seed we sweep over [100, 13, 42, 87, 21]
# 3) For seed we sweep over [1, 10, 100, 1000]     

python run_clm.py configs/pattern_free.json 
