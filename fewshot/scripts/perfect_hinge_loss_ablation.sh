# This scripts trains the ablation of -Hingeloss in table 9.

# Please adapt the following parameters:
# 1) We set the "max_seq_length" for different datasets as follows:   
# For ("rte", "cb", "wic", "cr", "subj"), we set it to 256 and we set to 
# 128 for the rest ("mrpc", "qqp",  "qnli", "SST-2", "sst-5",  "mr",  "trec")
# 2) For "soft_mask_labels_learning_rate" we sweep it over [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
# and choose the one obtaining the maximum validation performance. 

# We report the average/min/std performance across all data_seed and seeds:  
# 1) For data_seed we sweep over [100, 13, 42, 87, 21]
# 2) For seed we sweep over [1, 10, 100, 1000]      

python run_clm.py configs/perfect_hinge_loss_ablation.json 
