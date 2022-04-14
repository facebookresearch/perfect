# This scripts runs the pet baseline in table 1.
# Please adapt the following parameters:
# 1) We set the "max_seq_length" for different datasets as follows:   
# For ("rte", "cb", "wic", "cr", "subj"), we set it to 256 and we set to 
# 128 for the rest ("mrpc", "qqp",  "qnli", "SST-2", "sst-5",  "mr",  "trec")
# 2) for each dataset, we run it for different choice of patterns/verbalizers
# for this, please sweep `pattern_id` in the config file over the following values:
#        "boolq": [0, 1, 2, 3, 4, 5], 
#        "rte": [0, 1, 2, 3, 4],  
#        "cb": [0, 1, 2, 3, 4],
#        "copa": [0, 1], 
#        "multirc": [0, 1, 2, 3],  
#        "wic": [0, 1, 2],
#        "mrpc": [0, 1, 2, 3, 4, 5], 
#        "qqp": [0, 1, 2, 3, 4, 5], 
#        "qnli": [0, 1, 2, 3, 4, 5],
#        "SST-2": [0, 1, 2, 3],  
#        "sst-5": [0, 1, 2, 3],  
#        "mr": [0, 1, 2, 3],  
#        "cr": [0, 1, 2, 3],  
#        "trec": [0, 1, 2, 3],  
#        "subj": [0, 1, 2, 3]

# We report the average/min/std performance across all data_seed and seeds:  
# 1) For data_seed we sweep over [100, 13, 42, 87, 21]
# 2) For seed we sweep over [1, 10, 100, 1000]      

# For PET-Average we report the average test performance across all pattern_id choices 
# and for PET-Best we report the test performance for the pattern_id, for which it obtains
# the best validation performance.
python run_clm.py configs/pet.json 