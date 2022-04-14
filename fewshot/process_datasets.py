"""Converts data from Gao et al. ACL 2021 in a unified jason format."""
import os 
import sys 
from os.path import join 
from typing import Optional
import pandas as pd 
from dataclasses import dataclass, field

from transformers import HfArgumentParser
from fewshot.utils import utils 

DATASETS = ["SST-2", "cr",  "mr", "sst-5", "subj", "trec"]
SUFFIX = {data: "tsv" if data == "SST-2" else "csv" for data in DATASETS}
HEADERS = {data: 0 if data == "SST-2" else None for data in DATASETS} 
HEADER_NAMES= {data: ['source', 'label'] if data == "SST-2" else ['label', 'source'] for data in DATASETS} 

def save_dataset(df, args, task, split):
    out_dir = join(args.out_dir, task)
    utils.create_dir(out_dir)
    out = df.to_json(orient='records')[1:-1].replace('},{', '}\n{')
    with open(join(out_dir, f"{split}.json"), 'w') as f:
        f.write(out)

def convert_datasets(data_dir, task):
    for split in ["train", "test"]:
        dirname = os.path.join(data_dir, task)
        if task == "SST-2":
            split_data = "dev" if split == "test" else "train" 
            filename = os.path.join(dirname, f"{split_data}.tsv")
            df = pd.read_csv(filename, sep='\t', header=HEADERS[task], names=HEADER_NAMES[task], index_col=False)
        else:
            filename = os.path.join(dirname, f"{split}.csv")
            df = pd.read_csv(filename, sep=",", header=HEADERS[task], names=HEADER_NAMES[task], index_col=False)
            if split == "train" and task == "cr":
                # the last element is empty and we remove it.
                df = df[:-1]
        df['label'] = df['label'].astype(str)
        save_dataset(df, args, task, split)

@dataclass
class Arguments:
    """
    Arguments required for processing the data.
    """
    data_dir: Optional[str] = field(
        default="datasets",
        metadata={"help": "Original data directory containing the data from Gao et al, ACL, 2021."}
    )
    out_dir: Optional[str] = field(
        default="datasets_processed", 
        metadata={"help": "path to save the processed data."}       
    )

def process_data(args):
    for task in DATASETS:
        print(task)
        convert_datasets(args.data_dir, task)

if __name__ == "__main__":
    parser = HfArgumentParser((Arguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        args, = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        args, = parser.parse_args_into_dataclasses()
    process_data(args)




