"""
Preprocess the OpenR1-Math-46k-8192 dataset to parquet format
"""

import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/openr1_math_220k-8192")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source = "Elliott/Openr1-Math-46k-8192"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source, "default")

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
    
    def make_map_fn(split):
        def process_fn(example, idx):
            instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

            data_source_ = example.pop("data_source")
            prompt = example.pop("prompt")
            prompt = prompt[1]["content"] + " " + instruction_following
            target = example.pop("target")
            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": prompt}],
                "ability": "math",
                "reward_model": example.pop("reward_model"),
                "target": target,
                "extra_info": example.pop("extra_info"),
            }
            return data
        return process_fn
    
    dataset = dataset.map(function=make_map_fn("train"), with_indices=True)
    dataset = dataset["train"]
        
    save_path = os.path.join(local_dir, "train_instruct_w_expert_answer.parquet")
    dataset.to_parquet(save_path)

    print("Dataset preview:")
    print(dataset[0] if len(dataset) > 0 else "Empty dataset")