import json
import pickle
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
import pandas as pd

# Define CacheEntry structure (consistent with demo_cache.py)
@dataclass
class CacheEntry:
    uid: str
    question: str
    answer: str
    reward: float
    pass_rate: float
    timestamp: float
    metadata: dict

# Read data
local_dir = "~/data/math"
train_path = os.path.join(local_dir, "train.parquet")
output_pkl_path = "./demo_cache/math_demo.pkl"
dataset = pd.read_parquet(train_path)

# Convert to CacheEntry list
entries = {}
for idx, row in dataset.iterrows():
    if len(row["target"]) < len("Answer: -27 . Instructions. Exact answer: $4 \sqrt{5}-36$.") or \
        "https:" in row["target"]:
        continue
    entry = CacheEntry(
        uid=row["extra_info"]["index"],         # use unique_id as uid
        question=row["prompt"][0]["content"],   # read question from prompt field
        answer=row["target"],                   # use solution as answer
        reward=1.0,
        pass_rate=1.0,
        timestamp=datetime.now().timestamp(),
        metadata={
            "level": -1,
            "subject": row["data_source"],
            "final_answer": row["reward_model"]["ground_truth"]
        }
    )
    entries[entry.uid] = asdict(entry)
    print(entry.uid,"\n", entry.question, "\n", entry.answer, "\n", entry.metadata["final_answer"], "\n")

# Build cache data structure
cache_data = {
    "entries": entries,
    "stats": {
        "total_added": len(entries),
        "total_retrieved": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "evictions": 0
    },
    "max_size": 1000,
    "min_pass_rate": 1.0
}

# Save as .pkl file
os.makedirs(os.path.dirname(output_pkl_path), exist_ok=True)
with open(output_pkl_path, "wb") as f:
    pickle.dump(cache_data, f)
    print("Example:", list(entries.values())[0] if len(entries) > 0 else "No data")
    print("Number of entries:", len(list(entries.values())))

print(f"PKL file generated: {output_pkl_path}")
