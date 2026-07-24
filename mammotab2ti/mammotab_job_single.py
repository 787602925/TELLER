# nohup python -u -m mammotab2ti.mammotab_job_single > mammotab_job_single.log 2>&1 &

import json
import random
from tqdm import tqdm

input_file = "/DATA1/khli/mammotab/modified_mammotab/mammotab_jobs.jsonl"
output_file = "/DATA1/khli/mammotab/modified_mammotab/mammotab_jobs_single.jsonl"

TOTAL_LINES = 29_107_433

random.seed(42)

# source_file -> [count, sampled_record]
reservoir = {}

with open(input_file, "r", encoding="utf-8") as f:
    for line in tqdm(f, total=TOTAL_LINES, desc="Sampling"):
        record = json.loads(line)
        sf = record["source_file"]

        if sf not in reservoir:
            reservoir[sf] = [1, record]
        else:
            reservoir[sf][0] += 1
            n = reservoir[sf][0]

            if random.randint(1, n) == 1:
                reservoir[sf][1] = record

with open(output_file, "w", encoding="utf-8") as f:
    for _, sample in tqdm(
        reservoir.values(),
        total=len(reservoir),
        desc="Writing"
    ):
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

print(f"Tables: {len(reservoir):,}")