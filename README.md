# **TELLER: Dual-Path Iterative Preference Optimization for Table Entity Linking**

> 🇨🇳 中文文档 / Chinese version: [README.zh-CN.md](README.zh-CN.md)

This repository converts [MammoTab](https://unimib-datai.github.io/mammotab-docs/) table cell-entity-annotation (CEA) data into [TableInstruct](https://huggingface.co/datasets/osunlp/TableInstruct)-style instruction-tuning samples, then trains `meta-llama/Llama-3.1-8B` through SFT, CoT-SFT and DPO(RL), and evaluates the resulting checkpoints on both the MammoTab validation set and the TableInstruct entity-linking test set.

All the shell commands referenced below live as ready-to-run scripts in `[scripts/](scripts/)` at the repo root, numbered in execution order (`00_` … `45_`). This document only explains what each step does and why; for the actual command, open the corresponding `.sh` file (you can run it directly with `bash scripts/xxx.sh` — just edit the placeholder paths such as `~/DATA/...` / `/DATA1/khli/...` to match your own environment first).

## Table of contents

1. [Dataset downloads](#1-dataset-downloads)
2. [MammoTab → TableInstruct format conversion](#2-mammotab--tableinstruct-format-conversion-mammotab2ti)
3. [Environment setup](#3-environment-setup)
4. [Training pipeline overview](#4-training-pipeline-overview)
5. [Route A: SFT → DPO-1 → DPO-2](#5-route-a-sft--dpo-1--dpo-2)
6. [Route B: SFT → CoT-SFT → DPO-1 → DPO-2](#6-route-b-sft--cot-sft--dpo-1--dpo-2)
7. [Final comparison on the TableInstruct 6-dataset test set](#7-final-comparison-on-the-tableinstruct-6-dataset-test-set)
8. [Results summary](#8-results-summary)
9. [Script reference](#9-script-reference)

---



## 1. Dataset downloads



### 1.1 MammoTab (tables + Wikidata annotations, source of training data)

- Project page / docs: [https://unimib-datai.github.io/mammotab-docs/](https://unimib-datai.github.io/mammotab-docs/)
- Code repo (pipeline for generating/rebuilding the dataset): [https://github.com/unimib-datAI/mammotab](https://github.com/unimib-datAI/mammotab)
- Data download (Zenodo, `mammotab_full.zip`, ~14.6GB, 838,930 tables extracted from English Wikipedia + CEA/CTA/CPA annotations + JSON metadata):
[https://zenodo.org/records/16562700](https://zenodo.org/records/16562700)

After downloading and extracting, you need a directory of "one JSON file per table" (each file containing the table's `text` matrix, `caption`, page title, and the gold QID for each target cell). By default this repo expects it at:

```text
~/DATA/mammotab/mammotab_dataset_semtab/json/
```

> If what you download from Zenodo is the raw CSV (table + CEA annotations) + JSON metadata combination instead, you'll need to convert each table into the single-JSON-per-table format above first — either by following the official MammoTab pipeline (`unimib-datAI/mammotab` repo) or writing your own conversion script. At minimum each JSON needs the `text`/table matrix, `caption`, page/section title, and the gold QID/wiki title for each target cell. `mammotab2ti/stage1_extract_jobs.py` (step 2 below) scans this directory directly.



### 1.2 Wikidata truthy dump (offline candidate-retrieval index)

`mammotab2ti/stage0_build_wikidata_index.py` needs a Wikidata dump to build a local candidate-entity sqlite index. We recommend the much smaller **truthy** variant (only best-rank direct statements, no qualifiers/references):

```bash
bash scripts/01_download_wikidata_dump.sh
```

- Official index page: [https://dumps.wikimedia.org/wikidatawiki/entities/](https://dumps.wikimedia.org/wikidatawiki/entities/)
- Documentation: [https://www.wikidata.org/wiki/Wikidata:Database_download](https://www.wikidata.org/wiki/Wikidata:Database_download)
- The file is roughly 40GB (`.nt.bz2`); downloading + indexing takes a while, so run it with `nohup` in the background (see step 2 below).
- `.json`/`.json.gz`/`.json.bz2` dumps are also supported — `stage0_build_wikidata_index.py` auto-detects the format from the file extension.



### 1.3 TableInstruct (cross-task instruction-tuning dataset + out-of-domain test set)

- Hugging Face dataset page: [https://huggingface.co/datasets/osunlp/TableInstruct](https://huggingface.co/datasets/osunlp/TableInstruct)
- Paper / code: [https://github.com/OSU-NLP-Group/TableLlama](https://github.com/OSU-NLP-Group/TableLlama)

```python
from datasets import load_dataset
ds = load_dataset("osunlp/TableInstruct")
```

This project only uses the entity-linking test set from `/eval_data` for **cross-dataset generalization evaluation** — it is never used for training (training only ever uses MammoTab). The file referenced by the eval scripts below has already been processed with the same "same row/column + full header + capped at 10 rows" simplification rule as MammoTab:

```text
/DATA1/khli/tablellama/ent_link_test_simplified.json
```

---



## 2. MammoTab → TableInstruct format conversion (`mammotab2ti/`)

The conversion pipeline has 5 stages (stage 0 only needs to run once and is reused; stages 1-4 are the main data-processing chain), mapped to `scripts/02_*.sh` … `scripts/06_*.sh`:

```text
stage0_build_wikidata_index.py          Wikidata dump -> local sqlite candidate index        scripts/02_stage0_build_wikidata_index.sh
stage1_extract_jobs.py                  838k table JSONs -> per-mention jobs.jsonl            scripts/03_stage1_extract_jobs.sh
stage2_generate_candidates_offline.py   jobs.jsonl + sqlite -> tableInstruct-like samples      scripts/04_stage2_generate_candidates.sh
stage3_merge_single_shards.py           merge shard outputs + data cleaning                    scripts/05_stage3_merge_shards.sh
stage4_simplify.py                      keep same row/col + full header, cap rows at 10        scripts/06_stage4_simplify.sh
```



### Stage 0: build the local Wikidata index

```bash
bash scripts/02_stage0_build_wikidata_index.sh
```

- Input: `--dump`, a Wikidata dump file.
- Output: `--db`, a sqlite file with two tables — `entities` (qid/label/description/p31/aliases/enwiki_title/popularity, ...) and `alias_index` — used by stage 2 to look up candidate entities.
- This only needs to run once; every stage-2 job reuses the same `.sqlite`.



### Stage 1: extract mention-level jobs from MammoTab JSON

```bash
bash scripts/03_stage1_extract_jobs.sh
```

Expands the 838k table JSON files into a `jobs.jsonl` with one record per entity mention to be linked (each record has `mention`/`column_name`/`row_index`/`page_title`/`section_title`/`table_caption`/`gold_qid`/`gold_wiki_title`, etc.).

### Stage 2: local candidate retrieval + assembling tableInstruct-like samples

```bash
bash scripts/04_stage2_generate_candidates.sh
```

- For each mention, retrieves the top-20 candidate entities from the stage-0 local index and assembles a 4-field sample (`instruction`/`input_seg`/`question`/`output`); `output` is the gold answer formatted as a candidate string, and `question` contains the (still unshuffled) candidate list.
- Gold is **not** force-injected into the candidate list. A separate `--audit_output` jsonl is written with each sample's `candidate_qids`/`gold_in_topk`, used by stage 3 for cleaning and for later "was gold in top-20" statistics.
- Given the data volume (838k tables → many more mentions), shard the work with `--num_shards`/`--shard_id` and run one process per shard in parallel (this is a pure CPU/sqlite-IO job — plain `python` processes are enough, no `torchrun` needed; you can duplicate `scripts/04_stage2_generate_candidates.sh` with different `--shard_id` values and submit them concurrently). `--resume` supports resuming from a partial run.



### Stage 3: merge shards + data cleaning

```bash
bash scripts/05_stage3_merge_shards.sh
```

Merges the stage-2 per-shard outputs into one file while doing three things:

1. **Backfill gold**: if the gold answer isn't in the candidate list, append it (skipped when gold is `NIL` or empty);
2. **Shuffle candidates**: reorder each sample's candidate list with a `--seed`-fixed RNG to avoid a "correct answer is always at the same position" bias;
3. **Drop empty candidates**: samples with an empty candidate list are discarded.



### Stage 4: simplify table context (keep same row/col + full header, ≤10 rows)

```bash
bash scripts/06_stage4_simplify.sh
```

Goals:

1. Keep the header + the full target-entity row +, for other rows, only the cell in the "entity column";
2. Keep only a window of rows around the target entity (default max 10 rows);
3. Write the output as a JSON array (`.json`) that the SFT/CoT-SFT/DPO training scripts below read directly.

The key piece is how `_infer_entity_position_from_input_and_question()` locates "which row/column the entity is in":

1. Regex-extract two pieces from `question`: the **mention** from `selected entity mention in the table cell is: ...`, and the **column name** from `The column name for '...' is ...`. If either can't be parsed, inference fails (returns `None`).
2. Split `input_seg` into a `prefix` (everything before the first `row N:`, containing `[TAB] ... col: ...`, used to parse header cells) and a `rows_part` (starting at the first `row N:`).
3. Whitespace-normalize the header and locate the **entity column index** by matching the column name against it; fail if not found.
4. Split `rows_part` on `[SEP]` into individual rows, parse each row's number and cells from `row N: ...`, and check (case-insensitively) whether the cell in the "entity column" for that row contains the mention. The first matching row is the **entity row number**.
5. If nothing matches, inference fails entirely; by default (without `--keep_on_infer_failure`) such samples are **dropped**. Passing that flag instead keeps the original, un-simplified `input_seg`.

The resulting `stage4_single_shards_merge_simplified.json` is the unified data format used throughout this repo for training/evaluation (`instruction`/`input_seg`/`question`/`output`). Every `--data_path`/`--eval_data_path` mentioned below refers to files of this kind (or validation subsets sliced from it, e.g. `..._mam_val_last3750.json`).

---



## 3. Environment setup

```bash
bash scripts/00_setup_env.sh
```

- The training scripts (`el/sft.py` / `el/sft_CoT.py` / `rl/train_dpo.py` / `rl/train_dpo4sft.py`) all use `transformers.HfArgumentParser`-style dataclass arguments, so any `--field_name value` works directly on the command line; use `torchrun --nproc_per_node=<num_gpus>` for multi-GPU runs.
- DPO training uses `trl.DPOTrainer` (pinned to `trl==1.8.0`; other versions may have incompatible fields).
- The CoT-stage RL data generation (`rl/gen_dpo_data.py`) calls a DeepSeek teacher model to fix samples the local model got wrong, so `export DEEPSEEK_API_KEY=...` first. The SFT (non-CoT) RL data generation (`rl/gen_dpo_data4sft.py`) needs no teacher model — it uses the dataset's own gold answer as `accept`.
- Base model: `meta-llama/Llama-3.1-8B` (requires accepting the license on Hugging Face and running `huggingface-cli login`).

---



## 4. Training pipeline overview

![Dual-Path Iterative Preference Optimization: direct-answer error mining feeds DPO, while reasoning error mining feeds L-RPO, both refreshed every iteration.](assets/paper_figures/iterative_lrpo.png)

*Figure 1 from our paper — **Dual-Path Iterative Preference Optimization**. Top: the direct-answer path mines errors from the current policy and trains DPO on the resulting chosen/rejected pairs. Bottom: the reasoning path mines reasoning errors and trains L-RPO (length-normalized DPO + a chosen-response likelihood term). Both paths feed the updated policy back into the next iteration's error mining.*

Both routes share the same SFT starting point (`result_mammo2/checkpoint-21000`) and then fork:

```text
                         el/sft.py (SFT)
                               │
                 result_mammo2/checkpoint-21000
                 ┌─────────────┴─────────────────────────┐
                 │                                       │
     Route A (plain answer, no reasoning)        Route B (learn <think> reasoning first)
                 │                                       │
   rl/gen_dpo_data4sft.py                         el/sft_CoT.py (CoT-SFT, LoRA)
   rl/train_dpo4sft.py (DPO-1)                result_CoT_filtered_compressed_data_lora
   → result_dpo_sft                                       │
                 │                          rl/gen_dpo_data.py (DeepSeek teacher correction)
   same steps again (DPO-2)                  rl/filter_process_rl_data.py (filter/compress)
   → result_dpo_sft_i2                       rl/train_dpo.py (DPO-1) → result_dpo_v2
                                                             │
                                              same steps again (DPO-2) → result_dpo_i2
```

Every checkpoint from both routes is evaluated on two benchmarks:

- **MammoTab validation set** (in-domain, via `el/eval_checkpoint.py` or `el/eval_CoT_checkpoint.py`);
- **TableInstruct entity-linking test set** (out-of-domain generalization check, see section 7).

---



## 5. Route A: SFT → DPO-1 → DPO-2



### 5.1 SFT (`el/sft.py`)

```bash
bash scripts/10_sft_train.sh
```

Evaluate (MammoTab validation set):

```bash
bash scripts/11_eval_sft_mammotab.sh
```

MammoTab validation accuracy: **0.8759** (8532/9741).

### 5.2 Generate DPO training data (`rl/gen_dpo_data4sft.py`)

Runs inference with the SFT checkpoint over the training data; for samples it gets wrong, `accept` = the gold answer (normalized to candidate-entity format) and `reject` = the local model's raw generation.

```bash
bash scripts/20_gen_dpo_data4sft_round1.sh
```

> Defaults to reading `result_mammo2/checkpoint-21000` (pass `--model` to point at a different checkpoint — see `scripts/23_gen_dpo_data4sft_round2.sh` for round 2). `--num` controls sample count, and `--resume` supports multi-GPU checkpoint/resume via per-rank shard files. The resulting `{prompt, accept, reject}` pairs feed directly into `rl/train_dpo4sft.py` — unlike Route B, there's no need to run `filter_process_rl_data.py` here, since `accept` is just a single gold candidate string with no CoT reasoning that needs structural validation/compression.



### 5.3 DPO round 1 (`rl/train_dpo4sft.py`)

```bash
bash scripts/21_train_dpo4sft_round1.sh
```

By default this continues training the `result_mammo2/checkpoint-21000` adapter (override with `--adapter_name_or_path`). `--save_best_by_gen_eval True` periodically runs real generation on `--gen_eval_data_path` to estimate accuracy and keeps the best-scoring checkpoints.

Evaluate (MammoTab validation set):

```bash
bash scripts/22_eval_dpo_sft_round1_mammotab.sh
```

MammoTab validation accuracy: **0.8816** (8588/9741), up from 0.8759 for plain SFT.

### 5.4 DPO round 2 (regenerate data from the round-1 checkpoint)

Re-run inference with `result_dpo_sft/checkpoint-150` to produce a fresh batch of DPO pairs, then train a second round:

```bash
bash scripts/23_gen_dpo_data4sft_round2.sh
bash scripts/24_train_dpo4sft_round2.sh
```

> Note that `--adapter_name_or_path` must point at `result_dpo_sft/checkpoint-150` (continuing the same LoRA), not restart from the original SFT checkpoint.

Evaluate:

```bash
bash scripts/25_eval_dpo_sft_round2_mammotab.sh
```

MammoTab V2 accuracy: **88.20%** — a further (smaller) improvement over round 1's 88.16%, and consistent with the small further gain on the out-of-domain TableInstruct test set (see section 7).

---



## 6. Route B: SFT → CoT-SFT → DPO-1 → DPO-2



### 6.1 CoT-SFT (`el/sft_CoT.py`, continues training a new LoRA on top of the SFT LoRA)

Starting from Route A's SFT checkpoint (`result_mammo2/checkpoint-21000`), train a new LoRA adapter on data with `<think>...</think>` reasoning targets (`--lora_only True` means only the freshly-initialized LoRA is trained; embed/norm and everything else stays frozen):

```bash
bash scripts/30_sft_cot_train.sh
```

> The CoT training data (`merged_CoTSFT_all&all_filtered_cot_compressed.json`) is itself constructed via a teacher model (DeepSeek) plus a post-processing pipeline (structure filter → quality filter → length compression — see the three functions reused by `rl/filter_process_rl_data.py` in section 6.3) that produces SFT samples with `<think>` reasoning chains. Generating that dataset is out of scope for this document; see `data_clean/compress_long_cot.py`, `data_clean/filter_low_info_cot.py`, and `data_aug/augment_ent_link_thinking.py` for the approach.

Evaluate (`el/eval_CoT_checkpoint.py`):

```bash
bash scripts/31_eval_cot_sft_mammotab.sh
```

MammoTab validation accuracy: **0.7822** (704/900, small split) / **0.7909** (7704/9741, full split). CoT-SFT alone scores lower than plain SFT (0.8759) — this is expected, since it has to additionally learn to "reason before answering"; DPO below improves both reasoning quality and answer accuracy together.

### 6.2 Generate CoT-DPO training data (`rl/gen_dpo_data.py`, local model + DeepSeek teacher correction)

```bash
export DEEPSEEK_API_KEY=...
bash scripts/32_gen_dpo_data_cot_round1.sh
```

Flow: batch-inference the training data with the local model → find samples where the prediction is wrong, using their raw generation (including `<think>`) as `reject` → send a prompt with "the correct answer + instructions not to reference/hint at it" to the DeepSeek teacher model, which generates a genuine `<think>...</think>` chain that actually reasons its way to the correct answer, used as `accept` (`--accept_max_reasoning_*` caps the teacher's reasoning length). Multi-GPU runs shard work by round in a round-robin fashion; `--resume` supports resuming.

### 6.3 Filter + compress the DPO data (`rl/filter_process_rl_data.py`)

```bash
bash scripts/33_filter_rl_data_round1.sh
```

A three-step, CPU-only pipeline:

1. **Structure filter**: `accept` must parse into a `<think>...</think>\n<entity>` structure; unparsable samples are dropped entirely, and the parsed final entity must match gold or the sample is also dropped. `reject` has no structural requirement (truncation/repetition loops are themselves informative negative signal); only obvious format breakage is cleaned up.
2. **Quality filter**: scores the reasoning content of `accept` and removes low-information samples — rambling, answer leakage, copying the candidate list verbatim, lack of actual justification, etc.
3. **Length compression**: `accept` samples over `--compress-threshold-tokens` are extractively compressed down to `--target-output-tokens`; samples where compression fails (unparsable / entity changed / still too long / too repetitive) are dropped entirely.

Add `--dry_run` first to inspect the funnel statistics at each step before writing the final file.

### 6.4 DPO round 1 (`rl/train_dpo.py`)

```bash
bash scripts/34_train_dpo_cot_round1.sh
```

A few key hyperparameters (see the script's header comments for the full story):

- `--rpo_alpha 0.5`: in the CoT data, `accept` (the teacher's full reasoning) is systematically much longer than `reject` (the local model's raw generation). Standard DPO's sigmoid pairwise loss sums log-probs over the sequence without length normalization, which creates a spurious "longer = preferred" gradient. `rpo_alpha > 0` adds an `sft` component (cross-entropy NLL on chosen) to `loss_type`, preventing chosen likelihood from being pushed down unintentionally.
- `--length_normalize` (on by default): switches the pairwise loss to a length-normalized variant, further mitigating the length bias.
- `--max_chosen_reject_len_ratio 6.0`: drops samples where the chosen/rejected token-length ratio is abnormally large.
- `--save_best_by_gen_eval True`: keeps the checkpoint with the best periodic real-generation `accuracy + has_think` score.

Evaluate:

```bash
bash scripts/35_eval_dpo_cot_round1_mammotab.sh
```

MammoTab V2 accuracy: **79.37%**†, up from 79.09% for CoT-SFT. † this run used an earlier, fuller table-row serialization (only 99 of its 9,741 prompts exactly match the common serialization used everywhere else in this document) — see the footnote in section 8; it is reported for completeness and excluded from paired significance testing.

> What happens if `rpo_alpha`/`length_normalize` are turned off? Ablations `result_dpo_v2_nolennorm` (`--length_normalize False`) drop to 69.53% and `result_dpo_v2_norpo` (`--rpo_alpha 0`) drop to 68.52% — these length-debiasing tricks are necessary for this data, not optional (see the ablation table in section 8).



### 6.5 DPO round 2 (regenerate + refilter data from the round-1 checkpoint)

Repeat 6.2 → 6.3 with `--model`/`--checkpoint_dir` pointed at `result_dpo_v2/checkpoint-260`, then train a second round on the new pairs:

```bash
export DEEPSEEK_API_KEY=...
bash scripts/36_gen_dpo_data_cot_round2.sh
bash scripts/37_filter_rl_data_round2.sh
bash scripts/38_train_dpo_cot_round2.sh
```

Evaluate:

```bash
bash scripts/39_eval_dpo_cot_round2_mammotab.sh
```

MammoTab V2 accuracy: **81.85%**, a further improvement over round 1's 79.37%†.

---



## 7. Final comparison on the TableInstruct 6-dataset test set

Evaluate checkpoints from both routes on the TableInstruct entity-linking test set (`ent_link_test_simplified.json`, 2000 samples) to check whether training merely overfits MammoTab or genuinely generalizes to tables from a different source:

```bash
bash scripts/40_eval_tableinstruct_sft.sh               # SFT
bash scripts/41_eval_tableinstruct_dpo_sft_round1.sh     # SFT -> DPO-1 (Route A)
bash scripts/42_eval_tableinstruct_dpo_sft_round2.sh     # SFT -> DPO-2 (Route A)
bash scripts/43_eval_tableinstruct_cot_sft.sh            # CoT-SFT (Route B)
bash scripts/44_eval_tableinstruct_dpo_cot_round1.sh     # CoT-SFT -> DPO-1 (Route B)
bash scripts/45_eval_tableinstruct_dpo_cot_round2.sh     # CoT-SFT -> DPO-2 (Route B)
```

---



## 8. Results summary

The numbers below are the official ones reported in our paper — **TELLER: Dual-Path Iterative Preference Optimization for Table Entity Linking** (Kehao Li, Yixin Peng, Stefan Decker; OM 2026 Workshop @ ISWC, Bari, Italy) — evaluated on the full TableInstruct entity-linking subset and on **MammoTab V2** (the official 9,741-mention / 511-table evaluation set), not on the smaller 3,750-table set used only for in-training checkpoint selection. "Complete rationale" is the percentage of outputs with a well-formed `<think>...</think>` block; it does not apply to direct-answer models.


| Model stage                             | Checkpoint                                                | TableInstruct EL Acc. | TableInstruct EL Complete rat. | MammoTab V2 Acc. | MammoTab V2 Complete rat. |
| --------------------------------------- | --------------------------------------------------------- | --------------------- | ------------------------------ | ---------------- | ------------------------- |
| Llama-3.1-8B (unadapted)                | `meta-llama/Llama-3.1-8B`                                 | 58.60%                | n/a                            | 45.78%           | n/a                       |
| Direct-answer SFT                       | `result_mammo2/checkpoint-21000`                          | 94.35%                | n/a                            | 87.59%           | n/a                       |
| **Route A** — Iterative DPO round 1     | `result_dpo_sft/checkpoint-150`                           | 94.40%                | n/a                            | 88.16%           | n/a                       |
| **Route A** — Iterative DPO round 2     | `result_dpo_sft_i2/checkpoint-100`                        | **94.50%**            | n/a                            | **88.20%**       | n/a                       |
| CoT-SFT                                 | `result_CoT_filtered_compressed_data_lora/checkpoint-350` | 92.90%                | 98.50%                         | 79.09%           | 87.09%                    |
| **Route B** — L-RPO round 1             | `result_dpo_v2/checkpoint-260`                            | 92.90%                | 99.70%                         | 79.37%†          | 89.63%                    |
| **Route B** — L-RPO round 2 (refreshed) | `result_dpo_i2/checkpoint-210`                            | **92.95%**            | 99.55%                         | **81.85%**       | 91.86%                    |


† L-RPO round 1's MammoTab V2 number was evaluated with an earlier, fuller table-row serialization (only 99 of its 9,741 prompts exactly match the common serialization used everywhere else); we report it for completeness but exclude it from paired significance testing.

Comparison with public MammoTab V2 baselines (CEA score = exact-match accuracy here, since each mention gets exactly one prediction):


| Approach                         | CEA (MammoTab V2) |
| -------------------------------- | ----------------- |
| TURL (Deng et al.)               | 0.31              |
| Avogadro 2023                    | 0.62              |
| TableLlama (Zhang et al.)        | 0.86              |
| **Iterative DPO round 2 (ours)** | **0.882**         |


Loss ablations for L-RPO (LN = completion-length normalization, RPO = the chosen-response likelihood term; corresponds to `result_dpo_v2_nolennorm`, `result_dpo_v2_norpo`, `result_dpo_i2_nolennorm` in this repo):


| Variant               | LN  | RPO | MammoTab V2 Acc. | MammoTab V2 Complete rat. |
| --------------------- | --- | --- | ---------------- | ------------------------- |
| L-RPO round 2, full   | yes | yes | **81.85%**       | **91.86%**                |
| L-RPO round 2, no LN  | no  | yes | 79.42%           | 87.54%                    |
| L-RPO round 1, full†  | yes | yes | 79.37%           | 89.63%                    |
| L-RPO round 1, no LN  | no  | yes | 69.53%           | 77.06%                    |
| L-RPO round 1, no RPO | yes | no  | 68.52%           | 74.30%                    |


Takeaways:

- Iterative DPO improves accuracy monotonically on both benchmarks (94.35% → 94.40% → 94.50% on TableInstruct; 87.59% → 88.16% → 88.20% on MammoTab V2). The final MammoTab V2 score (0.882 CEA) surpasses the best previously published score on the public leaderboard (TableLlama, 0.86) by 2.2 points.
- Iterative L-RPO improves both accuracy and reasoning completeness over CoT-SFT (92.90% → 92.95% accuracy and 98.50% → 99.55% complete-rationale rate on TableInstruct; 79.09% → 81.85% accuracy and 87.09% → 91.86% complete-rationale rate on MammoTab V2). On identical prompts, the CoT-SFT → L-RPO-2 transition corrects 817 predictions while regressing only 548 — a net improvement significant under a two-sided exact McNemar test ($p<10^{-12}$).
- Direct-answer models (Route A) remain more accurate than reasoning models (Route B) on both benchmarks — explicit reasoning has a measurable accuracy cost that preference optimization narrows but does not fully close.
- Length normalization and the RPO-style chosen-response term are both necessary for L-RPO: dropping either one collapses round-1 MammoTab V2 accuracy from ~79% (with both) down to 69.53% (no LN) or 68.52% (no RPO); dropping length normalization in round 2 similarly drops accuracy from 81.85% to 79.42%.

---



## 9. Script reference


| Script                                              | Purpose                                                                                                                |
| --------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `mammotab2ti/stage0_build_wikidata_index.py`        | Wikidata dump → local sqlite candidate index                                                                           |
| `mammotab2ti/stage1_extract_jobs.py`                | MammoTab table JSON → per-mention job queue                                                                            |
| `mammotab2ti/stage2_generate_candidates_offline.py` | job + sqlite → tableInstruct-like samples (sharded)                                                                    |
| `mammotab2ti/stage3_merge_single_shards.py`         | Merge shards, backfill gold / shuffle candidates / drop empties                                                        |
| `mammotab2ti/stage4_simplify.py`                    | Simplify table context (same row/col + header, ≤10 rows)                                                               |
| `el/sft.py`                                         | Plain-answer SFT (no CoT)                                                                                              |
| `el/sft_CoT.py`                                     | CoT-SFT with `<think>` reasoning (LoRA, can continue an existing SFT adapter)                                          |
| `el/eval_checkpoint.py`                             | Evaluate a non-CoT checkpoint (SFT / SFT-DPO series)                                                                   |
| `el/eval_CoT_checkpoint.py`                         | Evaluate a CoT checkpoint (CoT-SFT / CoT-DPO series)                                                                   |
| `rl/gen_dpo_data4sft.py`                            | Generate DPO pairs for a non-CoT model (`accept`=gold, `reject`=local model's raw generation)                          |
| `rl/gen_dpo_data.py`                                | Generate DPO pairs for a CoT model (`reject`=local model's generation, `accept`=DeepSeek-teacher-corrected generation) |
| `rl/filter_process_rl_data.py`                      | Structure filter + quality filter + length compression for CoT DPO data                                                |
| `rl/train_dpo4sft.py`                               | DPO training for the non-CoT model (`trl.DPOTrainer`)                                                                  |
| `rl/train_dpo.py`                                   | DPO training for the CoT model (`trl.DPOTrainer`, with length-debiasing / AuxDPO options)                              |


All of the commands above have corresponding shell scripts under `[scripts/](scripts/)`, numbered in execution order — run them directly or copy and adapt as needed.