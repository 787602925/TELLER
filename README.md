# TableLlama Entity Linking：MammoTab → TableInstruct → SFT/CoT-SFT → DPO

本仓库做的事情：把 [MammoTab](https://unimib-datai.github.io/mammotab-docs/) 表格实体链接（Cell-Entity Annotation, CEA）数据转成 [TableInstruct](https://huggingface.co/datasets/osunlp/TableInstruct) 风格的指令微调样本，然后在 `meta-llama/Llama-3.1-8B` 上依次做 SFT、CoT-SFT、DPO(RL) 训练，并在 MammoTab 验证集和 TableInstruct 的 entity-linking 测试集上做评测。

## 目录

1. [数据集下载](#1-数据集下载)
2. [MammoTab → TableInstruct 格式转换](#2-mammotab--tableinstruct-格式转换mammotab2ti)
3. [环境准备](#3-环境准备)
4. [训练流程总览](#4-训练流程总览)
5. [路线 A：SFT → DPO-1 → DPO-2](#5-路线-asft--dpo-1--dpo-2)
6. [路线 B：SFT → CoT-SFT → DPO-1 → DPO-2](#6-路线-bsft--cot-sft--dpo-1--dpo-2)
7. [TableInstruct 6 数据集上的最终对比](#7-tableinstruct-6-数据集上的最终对比)
8. [结果汇总](#8-结果汇总)
9. [脚本速查表](#9-脚本速查表)

---

## 1. 数据集下载

### 1.1 MammoTab（表格 + Wikidata 标注，训练数据来源）

- 项目主页 / 文档：<https://unimib-datai.github.io/mammotab-docs/>
- 代码仓库（生成/重建数据集的 pipeline）：<https://github.com/unimib-datAI/mammotab>
- 数据下载（Zenodo，`mammotab_full.zip`，约 14.6GB，共 838,930 张从英文 Wikipedia 抽取的表格 + CEA/CTA/CPA 标注 + JSON 元数据）：
  <https://zenodo.org/records/16562700>

下载解压后，需要的是"每张表格一个 JSON 文件"的目录（文件里含表格文本 `text` 矩阵、`caption`、页面标题、目标单元格的 gold QID 等），本仓库里默认放在：

```text
~/DATA/mammotab/mammotab_dataset_semtab/json/
```

> 如果你从 Zenodo 下载到的是原始 CSV（table + CEA 标注）+ JSON 元数据的组合，需要先按 MammoTab 官方 pipeline（`unimib-datAI/mammotab` 仓库）或自己写一个转换脚本，把每张表格拼成上面这种单文件 JSON（至少要包含 `text`/表格矩阵、`caption`、页面/章节标题，以及每个目标单元格的 gold QID/wiki 标题），下面第 2 步的 `mammotab2ti/stage1_extract_jobs.py` 就是直接扫描这个目录。

### 1.2 Wikidata truthy dump（离线候选实体检索库）

`mammotab2ti/stage0_build_wikidata_index.py` 需要一份 Wikidata dump 来建本地候选实体索引（sqlite）。推荐用体积小很多的 **truthy** 版本（只含 best-rank 的直接陈述，不含 qualifier/reference）：

```bash
wget -c https://dumps.wikimedia.org/wikidatawiki/entities/latest-truthy.nt.bz2 \
  -O ~/DATA/wikidata/latest-truthy.nt.bz2
```

- 官方索引页：<https://dumps.wikimedia.org/wikidatawiki/entities/>
- 说明页：<https://www.wikidata.org/wiki/Wikidata:Database_download>
- 文件约 40GB（`.nt.bz2`），下载 + 解析建库耗时较长，建议 `nohup` 后台跑（见第 2.1 步）。
- 也支持 `.json`/`.json.gz`/`.json.bz2` 格式的 dump，`stage0_build_wikidata_index.py` 会按后缀自动识别走对应解析分支。

### 1.3 TableInstruct（跨任务指令微调数据集 + out-of-domain 测试集）

- Hugging Face 数据集页：<https://huggingface.co/datasets/osunlp/TableInstruct>
- 论文/代码：<https://github.com/OSU-NLP-Group/TableLlama>

```python
from datasets import load_dataset
ds = load_dataset("osunlp/TableInstruct")
```

本项目主要用到其中的实体链接（entity linking）测试集（`/eval_data` 下），用来做**跨数据集泛化测试**（不是训练数据，只是评测基准，训练全程只用 MammoTab）。评测脚本里对应的文件是（按同一套"同行同列 + 全部表头 + 缩到 10 行"的简化规则处理过）：

```text
/DATA1/khli/tablellama/ent_link_test_simplified.json
```

---

## 2. MammoTab → TableInstruct 格式转换（`mammotab2ti/`）

整条转换 pipeline 分 5 步（stage0 只建索引一次，可复用；stage1~4 是数据处理主链路）：

```text
stage0_build_wikidata_index.py   Wikidata dump -> 本地 sqlite 候选检索库
stage1_extract_jobs.py           88万张表格 JSON -> 逐 mention 的 jobs.jsonl
stage2_generate_candidates_offline.py   jobs.jsonl + sqlite -> tableInstruct-like 样本（分 shard 跑）
stage3_merge_single_shards.py    合并各 shard 输出 + 数据清洗
stage4_simplify.py               只保留同行同列 + 全部表头，行数压到 <=10
```

### Stage 0：建 Wikidata 本地索引

```bash
nohup python -m mammotab2ti.stage0_build_wikidata_index \
  --dump ~/DATA/wikidata/latest-truthy.nt.bz2 \
  --db ~/DATA/wikidata/wikidata_candidates.sqlite \
  > build_index.log 2>&1 &
```

- 输入：`--dump` Wikidata dump 文件。
- 输出：`--db` 一个 sqlite 文件，里面有 `entities`（qid/label/description/p31/aliases/enwiki_title/popularity...）和 `alias_index` 两张表，供 stage2 检索候选实体用。
- 这一步只需要跑一次，之后所有 stage2 任务复用同一个 `.sqlite`。

### Stage 1：从 MammoTab JSON 抽取 mention 级别的 job 队列

```bash
python -m mammotab2ti.stage1_extract_jobs \
  --input_dir ~/DATA/mammotab/mammotab_dataset_semtab/json \
  --output_jobs ~/DATA/mammotab/modified_mammotab/mammotab_jobs.jsonl
```

- 把 88 万张表格 JSON 展开成"一个待链接的实体 mention 一条记录"的 `jobs.jsonl`（每条含 `mention`/`column_name`/`row_index`/`page_title`/`section_title`/`table_caption`/`gold_qid`/`gold_wiki_title` 等）。

### Stage 2：本地候选检索 + 组装 tableInstruct-like 样本

```bash
python -m mammotab2ti.stage2_generate_candidates_offline \
  --jobs ~/DATA/mammotab/modified_mammotab/mammotab_jobs.jsonl \
  --wikidata_db ~/DATA/wikidata/wikidata_candidates.sqlite \
  --output ~/DATA/mammotab/modified_mammotab/stage2_single_shards/ent_link_train_generated.shard_0.jsonl \
  --audit_output ~/DATA/mammotab/modified_mammotab/stage2_single_shards/ent_link_train_generated.audit.shard_0.jsonl \
  --num_shards 24 --shard_id 0 --resume
```

- 对每条 mention，用 stage0 建好的本地索引取 top-20 候选实体，拼成 `instruction/input_seg/question/output` 四字段的样本（`output` 是 gold 答案的候选格式化文本，`question` 里带着未打乱的候选列表）。
- **不会**强制把 gold 塞进候选列表，同时写一份 `--audit_output`，记录每条样本的 `candidate_qids`/`gold_in_topk`，供 stage3 做清洗、以及事后统计"gold 有没有进 top20"。
- 数据量大（88 万张表 = 更多 mention），建议按 `--num_shards`/`--shard_id` 分片、每个 shard 起一个进程并行跑（这是纯 CPU/sqlite IO 任务，用 `python` 直接跑多个进程即可，不需要 `torchrun`），`--resume` 支持断点续跑。

### Stage 3：合并 shard + 数据清洗

```bash
python -m mammotab2ti.stage3_merge_single_shards \
  --shard_dir ~/DATA/mammotab/modified_mammotab/stage2_single_shards \
  --output_data ~/DATA/mammotab/modified_mammotab/stage3_single_shards_merge.jsonl \
  --output_audit ~/DATA/mammotab/modified_mammotab/stage3_single_shards_audit_merge.jsonl \
  --num_shards 24
```

把 stage2 分 shard 生成的文件合并成一份，同时做三件事：

1. **补 gold**：如果 gold 答案不在候选列表里，把它加进候选列表（gold 是 `NIL` 或空字符串时不加）；
2. **打乱候选顺序**：用 `--seed` 固定的随机数对每条样本的候选列表重新排序，避免"正确答案总在同一位置"这种位置偏置；
3. **丢弃空候选**：候选列表为空的样本直接扔掉。

### Stage 4：简化表格上下文（只留同行同列 + 全部表头，行数 ≤10）

```bash
python -m mammotab2ti.stage4_simplify \
  --in_jsonl ~/DATA/mammotab/modified_mammotab/stage3_single_shards_merge.jsonl \
  --out_json ~/DATA/mammotab/modified_mammotab/stage4_single_shards_merge_simplified.json
```

目标：

1. 保留表头 + 目标实体所在的整行 + 其他行里"实体所在列"的那一格；
2. 只保留目标实体行附近的窗口（默认最多 10 行）；
3. 输出为一个 JSON 数组（`.json`），供后面 SFT/CoT-SFT/DPO 训练脚本直接读取。

关键是 `_infer_entity_position_from_input_and_question()` 怎么找到"实体在哪一行哪一列"：

1. 从 `question` 里正则抽两段信息：`selected entity mention in the table cell is: ...` 里的 **mention**，以及 `The column name for '...' is ...` 里的 **列名**；任一段抽不到就判定推断失败（返回 `None`）。
2. 从 `input_seg` 里把文本切成两段：`[TAB] ... col: ...` 之前的 `prefix`（用来解析表头 header cells），和从第一个 `row N:` 开始的 `rows_part`。
3. 把表头做空白归一化，用列名在表头列表里的位置定位**实体列 index**；找不到就失败。
4. 把 `rows_part` 按 `[SEP]` 切成每一行，每行再按 `row N: ...` 解析出行号和各单元格文本；逐行检查该行"实体列"对应的单元格文本里是否（大小写不敏感）包含 mention，第一个命中的行就是**实体行号**。
5. 都没命中则整体推断失败，此时默认（不加 `--keep_on_infer_failure`）会**丢弃**该样本；加上这个开关则保留原始未简化的 `input_seg`。

得到的 `stage4_single_shards_merge_simplified.json` 就是本仓库训练/评测统一使用的数据格式（`instruction`/`input_seg`/`question`/`output` 四字段），下文所有 `--data_path`/`--eval_data_path` 指的都是这类文件（或从其中切出的验证子集，如 `..._mam_val_last3750.json`）。

---

## 3. 环境准备

```bash
conda create -n tablellama-fa python=3.10
conda activate tablellama-fa
pip install torch transformers peft accelerate trl==1.8.0 \
    flash-attn --no-build-isolation \
    wandb tqdm
```

- 训练脚本（`el/sft.py` / `el/sft_CoT.py` / `rl/train_dpo.py` / `rl/train_dpo4sft.py`）都用 `transformers.HfArgumentParser` 的 dataclass 参数风格，命令行直接 `--字段名 值` 传参即可，多卡用 `torchrun --nproc_per_node=<GPU数>` 拉起。
- DPO 用的是 `trl.DPOTrainer`（固定 `trl==1.8.0`，其它版本字段可能不兼容）。
- CoT 阶段的 RL 数据生成（`rl/gen_dpo_data.py`）需要调用 DeepSeek 教师模型纠正本地模型答错的样本，需要先 `export DEEPSEEK_API_KEY=...`；SFT（非 CoT）阶段的 RL 数据生成（`rl/gen_dpo_data4sft.py`）不需要教师模型，直接用数据集里的 gold 作为 accept。
- base 模型：`meta-llama/Llama-3.1-8B`（需要 HuggingFace 账号 accept license 后 `huggingface-cli login`）。

---

## 4. 训练流程总览

两条路线共享同一个 SFT 起点（`result_mammo2/checkpoint-21000`），之后分叉：

```text
                         el/sft.py (SFT)
                               │
                 result_mammo2/checkpoint-21000
                 ┌─────────────┴─────────────────────────┐
                 │                                       │
        路线 A（纯答案，不带推理）              路线 B（先学会 <think> 推理）
                 │                                       │
   rl/gen_dpo_data4sft.py                         el/sft_CoT.py (CoT-SFT, LoRA)
   rl/train_dpo4sft.py (DPO-1)                result_CoT_filtered_compressed_data_lora
   → result_dpo_sft                                       │
                 │                          rl/gen_dpo_data.py (DeepSeek 教师纠错)
   同样流程 (DPO-2)                          rl/filter_process_rl_data.py (过滤/压缩)
   → result_dpo_sft_i2                       rl/train_dpo.py (DPO-1) → result_dpo_v2
                                                             │
                                              同样流程 (DPO-2) → result_dpo_i2
```

两条路线的每个 checkpoint 都用两套评测：

- **MammoTab 验证集**（in-domain，`el/eval_checkpoint.py` 或 `el/eval_CoT_checkpoint.py`）；
- **TableInstruct 实体链接测试集**（out-of-domain 泛化测试，见第 7 节）。

---

## 5. 路线 A：SFT → DPO-1 → DPO-2

### 5.1 SFT（`el/sft.py`）

```bash
torchrun --nproc_per_node=2 el/sft.py \
  --bf16 True \
  --output_dir result_mammo2 \
  --eval_data_path /DATA1/khli/mammotab/modified_mammotab/stage4_single_shards_merge_simplified_mam_val_last3750.json \
  --eval_strategy steps \
  --eval_steps 1000 \
  --eval_data_limit 500 \
  --logging_steps 10 \
  --save_strategy steps \
  --save_steps 1000 \
  --save_total_limit 4 \
  --learning_rate 2e-5 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --tf32 True \
  --run_name sft2 \
  --num_train_epochs 2 \
  --per_device_train_batch_size 3 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 1 \
  --model_max_length 2048 \
  --use_flash_attn True \
  --report_to wandb \
  > result_mammo2/train.log 2>&1
```

评测（MammoTab 验证集）：

```bash
torchrun --nproc_per_node=2 --master_port=29615 el/eval_checkpoint.py \
  --checkpoint_dir result_mammo2/checkpoint-21000 \
  --eval_data_path /DATA1/khli/mammotab/mammotab_dataset_semtab/mammotab_2024_prompts_50.json \
  --output_dir result_mammo2/eval_checkpoint-21000_v2 \
  --step 21000 \
  --model_max_length 2048 \
  --max_new_tokens 128 \
  --use_flash_attn True \
  --bf16 True \
  > result_mammo2/eval_sft_21000_v2.log 2>&1 &
```

MammoTab 验证集 accuracy：**0.8759**（8532/9741）。

### 5.2 生成 DPO 训练数据（`rl/gen_dpo_data4sft.py`）

用 SFT checkpoint 对训练数据做推理，答错的样本里：`accept` = gold 答案（规范化为候选实体格式），`reject` = 本地模型的原始生成。

```bash
torchrun --nproc_per_node=3 rl/gen_dpo_data4sft.py \
    --gen_batch_size 24 \
    --sync_every_rounds 5 \
    > rl/data_generate_sft_i2_full.log 2>&1 &
```

> 默认读 `result_mammo2/checkpoint-21000`（可用 `--model` 指定其它 checkpoint，比如第二轮时指向 `result_dpo_sft/checkpoint-150`），`--num` 控制样本量，`--resume` 支持多卡断点续跑（读取 `.rank*.jsonl` 分片）。这一步产出的 `{prompt, accept, reject}` pair 直接喂给 `rl/train_dpo4sft.py`，不需要像路线 B 那样再跑 `filter_process_rl_data.py`（因为这里的 `accept` 只是一句候选实体格式的 gold，没有 CoT 推理需要结构校验/压缩）。

### 5.3 DPO 第一轮（`rl/train_dpo4sft.py`）

```bash
torchrun --nproc_per_node=2 rl/train_dpo4sft.py \
    --output_dir result_dpo_sft \
    --data_path /DATA1/khli/t\&m/rl_sft_checkpoint-21000_all.jsonl \
    --bf16 True \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-6 \
    --num_train_epochs 2 \
    --gen_eval_steps 50 \
    --gen_eval_data_path /DATA1/khli/mammotab/modified_mammotab/stage4_single_shards_merge_simplified_mam_val_last3750.json \
    --gen_eval_num_samples 500 \
    --gen_eval_batch_size 16 \
    --gen_eval_max_new_tokens 128 \
    --save_best_by_gen_eval True \
    --save_total_limit 5 \
    > /home/khli/tableLlama/result_dpo_sft/rl.log 2>&1
```

默认从 `result_mammo2/checkpoint-21000` 这个 adapter 继续训练（`--adapter_name_or_path` 可覆盖）。`--save_best_by_gen_eval True` 会在训练过程中周期性用真实 generate 在 `--gen_eval_data_path` 上跑一批样本估计 accuracy，按分数保留最好的几个 checkpoint。

评测（MammoTab 验证集）：

```bash
torchrun --nproc_per_node=3 --master_port=29612 el/eval_checkpoint.py \
  --checkpoint_dir result_dpo_sft/checkpoint-150 \
  --eval_data_path /DATA1/khli/mammotab/mammotab_dataset_semtab/mammotab_2024_prompts_50.json \
  --output_dir result_dpo_sft/eval_checkpoint-150 \
  --step 150 \
  --model_max_length 2048 \
  --max_new_tokens 128 \
  --use_flash_attn True \
  --bf16 True \
  > result_dpo_sft/eval_checkpoint-150.log 2>&1 &
```

MammoTab 验证集 accuracy：**0.8816**（8588/9741），相对 SFT 的 0.8759 有提升。

### 5.4 DPO 第二轮（用第一轮 checkpoint 重新生成数据）

重新用 `result_dpo_sft/checkpoint-150` 对训练数据推理，产出新一批 DPO pair（同 5.2，把 `--model` 指向 `result_dpo_sft/checkpoint-150`），再训练第二轮：

```bash
torchrun --nproc_per_node=3 rl/train_dpo4sft.py \
    --output_dir result_dpo_sft_i2 \
    --data_path /DATA1/khli/t\&m/rl_sft_checkpoint-150_all.jsonl \
    --bf16 True \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-6 \
    --num_train_epochs 2 \
    --gen_eval_steps 50 \
    --gen_eval_data_path /DATA1/khli/mammotab/modified_mammotab/stage4_single_shards_merge_simplified_mam_val_last3750.json \
    --gen_eval_num_samples 500 \
    --gen_eval_batch_size 16 \
    --gen_eval_max_new_tokens 128 \
    --save_best_by_gen_eval True \
    --save_total_limit 5 \
    > /home/khli/tableLlama/result_dpo_sft_i2/rl.log 2>&1
```

> 注意 `--adapter_name_or_path` 需要指向 `result_dpo_sft/checkpoint-150`（继续训练同一条 LoRA），而不是从 base SFT checkpoint 重新开始。

评测：

```bash
torchrun --nproc_per_node=4 --master_port=29612 el/eval_checkpoint.py \
  --checkpoint_dir result_dpo_sft_i2/checkpoint-100 \
  --eval_data_path /DATA1/khli/mammotab/mammotab_dataset_semtab/mammotab_2024_prompts_50.json \
  --output_dir result_dpo_sft_i2/eval_checkpoint-100 \
  --step 100 \
  --model_max_length 2048 \
  --max_new_tokens 128 \
  --use_flash_attn True \
  --bf16 True \
  > result_dpo_sft_i2/eval_checkpoint-100.log 2>&1 &
```

MammoTab 验证集 accuracy：**0.8749**（8522/9741），与第一轮基本持平（第一轮的验证集提升已接近饱和），但在 5.5 的 out-of-domain 测试集上仍有小幅提升（见第 7 节）。

---

## 6. 路线 B：SFT → CoT-SFT → DPO-1 → DPO-2

### 6.1 CoT-SFT（`el/sft_CoT.py`，在 SFT LoRA 基础上继续训练一个新的 LoRA）

在路线 A 的 SFT checkpoint（`result_mammo2/checkpoint-21000`）基础上，用带 `<think>...</think>` 推理过程的 CoT 数据训一个新的 LoRA adapter（`--lora_only True` 表示只训新 LoRA，不动 embed/norm 等其它参数）：

```bash
torchrun --nproc_per_node=2 --master_port=29612 el/sft_CoT.py \
  --output_dir /home/khli/tableLlama/result_CoT_filtered_compressed_data_lora \
  --bf16 True \
  --model_name_or_path meta-llama/Llama-3.1-8B \
  --adapter_name_or_path /home/khli/tableLlama/result_mammo2/checkpoint-21000 \
  --data_path "/DATA1/khli/t&m/merged_CoTSFT_all&all_filtered_cot_compressed.json" \
  --eval_data_path "/DATA1/khli/mammotab/modified_mammotab/stage4_single_shards_merge_simplified_mam_val_last3750.json" \
  --eval_data_limit 200 \
  --eval_strategy steps \
  --eval_steps 50 \
  --save_strategy steps \
  --save_steps 50 \
  --save_total_limit 4 \
  --load_best_model_at_end True \
  --metric_for_best_model accuracy \
  --greater_is_better True \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 1 \
  --model_max_length 3072 \
  --max_output_tokens 768 \
  --logging_steps 20 \
  --ddp_timeout 36000 \
  --report_to wandb \
  --lora_only True \
  > result_CoT_filtered_compressed_data_lora/train_lora.log 2>&1
```

> CoT 训练数据（`merged_CoTSFT_all&all_filtered_cot_compressed.json`）本身是通过教师模型（DeepSeek）+ 后处理管线（结构过滤 → 质量过滤 → 长度压缩，见 6.3 的 `rl/filter_process_rl_data.py` 复用的三个函数）构造出的带 `<think>` 推理链的 SFT 样本；如何生成这份数据不在本文档范围内，可参照 `data_clean/compress_long_cot.py`、`data_clean/filter_low_info_cot.py`、`data_aug/augment_ent_link_thinking.py` 的思路。

评测（`el/eval_CoT_checkpoint.py`）：

```bash
torchrun --nproc_per_node=4 --master_port=29612 el/eval_CoT_checkpoint.py \
  --checkpoint_dir result_CoT_filtered_compressed_data_lora/checkpoint-350 \
  --data_path "/DATA1/khli/mammotab/mammotab_dataset_semtab/mammotab_2024_prompts_50.json" \
  --out_path result_CoT_filtered_compressed_data_lora/checkpoint-350_mam_val_predictions.jsonl \
  --eval_data_limit -1 \
  --max_new_tokens 768 \
  --min_new_tokens 2 \
  --resume \
  > result_CoT_filtered_compressed_data_lora/eval_checkpoint-350.log 2>&1
```

MammoTab 验证集 accuracy：**0.7822**（704/900，小样本口径）/ **0.7909**（7704/9741，全量口径）。CoT-SFT 单独看比纯 SFT（0.8759）略低——这是预期内的，因为它要多学一步"先推理再回答"，后面靠 DPO 把推理质量和答案正确率一起提上去。

### 6.2 生成 CoT-DPO 训练数据（`rl/gen_dpo_data.py`，本地模型 + DeepSeek 教师纠错）

```bash
export DEEPSEEK_API_KEY=...
torchrun --nproc_per_node=2 rl/gen_dpo_data.py \
  --model result_CoT_filtered_compressed_data_lora/checkpoint-350 \
  --input_data_file '/DATA1/khli/t&m/merged_SFT_70k&35k.json' \
  --num 9000 \
  --accept_max_reasoning_sentences 5 \
  --accept_max_reasoning_chars 800 \
  --resume > rl/data_generate_resume_9000.log 2>&1
```

流程：本地模型对训练数据批量推理 → 找出预测错误的样本，把其原始生成（含 `<think>`）作为 `reject` → 把"正确答案 + 要求不能引用/暗示答案"的 prompt 发给 DeepSeek 教师模型，生成一段真正推理出正确答案的 `<think>...</think>` 作为 `accept`（`--accept_max_reasoning_*` 控制教师生成的推理长度上限）。多卡按 round 轮转分片并行、`--resume` 支持断点续跑。

### 6.3 过滤 + 压缩 DPO 数据（`rl/filter_process_rl_data.py`）

```bash
python rl/filter_process_rl_data.py \
  --in_path "/DATA1/khli/t&m/rl_checkpoint-350_10000.jsonl" \
  --out_path "/DATA1/khli/t&m/rl_checkpoint-350_10000_filtered.jsonl"
```

三步流水线（全程 CPU，不需要 GPU）：

1. **结构过滤**：`accept` 必须能解析出 `<think>...</think>\n<entity>` 的结构，解析失败的整条丢弃；同时校验解析出的最终实体是否与 gold 一致，不一致也丢弃。`reject` 不做结构要求（截断/重复本身也是有信息量的负样本），只清理明显的格式崩坏。
2. **质量过滤**：给 `accept` 的推理内容打分，去掉啰嗦（rambling）、答案泄露（answer leakage）、照抄候选列表（copies candidate list）、没有实际论证（no justification）等低信息量样本。
3. **长度压缩**：`accept` 超过 `--compress-threshold-tokens` 时做抽取式压缩到 `--target-output-tokens` 以内；压缩失败（解析不出来 / 实体变了 / 仍超长 / 重复率太高）的样本整条丢弃。

可以先加 `--dry_run` 看每一步的漏斗统计，确认比例合理后再正式写出最终文件。

### 6.4 DPO 第一轮（`rl/train_dpo.py`）

```bash
torchrun --nproc_per_node=2 --master_port=29621 rl/train_dpo.py \
  --data_path "/DATA1/khli/t&m/rl_checkpoint-350_9000_filtered.jsonl" \
  --adapter_name_or_path result_CoT_filtered_compressed_data_lora/checkpoint-350 \
  --output_dir result_dpo_v2 \
  --bf16 True \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1.5e-6 \
  --beta 0.3 \
  --rpo_alpha 0.5 \
  --num_train_epochs 1 \
  --eval_ratio 0.05 --eval_steps 50 \
  --max_chosen_reject_len_ratio 6.0 \
  --gen_eval_data_path /DATA1/khli/mammotab/modified_mammotab/stage4_single_shards_merge_simplified_mam_val_last3750.json \
  --gen_eval_steps 20 --gen_eval_num_samples 48 --gen_eval_batch_size 8 \
  --gen_eval_max_new_tokens 768 \
  --save_best_by_gen_eval True --save_total_limit 5 \
  --report_to none \
  > /home/khli/tableLlama/result_dpo_v2/rl.log 2>&1
```

几个关键超参（脚本头部注释里有详细踩坑记录，简述如下）：

- `--rpo_alpha 0.5`：CoT 数据里 `accept`（教师生成的完整推理）系统性比 `reject`（本地模型的原始生成）长很多，标准 DPO 的 sigmoid pairwise loss 对序列 log-prob 求和、不做长度归一化，天然带"越长越优"的虚假梯度。`rpo_alpha>0` 会给 `loss_type` 加一个 `sft`（对 chosen 的交叉熵 NLL）分量，防止 chosen 似然被无意压低。
- `--length_normalize`（默认开）：把 pairwise loss 换成长度归一化版本，进一步缓解长度偏置。
- `--max_chosen_reject_len_ratio 6.0`：丢弃 `chosen`/`rejected` token 长度比过大的异常样本。
- `--save_best_by_gen_eval True`：按周期性真实 generate 得到的 `accuracy + has_think` 分数保留最好的 checkpoint。

评测：

```bash
torchrun --nproc_per_node=4 --master_port=29610 el/eval_CoT_checkpoint.py \
  --checkpoint_dir /home/khli/tableLlama/result_dpo_v2/checkpoint-260 \
  --data_path "/DATA1/khli/mammotab/mammotab_dataset_semtab/mammotab_2024_prompts_50.json" \
  --out_path /home/khli/tableLlama/result_dpo_v2/checkpoint-260_mam_val_predictions.jsonl \
  --eval_data_limit -1 \
  --max_new_tokens 768 \
  --min_new_tokens 2 \
  --resume \
  > /home/khli/tableLlama/result_dpo_v2/eval_checkpoint-260.log 2>&1
```

MammoTab 验证集 accuracy：**0.7938**（7747/9759），相对 CoT-SFT 的 0.7909 有提升。

> `rpo_alpha`/`length_normalize` 关掉会怎样？消融实验 `result_dpo_v2_nolennorm`（`--length_normalize False`）验证集掉到 0.6953，`result_dpo_v2_norpo`（`--rpo_alpha 0`）掉到 0.6852——说明这两个长度去偏技巧对这份数据是必需的，不是可选项。

### 6.5 DPO 第二轮（用第一轮 checkpoint 重新生成 + 过滤数据）

重复 6.2 → 6.3，把 `--model`/`--checkpoint_dir` 换成 `result_dpo_v2/checkpoint-260`，产出新一批 pair 数据后训练第二轮：

```bash
torchrun --nproc_per_node=4 rl/gen_dpo_data.py \
  --model result_dpo_v2/checkpoint-260 \
  --input_data_file '/DATA1/khli/t&m/merged_SFT_70k&35k.json' \
  --num 5000 \
  --accept_max_reasoning_sentences 5 \
  --accept_max_reasoning_chars 800 \
  --resume > rl/data_generate_iteration2_5000.log 2>&1
```

```bash
python rl/filter_process_rl_data.py \
  --in_path "/DATA1/khli/t&m/rl_checkpoint-260_5000.jsonl" \
  --out_path "/DATA1/khli/t&m/rl_checkpoint-260_5000_filtered.jsonl"
```

```bash
torchrun --nproc_per_node=2 --master_port=29621 rl/train_dpo.py \
  --data_path "/DATA1/khli/t&m/rl_checkpoint-260_5000.jsonl" \
  --adapter_name_or_path result_dpo_v2/checkpoint-260 \
  --output_dir result_dpo_i2 \
  --bf16 True \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1.5e-6 \
  --beta 0.3 \
  --rpo_alpha 0.5 \
  --num_train_epochs 1 \
  --max_chosen_reject_len_ratio 6.0 \
  --gen_eval_data_path /DATA1/khli/mammotab/modified_mammotab/stage4_single_shards_merge_simplified_mam_val_last3750.json \
  --gen_eval_steps 30 --gen_eval_num_samples 48 --gen_eval_batch_size 8 \
  --gen_eval_max_new_tokens 768 \
  --save_best_by_gen_eval True --save_total_limit 5 \
  --report_to none \
  > /home/khli/tableLlama/result_dpo_i2/rl.log 2>&1
```

评测：

```bash
torchrun --nproc_per_node=3 --master_port=29612 el/eval_CoT_checkpoint.py \
  --checkpoint_dir /home/khli/tableLlama/result_dpo_i2/checkpoint-210 \
  --data_path "/DATA1/khli/mammotab/mammotab_dataset_semtab/mammotab_2024_prompts_50.json" \
  --out_path /home/khli/tableLlama/result_dpo_i2/checkpoint-210_full_mam_val_predictions.jsonl \
  --eval_data_limit -1 \
  --max_new_tokens 768 \
  --min_new_tokens 2 \
  --resume \
  > /home/khli/tableLlama/result_dpo_i2/eval_checkpoint-210_full.log 2>&1
```

MammoTab 验证集 accuracy：**0.8185**（7973/9741），相对第一轮的 0.7938 继续提升。

---

## 7. TableInstruct 6 数据集上的最终对比

用两条路线各阶段的 checkpoint 在 TableInstruct 的实体链接测试集（`ent_link_test_simplified.json`，2000 条）上评测，检验训练是否只是过拟合 MammoTab、还是能泛化到别的表格来源：

```bash
# SFT
torchrun --nproc_per_node=4 --master_port=29615 el/eval_checkpoint.py \
  --checkpoint_dir result_mammo2/checkpoint-21000 \
  --eval_data_path /DATA1/khli/tablellama/ent_link_test_simplified.json \
  --output_dir result_mammo2/eval_checkpoint-21000_ti \
  --step 21000 --model_max_length 2048 --max_new_tokens 128 \
  --use_flash_attn True --bf16 True \
  > result_mammo2/eval_sft_21000_ti.log 2>&1 &

# SFT -> DPO-1（路线 A）
torchrun --nproc_per_node=3 --master_port=29615 el/eval_checkpoint.py \
  --checkpoint_dir result_dpo_sft/checkpoint-150 \
  --eval_data_path /DATA1/khli/tablellama/ent_link_test_simplified.json \
  --output_dir result_dpo_sft/checkpoint-150_ti \
  --step 150 --model_max_length 2048 --max_new_tokens 128 \
  --use_flash_attn True --bf16 True \
  > result_dpo_sft/eval_sft_150_ti.log 2>&1

# SFT -> DPO-2（路线 A）
torchrun --nproc_per_node=3 --master_port=29615 el/eval_checkpoint.py \
  --checkpoint_dir result_dpo_sft_i2/checkpoint-100 \
  --eval_data_path /DATA1/khli/tablellama/ent_link_test_simplified.json \
  --output_dir result_dpo_sft_i2/checkpoint-100_ti \
  --step 150 --model_max_length 2048 --max_new_tokens 128 \
  --use_flash_attn True --bf16 True \
  > result_dpo_sft_i2/eval_sft_100_ti.log 2>&1

# CoT-SFT（路线 B）
torchrun --nproc_per_node=4 --master_port=29612 el/eval_CoT_checkpoint.py \
  --checkpoint_dir result_CoT_filtered_compressed_data_lora/checkpoint-350 \
  --data_path "/DATA1/khli/tablellama/ent_link_test_simplified.json" \
  --out_path result_CoT_filtered_compressed_data_lora/checkpoint-350_full_ti_val_predictions.jsonl \
  --eval_data_limit -1 --max_new_tokens 768 --min_new_tokens 2 --resume \
  > result_CoT_filtered_compressed_data_lora/eval_checkpoint-350_ti.log 2>&1

# CoT-SFT -> DPO-1（路线 B）
torchrun --nproc_per_node=4 --master_port=29611 el/eval_CoT_checkpoint.py \
  --checkpoint_dir result_dpo_v2/checkpoint-260 \
  --data_path "/DATA1/khli/tablellama/ent_link_test_simplified.json" \
  --out_path result_dpo_v2/checkpoint-260_full_ti_val_predictions.jsonl \
  --eval_data_limit -1 --max_new_tokens 768 --min_new_tokens 2 --resume \
  > result_dpo_v2/checkpoint-260_ti.log 2>&1

# CoT-SFT -> DPO-2（路线 B）
torchrun --nproc_per_node=3 --master_port=29613 el/eval_CoT_checkpoint.py \
  --checkpoint_dir result_dpo_i2/checkpoint-210 \
  --data_path "/DATA1/khli/tablellama/ent_link_test_simplified.json" \
  --out_path result_dpo_i2/checkpoint-210_full_ti_val_predictions.jsonl \
  --eval_data_limit -1 --max_new_tokens 768 --min_new_tokens 2 --resume \
  > result_dpo_i2/checkpoint-210_ti.log 2>&1
```

---

## 8. 结果汇总

| 阶段 | Checkpoint | MammoTab 验证集 accuracy | TableInstruct 实体链接测试集 accuracy (2000条) |
| --- | --- | --- | --- |
| SFT | `result_mammo2/checkpoint-21000` | 0.8759 (8532/9741) | 0.9435 |
| **路线 A** SFT → DPO-1 | `result_dpo_sft/checkpoint-150` | 0.8816 (8588/9741) | 0.9440 (1888/2000) |
| **路线 A** SFT → DPO-2 | `result_dpo_sft_i2/checkpoint-100` | 0.8749 (8522/9741) | **0.9450** (1890/2000) |
| **路线 B** CoT-SFT | `result_CoT_filtered_compressed_data_lora/checkpoint-350` | 0.7909 (7704/9741) | 0.9290 (1858/2000); has_think=1970/2000 |
| **路线 B** CoT-SFT → DPO-1 | `result_dpo_v2/checkpoint-260` | 0.7938 (7747/9759) | 0.9290 (1858/2000); has_think=1994/2000 |
| **路线 B** CoT-SFT → DPO-2 | `result_dpo_i2/checkpoint-210` | 0.8185 (7973/9741) | 0.9295 (1859/2000); has_think=1991/2000 |

简单结论：

- 两条路线的 DPO 迭代在 in-domain（MammoTab 验证集）和 out-of-domain（TableInstruct 测试集）上都基本呈单调（或至少不倒退）的提升趋势，说明 RL 阶段带来的是真实能力提升，不是单纯过拟合验证集。
- 路线 A（不带推理，直接输出答案）在这个任务上比路线 B（带 `<think>` 推理）accuracy 更高，但路线 B 的收益点在于可解释的推理链（`has_think` 接近 100%），以及为后续需要中间推理步骤的场景（比如更难的多跳表格任务）留下可扩展性。
- DPO 训练里的长度去偏技巧（`--length_normalize`、`--rpo_alpha`）在路线 B 是必需的：消融实验去掉任一个都会让验证集 accuracy 从 ~0.79 掉到 ~0.68-0.70（见 `result_dpo_v2_nolennorm`、`result_dpo_v2_norpo`）。

---

## 9. 脚本速查表

| 脚本 | 作用 |
| --- | --- |
| `mammotab2ti/stage0_build_wikidata_index.py` | Wikidata dump → 本地 sqlite 候选检索库 |
| `mammotab2ti/stage1_extract_jobs.py` | MammoTab 表格 JSON → 逐 mention job 队列 |
| `mammotab2ti/stage2_generate_candidates_offline.py` | job + sqlite → tableInstruct-like 样本（分 shard） |
| `mammotab2ti/stage3_merge_single_shards.py` | 合并 shard，补 gold / 打乱候选 / 丢空候选 |
| `mammotab2ti/stage4_simplify.py` | 简化表格上下文（同行同列 + 表头，≤10 行） |
| `el/sft.py` | 纯答案 SFT（无 CoT） |
| `el/sft_CoT.py` | 带 `<think>` 推理的 CoT-SFT（LoRA，可 continue 已有 SFT adapter） |
| `el/eval_checkpoint.py` | 评测非 CoT checkpoint（SFT / SFT-DPO 系列） |
| `el/eval_CoT_checkpoint.py` | 评测 CoT checkpoint（CoT-SFT / CoT-DPO 系列） |
| `rl/gen_dpo_data4sft.py` | 为非 CoT 模型生成 DPO pair（`accept`=gold，`reject`=本地模型原始生成） |
| `rl/gen_dpo_data.py` | 为 CoT 模型生成 DPO pair（`reject`=本地模型生成，`accept`=DeepSeek 教师纠错生成） |
| `rl/filter_process_rl_data.py` | CoT DPO 数据的结构过滤 + 质量过滤 + 长度压缩 |
| `rl/train_dpo4sft.py` | 非 CoT 模型的 DPO 训练（`trl.DPOTrainer`） |
| `rl/train_dpo.py` | CoT 模型的 DPO 训练（`trl.DPOTrainer`，含长度去偏 / AuxDPO 等选项） |
