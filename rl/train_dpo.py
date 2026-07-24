#!/usr/bin/env python3
"""
DPO (Direct Preference Optimization) 训练脚本：在已有 CoT-SFT LoRA checkpoint 的基础上，
用 rl/gen_dpo_data.py -> rl/filter_process_rl_data.py 产出的 accept/reject pair 数据继续做
偏好优化（RL 阶段），底层用 trl.DPOTrainer（本机 trl==1.8.0）。

===================== 关键设计点（写之前 / 改之前请先看这里） =====================

1) 继续训练同一个 LoRA adapter，而不是新开一个：
   默认 --adapter_name_or_path 指向 result_CoT_filtered_compressed_data_lora/checkpoint-350，
   直接 `PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=True)` 加载成可训练的
   PeftModel，不给 DPOTrainer 传 `peft_config`。trl 检测到传入的 model 已经是 PeftModel 且
   `ref_model=None` 时，会自动执行 `model.add_adapter("ref", default_config)` 并把 "default"
   adapter 的权重复制一份到新建的 "ref" adapter，训练时通过 disable/enable adapter 切换来算
   reference log-prob —— 也就是说参考模型 = 训练开始那一刻的 checkpoint-350，且不需要在显存里
   常驻一份完整的 8B 权重。训练结束保存前会显式 delete_adapter("ref")，只落盘 "default"。

2) EOS 处理：checkpoint 的 tokenizer 是训练时用 add_special_tokens 加的自定义 `"</s>"`
   （不是 Llama-3.1 原生的 <|eot_id|>），必须从 --adapter_name_or_path 目录加载 tokenizer
   （里面已经 resize 过 vocab）。rl/gen_dpo_data.py 产出的 accept/reject 文本本身不含结尾
   EOS（生成时特意 strip 掉了/是 teacher 原始输出），trl 的 `_prepare_dataset` 会在
   tokenize 之前自动给不以 eos_token 结尾的 chosen/rejected 补上 tokenizer.eos_token，
   这里不需要再手动处理，但 tokenizer 必须来自 checkpoint 目录，否则 eos id 会对不上。

3) 长度处理 —— 不依赖 trl 的截断，训练前显式过滤超长样本：
   trl 的 DataCollatorForPreference 在样本超过 `--max_length` 时，是对
   "prompt_ids + completion_ids" 拼接后的整段序列做切片截断（`truncation_mode="keep_start"`
   时 = `ids[:max_length]`，即从头保留、砍掉尾部）。这意味着如果某条样本的 prompt 已经接近甚至
   超过 max_length，被截断后 completion（含最终答案实体）会被整段砍掉，DPO 会在一段不完整/
   语义不对齐的序列上算 loss，属于静默污染数据。所以这里在构造数据集时，用同一个 tokenizer
   显式计算每条样本 `len(tokenize(prompt + accept))` / `len(tokenize(prompt + reject))`，超过
   --max_length 的样本直接丢弃并打印统计（而不是交给 trl 静默截断）。对
   /DATA1/khli/t&m/rl_checkpoint-350_10000_filtered.jsonl 实测：prompt token 数中位数约 821、
   p99 约 1161（个别表格候选很长的样本能到 3000+）；accept 被 filter_process_rl_data.py 压缩到
   <=700 token；--max_length 默认设 2560，实测能覆盖约 99% 的样本。

4) chosen 明显比 rejected 长（本数据集里 accept 是 teacher 生成的完整压缩 CoT，中位数约
   396 token；reject 是本地模型的原始生成，中位数约 147 token，只有 ~14% 的样本 reject 比
   accept 长）。标准 DPO loss 是对序列 log-prob 求和而非取平均，这种系统性长度差会混入一个
   "更长 = 更容易被偏好" 的虚假信号。缓解手段：
     - `--rpo_alpha > 0` 时，会把 `loss_type` 从默认 `["sigmoid"]` 扩展成
       `["sigmoid", "sft"]` 并配 `loss_weights=[1.0, rpo_alpha]`——这是当前 trl 版本里
       `rpo_alpha`（Llama-3/RPO 论文技巧）的等价写法：给 chosen 额外加一个交叉熵 NLL 项，防止
       chosen 的似然在训练中被无意压低。推荐从 --rpo_alpha 1.0 开始试。
     - 也可以直接 `--loss_type sigmoid_norm`（对 chosen/rejected 都做长度归一化后再比较），
       对长度失衡更不敏感，可与 --rpo_alpha 二选一或都关掉对比着跑。

5) 4 x H100 80G 环境下，多卡直接用 torchrun 拉起本脚本即可（HF Trainer + accelerate 自动处理
   DDP，不需要像 rl/gen_dpo_data.py 那样手写 torch.distributed 分片逻辑）。

6) --use_auxdpo：基于 ICLR 2026 论文 "Why DPO is a Misspecified Estimator and How to Fix It"
   （Gopalan, Chowdhury, Banerjee, arXiv:2510.20413）提出的 AuxDPO 思路的一个工程近似实现。
   论文的核心诊断——DPO 本质是把"真实偏好背后的 reward"往参数化策略能表达的、低维 implicit
   reward manifold 上做一次"按 pairwise 数据频率加权"的投影，一旦真实 reward 不在这个 manifold
   里（几乎总是如此），投影结果就依赖于训练数据里各种 pair 出现的相对频率，会出现"偏好顺序被
   反转""整体 reward 不升反降"等病态——跟本文件"第二个教训"里复盘的现象是同一件事的两种描述：
   我们的 chosen/rejected 长度系统性失衡，是训练数据分布里的一种"频率偏斜"，而 8B LoRA 策略网络
   显然不是 tabular（能表达任意 (prompt,response) -> 分数的映射），所以标准 DPO 的 sigmoid loss
   在"任务正确性"这个真实但难以被参数化策略直接表达的 reward 方向上收敛得不够，反而顺着"长度"这个
   容易被表达、又恰好和训练数据分布强相关的方向走，导致模型学会"啰嗦不收尾"而不是"回答对"。
   论文给出的修复（AuxDPO，见 Sec 4.2 / Appendix B.2）是给每条 pair 引入一个可训练的辅助偏移量
   δ，加进 DPO 的 margin 里，且约束 δ 落在参考策略梯度矩阵 A_θ0 的零空间（nullspace）内——
   直觉是：让 δ 专门去吸收那些"θ 这个参数化策略天生表达不了、但数据里又确实存在"的偏好残差，
   不去和 θ 的梯度方向抢功，从而让 θ 的更新更纯粹地对齐真实 reward，而不是被投影误差带偏。
   本文件的实现（见 AuxDPOTrainer）是这个思路的一个可落地简化版，而不是论文公式的逐字翻译：
     - 论文的零空间约束是通过 ||A_θ0,B δ_B||² 这个惩罚项实现的，A_θ0 是"参考模型 log-prob 对
       全部可训 参数的梯度"矩阵——对我们这个 8B + LoRA 的设置，LoRA 可训参数量级是千万级，逐样本
       算这个梯度、再做零空间投影/惩罚，计算量在常规训练脚本里不现实（论文自己的 8B 全量微调实验
       用的是 8x GPU、约 120GiB/replica 的显存预算，且没有公开可复用的 trl 集成代码）。
     - 这里改用"tanh 硬 cap + L2 收缩 + 独立小学习率"三件套来近似同一个目的：δ 用
       `auxdpo_delta_cap * tanh(raw_param)` 限幅（raw_param 是可训练标量），保证它不可能单独把
       margin 撑到任意大从而让 θ 完全不用学；配合 `auxdpo_l2` 对 δ² 做收缩，进一步压低 δ 能"白嫖"
       掉的 loss；δ 的 raw_param 用独立的、比主学习率小的 --auxdpo_lr 更新（对应论文 Table 3 里的
       aux_lr）。这不是真正的零空间投影，只是一个更粗糙但计算上零成本的"别让 δ 抢主角"的约束，
       所以只应把它当作"给 θ 的梯度信号松绑、减少长度等虚假信号干扰"的辅助手段，不要指望它能像
       论文里那样精确地把解推到两阶段 RLHF 的最优点。
     - δ 是"每条训练样本一个可训练标量"（只存 chosen-rejected 的差值，因为 margin 里只用得到这个
       差值，比论文里 chosen/rejected 各一个标量的参数化省一半参数，数值上等价），用数据集里的行号
       （aux_row_id，训练集内 0..N-1，构造时分配，与 rl/filter_process_rl_data.py 产出的原始 idx
       字段无关）索引，存在 AuxDPOTrainer.aux_delta_raw 里，不是模型的一部分——不参与 PEFT
       adapter 的保存/加载，训练结束后随 trainer 一起丢弃即可，不影响 checkpoint 目录结构。
     - 已知限制（用之前先读一遍）：
       (a) 只接了本文件实际会产出的 loss_type 组合：base ∈ {"sigmoid","sigmoid_norm"}
           （对应 --length_normalize）+ 可选的尾部 "sft" 项（对应 --rpo_alpha），trl 其它
           loss_type（hinge/ipo/aot/...）没有接入 δ，若组合出这些会直接报错，不会静默退化成
           普通 DPO。
       (b) aux_delta_raw 这张表按"训练集内行号"索引，且没有做多卡同步：多卡训练下每个样本只会被
           某一张卡的 DistributedSampler 分片处理到，只要 --num_train_epochs <= 1（本文件默认
           就是 1），每个行号在整个训练过程中只被一张卡碰一次，天然不存在别的卡需要它的一致性
           问题；但如果设成多个 epoch 又是多卡训练，HF 默认的 DistributedSampler 每个 epoch 会
           重新洗牌分片，同一行号在不同 epoch 可能落到不同卡上，而各卡的 aux_delta_raw 互不同步，
           会导致该样本的 δ 在换卡后"从 0 重新学"，语义打折但不会报错——建议多 epoch 训练时只用
           单卡，或者接受这个近似。
       (c) 不支持 DeepSpeed/FSDP（aux_delta_raw 不在 model 里，没有走 accelerator 的模型分片
           逻辑），只在本文件已有的 torchrun + 默认 accelerate DDP 场景下测试过。
       (d) 训练日志里额外多几个 `aux/*` 指标：`aux/delta_mean` `aux/delta_abs_mean`
           （δ 的均值/绝对值均值，衡量 δ 实际被用到了多大）、`aux/reg_loss`（δ 的 L2 惩罚项）、
           `aux/reward_accuracies_no_aux`（只用 β·Δlogratio、不含 δ 时的 pairwise 胜率——这是
           判断"θ 是否真的在学、还是全靠 δ 兜底"的关键诊断：如果这个值明显低于常规
           `rewards/accuracies`、且 `aux/delta_abs_mean` 一直逼近 `auxdpo_delta_cap`，说明 δ 在
           抢 θ 的活，应该调小 --auxdpo_delta_cap 或调大 --auxdpo_l2）。跟已有的
           GenerationDegenerationCallback（真实 generate 的 has_think/acc）配合看，不要只看
           这些 teacher-forcing 指标。
     - 跟已有缓解手段的关系：不是互斥，是同一个问题（DPO 在 misspecified 情形下对训练数据分布
       敏感）的不同应对角度。--max_chosen_reject_len_ratio 是从"数据"下手（丢弃长度失衡最极端的
       pair，减少虚假频率信号本身）；--length_normalize / --rpo_alpha 是从"loss 形式"下手（让
       loss 本身对长度更不敏感、给 chosen 兜底似然）；--use_auxdpo 是从"给模型一个逃生舱"下手
       （给参数化策略表达不了的残差一个专门的出口，减少它污染 θ 的梯度）。三者可以同时开，建议先
       单独跑一次 --use_auxdpo（配合已有的 --rpo_alpha/--length_normalize 默认值），用
       --gen_eval_steps 的真实 generate 指标和不开 --use_auxdpo 的对照跑对比，而不是直接叠加
       一堆开关再去猜是哪个起了作用。

===================== 用法示例 =====================
  conda activate tablellama-fa

  # 0) dry-run：只做数据加载 + 长度过滤统计，不加载模型（几秒钟出结果，改参数前先跑一下）
  python rl/train_dpo.py --output_dir result_dpo_checkpoint-350 --dry_run

  # 1) 单卡试跑（小 batch，先确认能跑通、看前几十步 reward margin 是否在涨）
  python rl/train_dpo.py \\
      --output_dir result_dpo_checkpoint-350 \\
      --bf16 True \\
      --per_device_train_batch_size 2 --gradient_accumulation_steps 8 \\
      --learning_rate 5e-6 --num_train_epochs 1 \\
      --eval_ratio 0.05 --eval_steps 100 \\
      --rpo_alpha 1.0

  # 2) 多卡（每卡 1 进程，与 el/sft_CoT.py / rl/gen_dpo_data.py 的 torchrun 用法一致）
  torchrun --nproc_per_node=2 rl/train_dpo.py \\
      --output_dir result_dpo_checkpoint-350 \\
      --bf16 True \\
      --per_device_train_batch_size 2 --gradient_accumulation_steps 4 \\
      --learning_rate 5e-6 --num_train_epochs 1 \\
      --eval_ratio 0.05 --eval_steps 100 \\
      --rpo_alpha 1.0

  # 2.1) 在 2) 的基础上，加上针对"第二个教训"（chosen 系统性长于 rejected -> 生成时容易不收尾
  #      重复退化）的缓解手段：--length_normalize 默认已经开着；额外丢弃长度比过于极端的 pair；
  #      并且训练中定期做一次真实 generate 监控 has_think 闭合率/准确率（而不是只看 eval_loss）：
  torchrun --nproc_per_node=2 rl/train_dpo.py \\
      --output_dir result_dpo_checkpoint-350 \\
      --bf16 True \\
      --per_device_train_batch_size 2 --gradient_accumulation_steps 4 \\
      --learning_rate 5e-6 --num_train_epochs 1 \\
      --eval_ratio 0.05 --eval_steps 100 \\
      --rpo_alpha 1.0 \\
      --max_chosen_reject_len_ratio 6.0 \\
      --gen_eval_steps 50 --gen_eval_num_samples 32 --gen_eval_batch_size 8 \\
      --gen_eval_max_new_tokens 768 \\
      --save_best_by_gen_eval True --save_total_limit 4
  # 注：--save_best_by_gen_eval True 时，按 gen_eval score=acc+has_think 保留 top-K 个
  # checkpoint-{step}（K=--save_total_limit；score 相同时更小 step 优先），不再按“最近 N 个”轮转，也不再只覆盖一个
  # checkpoint-best-gen-eval。排行榜写在 {output_dir}/gen_eval_topk.json。
  # 注：--max_chosen_reject_len_ratio 该设多少，先跑一次不带这个参数的 --dry_run 看日志里打印的
  # chosen/rejected token-len ratio 分布（本数据集实测 p50=2.57, p90=5.81, p99=9.48, max=13.87），
  # 只想切掉最极端的长尾就取接近 p90~p99 的值（如 6~10），设成接近 p50 的 3.0 会砍掉 ~44% 的数据，
  # 过于激进。

  # 2.2) 在 2) 的基础上，加上 AuxDPO（ICLR 2026, arXiv:2510.20413）思路的近似实现（见文件头第 6
  #      点），单卡或多卡 + --num_train_epochs 1（默认值）时都可以用；重点看训练日志里的
  #      aux/reward_accuracies_no_aux 是否明显低于 rewards/accuracies（低太多说明 δ 在兜底，
  #      调小 --auxdpo_delta_cap 或调大 --auxdpo_l2）：
  torchrun --nproc_per_node=2 rl/train_dpo.py \\
      --output_dir result_dpo_checkpoint-350 \\
      --bf16 True \\
      --per_device_train_batch_size 2 --gradient_accumulation_steps 4 \\
      --learning_rate 5e-6 --num_train_epochs 1 \\
      --eval_ratio 0.05 --eval_steps 100 \\
      --rpo_alpha 1.0 \\
      --use_auxdpo True --auxdpo_delta_cap 1.0 --auxdpo_l2 0.01 --auxdpo_lr 5e-3 \\
      --gen_eval_steps 50 --gen_eval_num_samples 32 --gen_eval_batch_size 8 \\
      --gen_eval_max_new_tokens 768

  # 3) 断点续训
  python rl/train_dpo.py --output_dir result_dpo_checkpoint-350 --resume_from_checkpoint result_dpo_checkpoint-350/checkpoint-200 ...

训练产物：--output_dir 下是一个可以像 checkpoint-350 一样直接被
el/eval_CoT_checkpoint.py / rl/gen_dpo_data.py 的 --model 参数加载的 LoRA adapter 目录
（同一套 base_model_name_or_path + tokenizer），建议训练后用同一套生成式评测脚本在一个
干净的、未参与 SFT/RL 训练的 eval 集上重新跑一遍准确率，对比 DPO 前后效果，不要只看训练 loss。
"""

from __future__ import annotations

import json
import logging
import random
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import transformers
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState

from trl import DPOConfig, DPOTrainer
from trl.trainer.dpo_trainer import DataCollatorForPreference
from trl.trainer.utils import selective_log_softmax

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# 复用 el/eval_CoT_checkpoint.py 同款的 CoT 解析/评分逻辑，让训练中的退化监控
# （GenerationDegenerationCallback）跟线下生成式评测用的是同一套 has_think / 抽取答案规则，
# 结果才可比。
from data_aug.augment_ent_link_thinking import build_prompt as build_cot_prompt  # noqa: E402
from el.sft import (  # noqa: E402
    _apply_cot_candidate_constraint,
    _apply_eval_table_compression,
    _has_think_tags,
    _normalize_description_marker,
    _normalize_pred_text,
)

# ===== 曾经的错误结论，留在这里作为教训 =====
# 之前这里以为 trl 的 "Mismatch between tokenized prompt and the start of tokenized
# prompt+chosen/rejected" 只是刷屏噪音，于是把 dpo_trainer 的 logger 调到 ERROR 级别整个消音。
# 实际排查（result_dpo_checkpoint-350 checkpoint-150 生成式评测：CoT-SFT 0.7822 -> DPO 0.6389，
# 且 900 条里 has_think=0）发现这不是噪音，是真实的数据污染：
#   prompt 固定以 "### Response:" 结尾（末尾是纯 ":" token），completion 固定以 "<think>\n" 开头。
#   trl.DPOTrainer._prepare_dataset 的 tokenize_fn 是"prompt 单独 tokenize 拿 prompt_ids"，
#   再"prompt+chosen 拼成字符串一起 tokenize"，然后 chosen_ids = prompt_chosen_ids[len(prompt_ids):]。
#   BPE 会把结尾的 ":" 和开头的 "<" 合并成同一个 token（实测 id 32352 "：<"），这个合并 token的
#   index < len(prompt_ids)，于是被算进了 prompt 里；chosen_ids 实际是从 "think>\n..." 开始的
#   ——"<" 这个字符从头到尾就没有出现在真正喂给模型的 chosen_input_ids / rejected_input_ids 里。
#   跟 el/sft.py 的 preprocess() 不是一回事：SFT 是对拼接后的整段字符串只 tokenize 一次得到
#   input_ids（"<" 实际存在于输入里，只是那一个边界 token 的 label 被 mask 掉，没损失梯度），
#   DPO 这里是拿两次分别 tokenize 的结果拼 id 序列，"<" 直接从输入序列里被物理删除。配合
#   --rpo_alpha 1.0（给 chosen 加显式交叉熵 NLL），相当于用监督信号反复告诉模型"### Response:
#   后面直接接 think，不需要 <"，900/900 复现，不是随机噪声。
#
# 修复方式见 build_dpo_records()：把两边 completion 共享的 "<think>\n" 前缀挪进 prompt 里，
# 让 prompt/chosen/rejected 的边界落在 "\n" + 字母 上（实测不会触发 BPE 跨边界合并，
# tokenize(prompt) 一定是 tokenize(prompt+chosen) 的严格前缀）。修复后不应该再触发这条警告，
# 所以不再消音——如果之后换了别的数据格式又触发了，就是新的边界问题，需要看到。
#
# ===== 第二个教训（result_dpo_checkpoint-350 checkpoint-400，边界 bug 修复后复测）=====
# 用 el/eval_CoT_checkpoint.py 在同一个 900 条验证集上对比：
#   CoT-SFT checkpoint-350（DPO 前）  : acc=0.7822, has_think(闭合)=800/900(88.9%)，
#       其中 has_think 子集 acc=0.851，no_think(未闭合/跑飞) 子集 acc=0.230。
#   DPO   checkpoint-400（本 rpo_alpha=1.0 跑）: acc=0.6378, has_think=589/900(65.4%)，
#       has_think 子集 acc=0.922（比 SFT 还高！），no_think 子集 acc=0.100（几乎是瞎猜，
#       因为 <think> 一直不闭合时，has_think=False 分支会 fallback 到对整段原文做正则抓取，
#       抓到的第一个 "<...>" 模式往往就是字面 "<think>" 本身，跟"模型只吐了一个 <think>"
#       是同一个现象，但根因是"生成到 max_new_tokens=768 还没写出 </think>"，不是真的只
#       生成了 4 个字符——raw_prediction 实际很长，往往是同一句话/同一个短语循环重复到
#       截断（重复退化 / repetition loop），而不是"提前停止"）。
# 即：DPO 让"能正常收尾"的推理质量变好了（85%→92%），但让模型"跑飞不收尾"的比例从
# 11%飙到了35%，这才是 overall accuracy 掉的主因。根因跟长度失衡（见上面第 4 点）是同一
# 件事的另一面：DPO 的 sigmoid pairwise loss 对序列 log-prob 求和、不做长度归一化，chosen
# 系统性比 rejected 长（本次过滤后训练集 chosen 中位数约 1565 字符 vs. 结构完整的 rejected
# 中位数约 577 字符），天然带有"越长越容易被偏好"的梯度；配合 rpo_alpha 的 sft 交叉熵项
# 进一步固化"抬高这些长 chosen 序列的似然"，训练日志里 rewards/accuracies 在前 50~100 step
# 内就已经饱和到 ~1.0、margins 从 0.6 一路涨到 5.8，之后继续训练大概率只是在已经饱和的方向上
# 继续加码，容易发生"reward over-optimization"式的长度/风格漂移。同时训练数据里真正代表
# "重复不收尾"这种失败模式的 reject（parse 不出 <think>...</think> 结构）只占过滤后训练集的
# 9.5%（700/7358），且是用 --max_new_tokens=512 生成的（比线上评测的 768 还短），负样本对这个
# 具体失败模式的覆盖 + 强度都不够，模型没被充分教会"必须在预算内收尾"这件事。
# 缓解手段见 TrainingArguments.length_normalize（把 pairwise loss 换成对长度归一化的
# 'sigmoid_norm'，直接消掉"越长越优"的梯度分量，比 rpo_alpha 更对症）、DataArguments.
# max_chosen_reject_len_ratio（训练前按 token 长度比过滤/丢弃长度失衡过于极端的 pair）、
# 以及 GenerationDegenerationCallback（训练中定期用跟线上评测一致的 max_new_tokens 做一次
# 真实 generate + has_think 闭合率/准确率统计，而不是只看 DPOTrainer 内部的 teacher-forcing
# eval_loss/rewards——那些指标从很早期就饱和到 ~1.0，完全暴露不出这个退化问题）。
DEFAULT_THINK_PREFIX = "<think>\n"

DEFAULT_ADAPTER = "result_CoT_filtered_compressed_data_lora/checkpoint-350"
DEFAULT_DATA_PATH = "/DATA1/khli/t&m/rl_checkpoint-350_10000_filtered.jsonl"
# prompt+completion 拼接后的 token 数上限：见文件头第 3 点，实测能覆盖该数据集约 99% 的样本。
DEFAULT_MAX_LENGTH = 2560


@dataclass
class ModelArguments:
    adapter_name_or_path: str = field(
        default=DEFAULT_ADAPTER,
        metadata={
            "help": "要继续训练的 LoRA checkpoint 目录（相对路径相对项目根），默认 "
            f"{DEFAULT_ADAPTER}。DPO 会在这个 adapter 上继续训练，同时把它训练开始时刻的 "
            "权重快照当作 reference model（见文件头说明第 1 点）。"
        },
    )
    base_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "base 模型名/路径，缺省从 --adapter_name_or_path 目录下 "
            "adapter_config.json 里的 base_model_name_or_path 读取（通常是 "
            "meta-llama/Llama-3.1-8B）。"
        },
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default=DEFAULT_DATA_PATH,
        metadata={
            "help": "rl/filter_process_rl_data.py 输出的 DPO pair jsonl，每行含 "
            "prompt/accept/reject/gold 字段（脚本内部把 accept/reject 重命名为 "
            "trl 约定的 chosen/rejected）。"
        },
    )
    eval_data_path: Optional[str] = field(
        default=None,
        metadata={"help": "可选：独立的同格式 eval jsonl。缺省时用 --eval_ratio 从 data_path 里切一份。"},
    )
    eval_ratio: float = field(
        default=0.0,
        metadata={
            "help": "缺省 0（不切 eval，只训练）。>0 时按 --seed 打散 data_path 后切出这个比例做 "
            "eval（仅用于监控 eval_loss / rewards/accuracies 等指标，衡量的是训练数据同分布下的 "
            "held-out 表现，不能替代用 el/eval_CoT_checkpoint.py 做的生成式准确率评测）。"
        },
    )
    debug_max_examples: Optional[int] = field(
        default=None,
        metadata={"help": "调试用：只取过滤后的前 N 条做 smoke test，不建议在正式训练时设置。"},
    )
    dry_run: bool = field(
        default=False,
        metadata={"help": "只加载数据、跑长度过滤统计并打印，不加载模型、不训练。"},
    )
    max_chosen_reject_len_ratio: Optional[float] = field(
        default=None,
        metadata={
            "help": "缺省 None（不过滤，只打印统计）。设置为正数时，丢弃 "
            "token_len(chosen)/token_len(rejected) 超过该比例的 pair。见文件头'第二个教训'："
            "chosen 系统性远长于 rejected 天然带来'越长越优'的梯度信号，是 DPO 后模型倾向于 "
            "生成更长、更容易在 max_new_tokens 内不收尾（重复退化）的一个主要根因，过滤掉长度 "
            "失衡最极端的一小部分 pair 能直接减少这个混淆信号（配合 --length_normalize 使用，"
            "二者互补，不是二选一）。这个比例设多少合适因数据集而异：先用 --dry_run 不带这个参数 "
            "跑一次，看日志打印的 chosen/rejected token-len ratio 分布（p50/p90/p99/max），只想切 "
            "掉极端长尾就取接近 p90~p99 的值，取接近 p50 的值会砍掉近一半数据，过于激进。"
        },
    )


@dataclass
class TrainingArguments(DPOConfig):
    use_flash_attn: bool = field(
        default=True,
        metadata={"help": "是否使用 flash_attention_2 加载 base 模型（与 SFT 阶段保持一致）。"},
    )
    rpo_alpha: float = field(
        default=0.0,
        metadata={
            "help": "0 表示关闭（默认）。>0 时把 loss_type 扩展为 ['sigmoid', 'sft']（若尚未包含 "
            "'sft'）并设置对应的 loss_weights=[..., rpo_alpha]，等价于 RPO/Llama-3 论文里的 "
            "rpo_alpha：给 chosen 额外加一个交叉熵 NLL 项，防止 chosen 似然被训练压低。见文件头 "
            "说明第 4 点（本数据集 chosen 系统性比 rejected 长，建议开启，先试 1.0）。注意这个 "
            "选项只解决'chosen 似然被意外压低'，不解决 pairwise sigmoid 项本身对长度的敏感性，"
            "两者要一起看，见 --length_normalize。"
        },
    )
    length_normalize: bool = field(
        default=True,
        metadata={
            "help": "默认开启。把 loss_type 里的基础 pairwise 项从 'sigmoid' 换成 trl 内置的 "
            "'sigmoid_norm'（chosen/rejected 按各自 completion token 数取平均 log-prob 后再比较，"
            "而不是像 'sigmoid' 那样直接比较 token 数求和后的 log-prob）。见文件头'第二个教训'："
            "本数据集 chosen 系统性远长于 rejected，标准 'sigmoid' loss 会带一个'越长越容易被 "
            "偏好'的虚假梯度信号，'sigmoid_norm' 是 trl 里对症的长度归一化实现（跟 rpo_alpha 的 "
            "sft 项不冲突，可以同时开）。如果要跟旧实验（未归一化）对比效果，显式传 "
            "--length_normalize False 关闭。"
        },
    )
    gen_eval_steps: int = field(
        default=0,
        metadata={
            "help": "默认 0（关闭）。>0 时每训练这么多 step，就用当前 policy 在 eval_dataset（若 "
            "没有则用训练集前若干条）上抽样做一次真实 generate（贪心解码，长度用 "
            "--gen_eval_max_new_tokens，跟线下 el/eval_CoT_checkpoint.py 的默认口径一致），统计 "
            "has_think 闭合率和抽取式准确率并写进 trainer 日志（键名 gen_eval/*）。见文件头'第二 "
            "个教训'：DPOTrainer 内部的 eval_loss / rewards/accuracies 是 teacher-forcing 指标，"
            "从训练很早期就饱和到 ~1.0，完全暴露不出'生成时是否会跑飞不收尾'这个问题，必须靠真实 "
            "generate 才能看到。只在 world process 0 上跑，且用 try/except 包裹，失败不影响训练；"
            "但每次会额外花 --gen_eval_num_samples/--gen_eval_batch_size 个 batch、每 batch 最多 "
            "--gen_eval_max_new_tokens 步的 decode 时间，不建议设得太频繁（可以先设成跟 "
            "--eval_steps 一样或更大的倍数）。多卡时这段只在 rank 0 上跑，其它 rank 会在同一 step "
            "继续往下走到需要所有 rank 一起参与的分布式 evaluate()/save 等 collective 操作并卡在那 "
            "里等 rank 0；如果这段跑得比 NCCL 默认的 collective 超时（常见约 30 分钟）还久，其它 "
            "rank 会直接超时报错、训练崩掉——所以 --gen_eval_num_samples/--gen_eval_batch_size/"
            "--gen_eval_max_new_tokens 的组合不要设得太大，且优先靠调大 --gen_eval_batch_size（批量 "
            "generate）而不是调小 --gen_eval_num_samples 来控制耗时。"
        },
    )
    gen_eval_data_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "可选：独立的生成式评测集（mammotab 风格 JSON list，字段含 "
            "instruction/input_seg/question/output，与 el/eval_CoT_checkpoint.py --data_path 同格式）。"
            "设置后 gen_eval 不再从 DPO eval/train pair 里抽样本，而是用该文件前 "
            "--gen_eval_num_samples 条（固定顺序，与线下 eval_data_limit 口径一致），"
            "prompt 用 CoT build_prompt，不预填 <think>，保证和线下 mam_val 评测可比。"
        },
    )
    gen_eval_num_samples: int = field(
        default=32,
        metadata={
            "help": "每次 --gen_eval_steps 触发时做真实 generate 的样本数（越大越准但越慢；"
            "实际耗时约等于 ceil(num_samples / --gen_eval_batch_size) 个 batch 的 generate 时间，"
            "不是线性于 num_samples 本身，调大 --gen_eval_batch_size 通常比调小这个参数更划算）。"
            "若设置了 --gen_eval_data_path，表示取该文件的前 N 条（不 shuffle）。"
        },
    )
    gen_eval_max_new_tokens: int = field(
        default=768,
        metadata={
            "help": "退化监控 generate 时的 max_new_tokens，默认 768，跟 "
            "el/eval_CoT_checkpoint.py 评测脚本的默认值保持一致，这样 has_think 闭合率才能跟线下 "
            "评测结果直接对比，而不是被一个更宽松/更严格的长度预算掩盖问题。"
        },
    )
    gen_eval_batch_size: int = field(
        default=8,
        metadata={
            "help": "退化监控一次 generate() 调用喂多少条样本（左 padding 后一起生成，而不是像早期"
            "版本那样一条一条串行 generate）。踩过的坑：--gen_eval_num_samples=32、batch_size=1 "
            "串行跑一次要好几分钟，且中间没有任何进度日志——从日志上看跟训练真的卡住长得一模一样，"
            "还会因为只在 rank 0 上跑、其余 rank 早已进入需要所有 rank 一起参与的分布式 evaluate() "
            "collective 而被晾在那等，如果这段跑得比 NCCL 默认的 collective 超时（常见约 30 分钟）"
            "还久，会直接导致其它 rank 超时报错、训练崩掉。批量 generate 能把这段时间压缩到接近 "
            "1/batch_size，同时每个 batch 结束都会打一行进度日志，不会再看起来像卡死。"
        },
    )
    save_best_by_gen_eval: bool = field(
        default=False,
        metadata={
            "help": "默认 False。True 时按 gen_eval 分数 score=acc+has_think 保留 top-K 个 "
            "checkpoint-{step}（K=--save_total_limit，默认 3），score 相同时保留 step 更小的；"
            "会关闭 HuggingFace Trainer 自带的“按最近 N 个”轮转，改由本 callback 在每次 "
            "gen_eval / on_save 后按分数删盘。每次更新写 {output_dir}/gen_eval_topk.json。"
            "不删训练中的 ref adapter。需要 --gen_eval_steps > 0。多卡时 rank0 存盘/"
            "删盘前后所有 rank 会 barrier，避免其它 rank 先跑进下一轮 collective。"
        },
    )
    use_auxdpo: bool = field(
        default=False,
        metadata={
            "help": "默认 False。True 时用 AuxDPOTrainer 替代 DPOTrainer，给每条训练 pair 引入一个 "
            "可训练的辅助偏移量 δ 加进 DPO 的 margin 里，思路来自 ICLR 2026 论文《Why DPO is a "
            "Misspecified Estimator and How to Fix It》(arXiv:2510.20413) 的 AuxDPO 算法，本文件是 "
            "该思路针对 8B+LoRA 场景的一个工程近似实现（不是论文公式的逐字翻译），细节、动机与已知 "
            "限制见文件头说明第 6 点，务必先读一遍再开。只支持 --length_normalize 产出的 "
            "'sigmoid'/'sigmoid_norm' 基础 loss（可选叠加 --rpo_alpha 的 'sft' 项），其它 trl "
            "loss_type 组合下开启会直接报错。"
        },
    )
    auxdpo_delta_cap: float = field(
        default=1.0,
        metadata={
            "help": "AuxDPO 辅助偏移量 δ 的硬限幅（δ = auxdpo_delta_cap * tanh(raw)），单位与 "
            "beta * logratio 的 margin 相同。越大 δ 能吸收的偏好信号越强，但也越容易让 θ 偷懒不学 "
            "（见 aux/reward_accuracies_no_aux 诊断指标）。仅在 --use_auxdpo True 时生效。"
        },
    )
    auxdpo_l2: float = field(
        default=0.01,
        metadata={
            "help": "对 δ² 的 L2 收缩系数，作用类似论文里约束 δ 落在零空间的惩罚项的粗糙近似：越大越 "
            "抑制 δ 被无成本地用来'解释'偏好、把更多梯度信号逼回 θ；越小则 δ 越自由（更容易吸收长度 "
            "等虚假信号，但也可能反过来让 θ 学得更纯粹——需要用 aux/reward_accuracies_no_aux 和真实 "
            "generate 结果一起判断，不能只看这一个数字）。仅在 --use_auxdpo True 时生效。"
        },
    )
    auxdpo_lr: float = field(
        default=5e-3,
        metadata={
            "help": "δ 的 raw 参数（tanh 之前）用独立的这个学习率更新（对应论文 Table 3 的 "
            "aux_lr），与 --learning_rate（θ 的学习率）解耦，通过 Trainer.create_optimizer 里 "
            "额外的 param group 实现。仅在 --use_auxdpo True 时生效。"
        },
    )

    # ---- 下面几个字段只是覆盖 DPOConfig 的默认值，字段本身继承自 DPOConfig/TrainingArguments ----
    max_length: Optional[int] = field(
        default=DEFAULT_MAX_LENGTH,
        metadata={"help": "prompt+completion 拼接后的 token 数上限，见文件头说明第 3 点。"},
    )
    learning_rate: float = field(
        default=5e-6,
        metadata={"help": "LoRA + DPO 常用学习率比全量微调 DPO（trl 默认 1e-6）高一个量级。"},
    )
    num_train_epochs: float = field(
        default=1.0,
        metadata={"help": "DPO 很容易 1~2 epoch 内就 reward 饱和/过拟合，默认只跑 1 epoch。"},
    )
    per_device_train_batch_size: int = field(default=2)
    gradient_accumulation_steps: int = field(default=8)
    gradient_checkpointing_kwargs: Optional[Dict[str, Any]] = field(
        default_factory=lambda: {"use_reentrant": False}
    )
    logging_steps: int = field(default=10)
    save_strategy: str = field(default="steps")
    save_steps: int = field(default=50)
    save_total_limit: Optional[int] = field(default=3)
    eval_strategy: str = field(default="no")
    ddp_find_unused_parameters: Optional[bool] = field(default=False)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _combined_token_len(tokenizer, prompt: str, completion: str) -> int:
    """粗略估计 tokenize(prompt + completion) 的 token 数（+1 给自动补的 eos 留余量）。

    与 trl 内部 `_prepare_dataset` 里 `prompt_ids`/`chosen_ids`/`rejected_ids` 的算法不完全
    一致（trl 是分别 tokenize prompt 和 prompt+completion 再切片），但差异在个别 token 级别，
    不影响用来做"是否超长该丢弃"的过滤判断。
    """
    return len(tokenizer(prompt + completion).input_ids) + 1


def _split_off_shared_think_prefix(
    chosen: str, rejected: str, think_prefix: str
) -> Optional[tuple]:
    """若 chosen/rejected 都以 think_prefix（如 "<think>\\n"）开头，把这段公共前缀切掉，
    返回 (chosen_without_prefix, rejected_without_prefix)；否则返回 None（调用方负责丢弃该样本，
    见 build_dpo_records 里的 dropped_missing_think_prefix 统计，实测占比 ~1.6%，多是 reject
    生成不完整/没走到 <think> 的坏样本，直接丢弃比冒着复现边界合并 bug 风险更划算）。
    """
    if chosen.startswith(think_prefix) and rejected.startswith(think_prefix):
        return chosen[len(think_prefix) :], rejected[len(think_prefix) :]
    return None


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))
    return s[idx]


def build_dpo_records(
    raw_records: List[Dict[str, Any]],
    tokenizer,
    max_length: int,
    *,
    dataset_name: str,
    think_prefix: str = DEFAULT_THINK_PREFIX,
    max_chosen_reject_len_ratio: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """把 {prompt, accept, reject, ...} 转成 trl 约定的 {prompt, chosen, rejected}，
    并按 --max_length 显式丢弃超长样本（原因见文件头说明第 3 点），打印漏斗统计。

    关键点（见文件头"曾经的错误结论"说明）：accept/reject 固定以 "<think>\\n" 开头，如果让这段
    前缀留在 chosen/rejected 里、prompt 固定以 "### Response:" 结尾，trl 内部"prompt 单独
    tokenize" + "prompt+chosen 拼接后再 tokenize"两次结果对不齐（BPE 把结尾 ":" 和开头 "<" 合并
    成同一个 token），会导致 chosen_ids/rejected_ids 真实序列里丢掉开头的 "<" 字符（900/900 复现
    过，见 eval_checkpoint-150.log 的 has_think=0/900）。这里把该共享前缀显式挪进 prompt 里，
    让 prompt 与 prompt+chosen 的边界落在 "\\n" + 字母上，两种 tokenize 方式结果严格一致，
    彻底消除这类边界合并 bug（而不是指望它"恰好不触发"）。
    """
    kept: List[Dict[str, Any]] = []
    dropped_missing_field = 0
    dropped_missing_think_prefix = 0
    dropped_too_long = 0
    dropped_len_ratio = 0
    len_ratios: List[float] = []
    for rec in raw_records:
        prompt = rec.get("prompt")
        chosen = rec.get("accept")
        rejected = rec.get("reject")
        if not prompt or not chosen or not rejected:
            dropped_missing_field += 1
            continue
        split = _split_off_shared_think_prefix(chosen, rejected, think_prefix)
        if split is None:
            dropped_missing_think_prefix += 1
            continue
        chosen, rejected = split
        prompt = prompt + think_prefix
        # 只算 completion 本身（不含 prompt）的 token 数，用来衡量 chosen/rejected 的长度失衡，
        # 见文件头"第二个教训"：这个比例越大，标准 'sigmoid' pairwise loss 里"越长越优"的虚假
        # 梯度信号越强，是 DPO 后模型倾向于生成更长/更容易不收尾的主要根因之一。
        chosen_completion_len = len(tokenizer(chosen, add_special_tokens=False).input_ids)
        rejected_completion_len = len(tokenizer(rejected, add_special_tokens=False).input_ids)
        len_ratio = chosen_completion_len / max(1, rejected_completion_len)
        len_ratios.append(len_ratio)
        if max_chosen_reject_len_ratio is not None and len_ratio > max_chosen_reject_len_ratio:
            dropped_len_ratio += 1
            continue
        chosen_len = _combined_token_len(tokenizer, prompt, chosen)
        rejected_len = _combined_token_len(tokenizer, prompt, rejected)
        if max(chosen_len, rejected_len) > max_length:
            dropped_too_long += 1
            continue
        kept.append(
            {
                "idx": rec.get("idx"),
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "gold": rec.get("gold", ""),
            }
        )

    total = len(raw_records)
    dropped_total = (
        dropped_missing_field + dropped_missing_think_prefix + dropped_too_long + dropped_len_ratio
    )
    ratio = (dropped_total / total) if total else 0.0
    logging.warning(
        f"[{dataset_name}] kept {len(kept)}/{total}; dropped {dropped_total}/{total} ({ratio:.2%}) "
        f"(missing_field={dropped_missing_field}, "
        f"missing_think_prefix={dropped_missing_think_prefix}, "
        f"exceeds_max_length_{max_length}={dropped_too_long}, "
        f"exceeds_len_ratio_{max_chosen_reject_len_ratio}={dropped_len_ratio})"
    )
    if len_ratios:
        logging.warning(
            f"[{dataset_name}] chosen/rejected completion token-len ratio: "
            f"p50={_percentile(len_ratios, 0.50):.2f}, p90={_percentile(len_ratios, 0.90):.2f}, "
            f"p99={_percentile(len_ratios, 0.99):.2f}, max={max(len_ratios):.2f} "
            "(越大代表 chosen 相对 rejected 越长，>1 越多说明长度失衡信号越强，见文件头'第二个"
            "教训'；可用 --max_chosen_reject_len_ratio 过滤掉最极端的一批)。"
        )
    return kept


def _assert_no_prompt_completion_boundary_mismatch(
    records: List[Dict[str, Any]], tokenizer, *, dataset_name: str, sample_size: int = 200
) -> None:
    """在真正开始训练前抽样自检：tokenize(prompt) 必须是 tokenize(prompt+chosen/rejected) 的严格
    前缀。这正是 trl.DPOTrainer 内部 tokenize_fn 用来切 chosen_ids/rejected_ids 的假设——一旦不
    满足，被切出来的 completion 会悄悄丢字符（就是本文件头部说明的那个 bug）。build_dpo_records
    已经通过把公共的 "<think>\\n" 前缀挪进 prompt 来规避，这里再做一次实测校验，防止以后数据
    格式变了又踩同一个坑却没人发现。
    """
    if not records:
        return
    rng = random.Random(0)
    sample = rng.sample(records, k=min(sample_size, len(records)))
    n_mismatch = 0
    first_mismatch_idx = None
    for rec in sample:
        prompt_ids = tokenizer(rec["prompt"], add_special_tokens=False).input_ids
        for key in ("chosen", "rejected"):
            full_ids = tokenizer(rec["prompt"] + rec[key], add_special_tokens=False).input_ids
            if full_ids[: len(prompt_ids)] != prompt_ids:
                n_mismatch += 1
                if first_mismatch_idx is None:
                    first_mismatch_idx = rec.get("idx")
    if n_mismatch:
        raise ValueError(
            f"[{dataset_name}] prompt/completion tokenize boundary mismatch in {n_mismatch}/"
            f"{len(sample) * 2} sampled (prompt, chosen/rejected) pairs (first offending idx="
            f"{first_mismatch_idx}). This means trl will silently drop leading characters from "
            "chosen/rejected when it slices completion_ids from prompt_ids length (see file header "
            "comment). Fix the prompt/completion split (e.g. move the shared literal prefix into "
            "the prompt) before training."
        )
    logging.warning(
        f"[{dataset_name}] boundary self-check passed: {len(sample)} sampled records, no "
        "prompt/completion tokenize mismatch."
    )


def make_dpo_dataset_module(tokenizer, data_args: DataArguments, max_length: int, seed: int) -> Dict[str, Any]:
    from datasets import Dataset

    data_path = Path(data_args.data_path).expanduser().resolve()
    if not data_path.is_file():
        raise SystemExit(f"data_path not found: {data_path}")

    raw_records = load_jsonl(data_path)
    logging.warning(f"[data] loaded {len(raw_records)} raw pairs from {data_path}")
    all_records = build_dpo_records(
        raw_records,
        tokenizer,
        max_length,
        dataset_name="train_pool",
        max_chosen_reject_len_ratio=data_args.max_chosen_reject_len_ratio,
    )
    _assert_no_prompt_completion_boundary_mismatch(all_records, tokenizer, dataset_name="train_pool")

    eval_records: Optional[List[Dict[str, Any]]] = None
    if data_args.eval_data_path:
        eval_path = Path(data_args.eval_data_path).expanduser().resolve()
        if not eval_path.is_file():
            raise SystemExit(f"eval_data_path not found: {eval_path}")
        eval_raw = load_jsonl(eval_path)
        eval_records = build_dpo_records(
            eval_raw,
            tokenizer,
            max_length,
            dataset_name="eval(独立文件)",
            max_chosen_reject_len_ratio=data_args.max_chosen_reject_len_ratio,
        )
        train_records = all_records
    elif data_args.eval_ratio > 0:
        shuffled = list(all_records)
        random.Random(seed).shuffle(shuffled)
        n_eval = max(1, int(round(len(shuffled) * data_args.eval_ratio)))
        eval_records = shuffled[:n_eval]
        train_records = shuffled[n_eval:]
        logging.warning(
            f"[data] split from {data_args.data_path}: train={len(train_records)}, "
            f"eval={len(eval_records)} (eval_ratio={data_args.eval_ratio}, seed={seed}); "
            "注意这是训练数据同分布下的 held-out 切片，只用于监控，不是独立的生成式评测集。"
        )
    else:
        train_records = all_records

    if data_args.debug_max_examples:
        train_records = train_records[: data_args.debug_max_examples]
        logging.warning(f"[debug] truncated train set to {len(train_records)} examples (--debug_max_examples)")

    if not train_records:
        raise ValueError("No training records left after filtering; check --max_length / --data_path.")

    train_dataset = Dataset.from_list(train_records)
    # 仅训练集需要这一列：AuxDPOTrainer 用它索引每条 pair 专属的辅助偏移量 δ（见文件头说明第 6
    # 点）。跟 rec["idx"]（filter_process_rl_data.py 产出的原始 jsonl 行号，可能不连续/缺失）
    # 无关，这里是过滤/切分后训练集内部的稠密 0..N-1 行号，专门给 δ 表定容量、做索引用。
    train_dataset = train_dataset.add_column("aux_row_id", list(range(len(train_dataset))))
    eval_dataset = Dataset.from_list(eval_records) if eval_records else None
    return {"train_dataset": train_dataset, "eval_dataset": eval_dataset}


def load_policy_model(model_args: ModelArguments, training_args: TrainingArguments):
    adapter_dir = Path(model_args.adapter_name_or_path)
    if not adapter_dir.is_absolute():
        adapter_dir = (_PROJECT_ROOT / adapter_dir).resolve()
    adapter_config_path = adapter_dir / "adapter_config.json"
    if not adapter_config_path.is_file():
        raise SystemExit(f"adapter_config.json not found under: {adapter_dir}")
    adapter_cfg = json.loads(adapter_config_path.read_text(encoding="utf-8"))

    base_model_name = model_args.base_model_name_or_path or adapter_cfg["base_model_name_or_path"]
    logging.warning(f"[model] base model : {base_model_name}")
    logging.warning(f"[model] adapter    : {adapter_dir}")

    # tokenizer 必须从 adapter 目录加载（含训练时扩充过的 vocab / 自定义 eos "</s>"，
    # 见文件头说明第 2 点）。trl 要求 padding_side="left"。
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), use_fast=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    attn_impl = "flash_attention_2" if training_args.use_flash_attn else "eager"
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16 if training_args.bf16 else None,
        attn_implementation=attn_impl,
    )
    # Llama-3.1-8B 原生上下文 131072 远大于本任务用到的 --max_length，不需要像 el/sft_CoT.py
    # 那样再做 RoPE 线性插值缩放。
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

    logging.warning(f"[model] loading trainable LoRA adapter from {adapter_dir}")
    model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=True)
    model.config.use_cache = False
    return tokenizer, model


_QUESTION_RE = re.compile(r"### Question:\n(.*?)\n\n### Response:", flags=re.S)


def _extract_question_from_prompt(prompt: str) -> str:
    """从 build_dpo_records 产出的 prompt（"...### Question:\\n{question}\\n\\n### Response:"
    + think_prefix）里抠出原始 question 文本，喂给 el.sft._apply_cot_candidate_constraint 做
    候选实体约束抽取。跟 rl/regen_incomplete_reject.py 用的是同一个正则，保持口径一致。
    """
    m = _QUESTION_RE.search(prompt or "")
    return m.group(1).strip() if m else ""


def load_mam_gen_eval_records(path: str, num_samples: int) -> List[Dict[str, Any]]:
    """加载 mammotab JSON（与 el/eval_CoT_checkpoint.py 同格式），取前 num_samples 条，
    转成 gen_eval 需要的 {idx, prompt, gold, question}。prompt 不含预填 <think>。
    """
    data_path = Path(path).expanduser().resolve()
    if not data_path.is_file():
        raise SystemExit(f"gen_eval_data_path not found: {data_path}")
    with data_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit(f"gen_eval_data_path must be a JSON list, got {type(data)}")
    n = max(1, num_samples)
    raw_samples = data[:n]
    records: List[Dict[str, Any]] = []
    for i, ex in enumerate(raw_samples):
        aligned = _apply_eval_table_compression(ex)
        for key in ("question", "output"):
            if key in aligned:
                aligned[key] = _normalize_description_marker(aligned[key])
        records.append(
            {
                "idx": i,
                "prompt": build_cot_prompt(aligned),
                "gold": aligned.get("output") or "",
                "question": aligned.get("question") or "",
                # 与线下 eval 一致：模型自己生成 <think>...，prompt 不预填 think_prefix
                "_prompt_has_think_prefix": False,
            }
        )
    logging.warning(
        f"[gen_eval] loaded mam eval set from {data_path}: "
        f"using first {len(records)}/{len(data)} samples (num_samples={num_samples})"
    )
    return records


def _unwrap_model(model):
    """剥掉 DDP/accelerate 的 .module 包装，拿到底层 PeftModel。"""
    unwrapped = model
    while hasattr(unwrapped, "module"):
        unwrapped = unwrapped.module
    return unwrapped


class GenerationDegenerationCallback(TrainerCallback):
    """每 --gen_eval_steps 个 step，用当前 policy 做一次跟线下评测同口径的真实 generate
    （贪心解码，max_new_tokens=--gen_eval_max_new_tokens），统计 has_think 闭合率和抽取式准确率。

    动机见文件头"第二个教训"：DPOTrainer 内部的 eval_loss / rewards/accuracies 全是
    teacher-forcing 指标（给定完整 chosen/rejected 序列算似然），从训练很早期就能饱和到 ~1.0，
    完全看不出模型自由生成时会不会因为长度失衡等原因养成"啰嗦到不收尾/重复退化"的坏习惯——这个
    问题只有真的调用一次 model.generate() 才能看到（也是本文件之所以要单独加这个 callback，而不
    是只依赖 DPOTrainer 自带 eval 的原因）。

    每次 gen_eval 还会把逐条结果写到
    ``{output_dir}/gen_eval/step-{global_step}_predictions.jsonl``，字段与
    el/eval_CoT_checkpoint.py 产出的 mam_val predictions.jsonl 一致：
    idx / prompt / raw_prediction / prediction / gold / has_think / correct。

    只在 world process 0 上跑 generate；所有 rank 在 gen_eval step 结束时 barrier，避免
    rank0 还在 generate/存盘/删盘时其它 rank 先跑进下一轮 collective。整段包在 try/except
    里：这只是诊断/监控指标，任何异常都不应该打断真正的训练 loop。

    ``save_best_by_gen_eval=True`` 时，按 gen_eval score=acc+has_think 保留 top-K 个 ``checkpoint-{step}``
    （K=``save_total_limit``），score 相同时保留更小 step；会关闭 HF Trainer 的“最近 N 个”轮转。
    """

    TOPK_META_NAME = "gen_eval_topk.json"
    _CKPT_DIR_RE = re.compile(r"^checkpoint-(\d+)$")

    def __init__(
        self,
        tokenizer,
        records: List[Dict[str, Any]],
        *,
        num_samples: int,
        max_new_tokens: int,
        every_n_steps: int,
        batch_size: int = 8,
        think_prefix: str = DEFAULT_THINK_PREFIX,
        seed: int = 0,
        shuffle: bool = True,
        save_best_by_gen_eval: bool = False,
        save_total_limit: Optional[int] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.think_prefix = think_prefix
        self.every_n_steps = every_n_steps
        self.max_new_tokens = max_new_tokens
        self.batch_size = max(1, batch_size)
        self.save_best_by_gen_eval = save_best_by_gen_eval
        self.save_total_limit = save_total_limit
        # step -> gen_eval metrics（至少含 acc / has_think）；用于 top-K 轮转。
        self.step_metrics: Dict[int, Dict[str, Any]] = {}
        pool = list(records)
        if shuffle:
            random.Random(seed).shuffle(pool)
        self.samples = pool[: max(1, num_samples)]

    @staticmethod
    def _rank_score(metrics: Optional[Dict[str, Any]]) -> float:
        """top-K 排名分：acc + has_think（均在 [0,1]，合计 [0,2]）。无分数视为 -1。"""
        if metrics is None:
            return -1.0
        acc = float(metrics.get("acc") or 0.0)
        has_think = float(metrics.get("has_think") or 0.0)
        return acc + has_think

    def on_train_begin(
        self,
        args: "TrainingArguments",
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        # resume 时把已有排行榜读回来，避免后续轮转把历史高分 ckpt 误删。
        if self.save_best_by_gen_eval and state.is_world_process_zero:
            self._load_topk_meta(args.output_dir)
        return control

    def on_step_end(  # noqa: D401
        self,
        args: "TrainingArguments",
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        if (
            self.every_n_steps <= 0
            or state.global_step <= 0
            or state.global_step % self.every_n_steps != 0
            or not self.samples
        ):
            return control
        model = kwargs.get("model")
        if model is None:
            return control

        if state.is_world_process_zero:
            try:
                metrics = self._run(model, global_step=state.global_step, output_dir=args.output_dir)
                if self.save_best_by_gen_eval and metrics is not None:
                    self._record_and_maybe_save_topk(
                        model,
                        output_dir=args.output_dir,
                        global_step=state.global_step,
                        metrics=metrics,
                    )
            except Exception as exc:  # noqa: BLE001 - 诊断性监控，绝不能打断训练
                logging.warning(f"[gen_eval] skipped at step {state.global_step} due to error: {exc!r}")

        # rank0 可能还在 generate / 写/删 top-K checkpoint；其它 rank 在此等候，避免先跑进
        # 下一轮需要全员参与的 evaluate()/save collective 导致 NCCL 超时。
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        return control

    def on_save(
        self,
        args: "TrainingArguments",
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        # HF 刚写完 checkpoint-{step} 后，再按 gen_eval 分数裁剪一次（覆盖 save_steps
        # 与 gen_eval_steps 不完全对齐、或本 step 尚无分数的情况）。
        if not self.save_best_by_gen_eval:
            return control
        if (
            state.is_world_process_zero
            and self.save_total_limit is not None
            and self.save_total_limit > 0
        ):
            try:
                self._rotate_checkpoints_by_gen_eval(args.output_dir)
            except Exception as exc:  # noqa: BLE001
                logging.warning(f"[gen_eval] top-K rotate on_save failed: {exc!r}")
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        return control

    def _topk_entries(self) -> List[Dict[str, Any]]:
        """按 (score=acc+has_think desc, step asc) 排序后的排行榜条目。"""
        entries: List[Dict[str, Any]] = []
        for step, metrics in self.step_metrics.items():
            score = self._rank_score(metrics)
            entries.append(
                {
                    "global_step": step,
                    "gen_eval_score": score,
                    "gen_eval_acc": metrics["acc"],
                    "gen_eval_has_think": metrics.get("has_think"),
                    "gen_eval_n": metrics.get("n"),
                    "gen_eval_n_correct": metrics.get("n_correct"),
                    "gen_eval_n_has_think": metrics.get("n_has_think"),
                }
            )
        # score 降序；同分时 step 升序（更早的 checkpoint 优先）。
        entries.sort(key=lambda e: (-e["gen_eval_score"], e["global_step"]))
        return entries

    def _load_topk_meta(self, output_dir: str) -> None:
        meta_path = Path(output_dir) / self.TOPK_META_NAME
        if not meta_path.is_file():
            return
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            logging.warning(f"[gen_eval] failed to load {meta_path}: {exc!r}")
            return
        for entry in data.get("topk") or data.get("all") or []:
            step = int(entry["global_step"])
            self.step_metrics[step] = {
                "acc": float(entry["gen_eval_acc"]),
                "has_think": float(entry["gen_eval_has_think"] or 0.0),
                "n": entry.get("gen_eval_n"),
                "n_correct": entry.get("gen_eval_n_correct"),
                "n_has_think": entry.get("gen_eval_n_has_think"),
            }
        if self.step_metrics:
            logging.warning(
                f"[gen_eval] restored {len(self.step_metrics)} scored steps from {meta_path}"
            )

    def _write_topk_meta(self, output_dir: str) -> None:
        all_entries = self._topk_entries()
        limit = self.save_total_limit
        topk = all_entries if limit is None or limit <= 0 else all_entries[:limit]
        payload = {
            "save_total_limit": self.save_total_limit,
            "topk": topk,
            "all": all_entries,
        }
        path = Path(output_dir) / self.TOPK_META_NAME
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _qualifies_for_topk(self, score: float, global_step: int) -> bool:
        limit = self.save_total_limit
        if limit is None or limit <= 0:
            return True
        if global_step in self.step_metrics:
            return True
        if len(self.step_metrics) < limit:
            return True
        ranked = sorted(
            ((self._rank_score(m), step) for step, m in self.step_metrics.items()),
            key=lambda x: (-x[0], x[1]),
        )
        worst_score, worst_step = ranked[limit - 1]
        # 严格优于当前 top-K 末位，或同分但 step 更小（会挤掉更晚的同分 ckpt）。
        return (score, -global_step) > (worst_score, -worst_step)

    def _save_step_checkpoint(
        self,
        model,
        *,
        output_dir: str,
        global_step: int,
        metrics: Dict[str, Any],
    ) -> Path:
        """保存当前 default adapter 到 checkpoint-{step}（保留训练中的 ref）。"""
        ckpt_dir = Path(output_dir) / f"checkpoint-{global_step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        unwrapped = _unwrap_model(model)
        save_kwargs: Dict[str, Any] = {}
        peft_config = getattr(unwrapped, "peft_config", None)
        if peft_config is not None and "default" in peft_config:
            # 只落盘可训练的 default adapter；绝不能 delete_adapter("ref")，否则后续 DPO 会坏掉。
            save_kwargs["selected_adapters"] = ["default"]
        unwrapped.save_pretrained(str(ckpt_dir), **save_kwargs)
        self.tokenizer.save_pretrained(str(ckpt_dir))
        meta = {
            "global_step": global_step,
            "gen_eval_score": self._rank_score(metrics),
            "gen_eval_acc": metrics["acc"],
            "gen_eval_has_think": metrics["has_think"],
            "gen_eval_n": metrics["n"],
            "gen_eval_n_correct": metrics["n_correct"],
            "gen_eval_n_has_think": metrics["n_has_think"],
        }
        (ckpt_dir / "gen_eval_metrics.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return ckpt_dir

    def _list_numeric_checkpoints(self, output_dir: str) -> List[Tuple[int, Path]]:
        root = Path(output_dir)
        if not root.is_dir():
            return []
        out: List[Tuple[int, Path]] = []
        for p in root.iterdir():
            if not p.is_dir():
                continue
            m = self._CKPT_DIR_RE.fullmatch(p.name)
            if m:
                out.append((int(m.group(1)), p))
        return out

    def _rotate_checkpoints_by_gen_eval(self, output_dir: str) -> None:
        """只保留 gen_eval score=acc+has_think 最高的 save_total_limit 个 checkpoint-{step}。"""
        limit = self.save_total_limit
        if limit is None or limit <= 0:
            self._write_topk_meta(output_dir)
            return

        ckpts = self._list_numeric_checkpoints(output_dir)
        if len(ckpts) <= limit:
            self._write_topk_meta(output_dir)
            return

        # 有分数的按 score 降序、同分 step 升序保留；无分数的视为最差（-1.0），优先被删。
        scored: List[Tuple[float, float, float, int, Path]] = []
        for step, path in ckpts:
            metrics = self.step_metrics.get(step)
            score = self._rank_score(metrics)
            acc = float(metrics["acc"]) if metrics is not None else -1.0
            has_think = float(metrics["has_think"]) if metrics is not None else -1.0
            scored.append((score, acc, has_think, step, path))
        scored.sort(key=lambda x: (-x[0], x[3]))

        def _fmt(score: float, acc: float, has_think: float, step: int) -> str:
            if score < 0:
                return f"checkpoint-{step}(no_gen_eval)"
            return (
                f"checkpoint-{step}(score={score:.4f},acc={acc:.2%},has_think={has_think:.2%})"
            )

        pruned: List[str] = []
        for score, acc, has_think, step, path in scored[limit:]:
            shutil.rmtree(path, ignore_errors=True)
            pruned.append(_fmt(score, acc, has_think, step))

        self._write_topk_meta(output_dir)
        keep_txt = ", ".join(
            _fmt(score, acc, has_think, step)
            for score, acc, has_think, step, _ in scored[:limit]
        )
        logging.warning(
            f"[gen_eval] top-{limit} by score=acc+has_think kept: [{keep_txt}]; "
            f"pruned: [{', '.join(pruned) or 'none'}]"
        )

    def _record_and_maybe_save_topk(
        self,
        model,
        *,
        output_dir: str,
        global_step: int,
        metrics: Dict[str, Any],
    ) -> None:
        """记录本 step 的 gen_eval 分数；若进入 top-K 则落盘 adapter，并按分数裁剪。"""
        score = self._rank_score(metrics)
        qualifies = self._qualifies_for_topk(score, global_step)
        self.step_metrics[global_step] = dict(metrics)
        if qualifies:
            ckpt_dir = self._save_step_checkpoint(
                model,
                output_dir=output_dir,
                global_step=global_step,
                metrics=metrics,
            )
            logging.warning(
                f"[gen_eval] top-K candidate score={score:.4f} "
                f"(acc={metrics['acc']:.2%}, has_think={metrics['has_think']:.2%}) "
                f"at step={global_step} -> saved {ckpt_dir}"
            )
        else:
            logging.warning(
                f"[gen_eval] step={global_step} score={score:.4f} "
                f"(acc={metrics['acc']:.2%}, has_think={metrics['has_think']:.2%}) "
                f"not in top-{self.save_total_limit}, skip saving checkpoint"
            )
        self._rotate_checkpoints_by_gen_eval(output_dir)

    @torch.no_grad()
    def _run(self, model, *, global_step: int, output_dir: str) -> Optional[Dict[str, Any]]:
        was_training = model.training
        model.eval()
        use_cache_before = getattr(model.config, "use_cache", None)
        model.config.use_cache = True
        # 生成必须左 padding（同一 batch 内不同长度的 prompt 对齐到右侧统一开始生成），跟
        # load_policy_model 里给 tokenizer 设的 padding_side="left" 一致；这里再显式兜底一次，
        # 防止 tokenizer 对象被其它地方改动过 padding_side。
        prev_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        device = next(model.parameters()).device
        n_has_think = 0
        n_correct = 0
        n_total = 0
        rows: List[Dict[str, Any]] = []
        n_batches = (len(self.samples) + self.batch_size - 1) // self.batch_size
        t_start = time.monotonic()
        try:
            for batch_idx in range(n_batches):
                batch = self.samples[batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size]
                prompts = [rec["prompt"] for rec in batch]
                inputs = self.tokenizer(
                    prompts, return_tensors="pt", padding=True, add_special_tokens=False
                ).to(device)
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    min_new_tokens=2,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                # 左 padding 后同一 batch 内所有 prompt 的 token 数一致，新生成内容统一从
                # inputs["input_ids"].shape[1] 开始，不需要逐条重新计算切片起点。
                prompt_len = inputs["input_ids"].shape[1]
                for local_i, (rec, out_row) in enumerate(zip(batch, output_ids)):
                    gen_text = self.tokenizer.decode(out_row[prompt_len:], skip_special_tokens=True)
                    # DPO pair：prompt 已含 think_prefix，需拼回完整 raw；
                    # mam_val gen_eval：prompt 不含 think，模型自己生成 <think>...（与线下一致）。
                    prompt_has_think = rec.get(
                        "_prompt_has_think_prefix",
                        rec["prompt"].endswith(self.think_prefix),
                    )
                    raw_pred = (self.think_prefix + gen_text) if prompt_has_think else gen_text
                    has_think = int(_has_think_tags(raw_pred))
                    question = rec.get("question") or _extract_question_from_prompt(rec["prompt"])
                    pred = _apply_cot_candidate_constraint(question, raw_pred)
                    gold = _normalize_pred_text(rec.get("gold", ""))
                    correct = int(pred == gold)
                    sample_idx = rec.get("idx")
                    if sample_idx is None:
                        sample_idx = batch_idx * self.batch_size + local_i
                    rows.append(
                        {
                            "idx": sample_idx,
                            "prompt": rec["prompt"],
                            "raw_prediction": raw_pred,
                            "prediction": pred,
                            "gold": gold,
                            "has_think": has_think,
                            "correct": correct,
                        }
                    )
                    n_total += 1
                    n_has_think += has_think
                    n_correct += correct
                logging.warning(
                    f"[gen_eval] step={global_step} progress: batch {batch_idx + 1}/{n_batches} "
                    f"({n_total}/{len(self.samples)} samples), elapsed={time.monotonic() - t_start:.1f}s"
                )
        finally:
            model.config.use_cache = use_cache_before
            self.tokenizer.padding_side = prev_padding_side
            if was_training:
                model.train()
        if not n_total:
            return None
        out_path = Path(output_dir) / "gen_eval" / f"step-{global_step}_predictions.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        acc = n_correct / n_total
        has_think = n_has_think / n_total
        logging.warning(
            f"[gen_eval] step={global_step} "
            f"n={n_total}, has_think={n_has_think}/{n_total} ({has_think:.2%}), "
            f"acc={n_correct}/{n_total} ({acc:.2%}), "
            f"total_time={time.monotonic() - t_start:.1f}s, "
            f"wrote {len(rows)} rows -> {out_path} "
            "(has_think 要求 <think> 和 </think> 都出现，即模型在 max_new_tokens 预算内正常收尾；"
            "acc 是抽取式准确率，跟 el/eval_CoT_checkpoint.py 的 has_think/Accuracy 同口径)。"
        )
        return {
            "n": n_total,
            "acc": acc,
            "has_think": has_think,
            "n_correct": n_correct,
            "n_has_think": n_has_think,
        }


def save_default_adapter_only(trainer: DPOTrainer, tokenizer, output_dir: str) -> None:
    """训练结束后只保存 "default" adapter（删除 trl 自动创建的 "ref" adapter 快照），
    产物目录结构与 --adapter_name_or_path 的 checkpoint 一致，可直接被
    el/eval_CoT_checkpoint.py / rl/gen_dpo_data.py 的 --model 参数加载。
    """
    unwrapped = trainer.accelerator.unwrap_model(trainer.model)
    peft_config = getattr(unwrapped, "peft_config", None)
    if peft_config is not None and "ref" in peft_config:
        unwrapped.delete_adapter("ref")
    trainer.save_model(output_dir=output_dir)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(output_dir)


class AuxDataCollatorForPreference(DataCollatorForPreference):
    """跟 trl 默认的 DataCollatorForPreference 完全一样，只多做一件事：把每条样本的
    "aux_row_id"（若存在）整理成一个 LongTensor 一起返回，供 AuxDPOTrainer 索引 δ 表。
    eval 数据集没有这一列（见 make_dpo_dataset_module），这里用 `"aux_row_id" in examples[0]`
    判断是否要加这一列，同一个 collator 实例可以同时服务 train/eval 两个 dataloader。
    """

    def torch_call(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        output = super().torch_call(examples)
        if "aux_row_id" in examples[0]:
            output["aux_row_id"] = torch.tensor([ex["aux_row_id"] for ex in examples], dtype=torch.long)
        return output


class AuxDPOTrainer(DPOTrainer):
    """DPOTrainer 的一个子类，实现文件头说明第 6 点里描述的、AuxDPO（arXiv:2510.20413）思路的
    工程近似：给每条训练 pair 一个可训练的辅助偏移量 δ（tanh 硬 cap + L2 收缩 + 独立小学习率），
    加进 DPO 的 margin 里再算 sigmoid loss。

    实现上没有复用 trl.DPOTrainer._compute_loss（那个方法要处理 hinge/ipo/aot/... 一长串 trl
    自带的 loss_type，逐一分支塞 δ 进去既繁琐又容易在 trl 升级时悄悄错位），而是重写了一个更精简
    的版本，只覆盖本文件实际会产出的两种组合：base loss_type ∈ {"sigmoid","sigmoid_norm"}
    （对应 --length_normalize），可选再叠加一个 "sft" 项（对应 --rpo_alpha）。其它 loss_type
    组合会在 __init__ 里直接报错，不会静默按普通 DPO 跑掉。reference log-prob 的计算直接复用
    父类已经封装好的 `compute_ref_log_probs`，没有重复实现 adapter 切换 / precompute 逻辑。
    """

    def __init__(
        self,
        *args: Any,
        aux_num_train_examples: int,
        auxdpo_delta_cap: float,
        auxdpo_l2: float,
        auxdpo_lr: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if self.f_divergence_type != "reverse_kl":
            raise NotImplementedError(
                "AuxDPOTrainer 只实现了默认的 f_divergence_type='reverse_kl'（标准 DPO），本文件也 "
                "没有暴露修改它的参数，出现这个错误说明代码被改动过。"
            )
        if self.use_weighting:
            raise NotImplementedError("AuxDPOTrainer 不支持 --use_weighting（WPO 加权），本文件未暴露该开关。")
        base_loss_type = self.loss_types[0]
        if base_loss_type not in ("sigmoid", "sigmoid_norm"):
            raise NotImplementedError(
                f"AuxDPOTrainer 只支持 base loss_type 为 'sigmoid' 或 'sigmoid_norm'（对应 "
                f"--length_normalize 的两种取值），当前 loss_type={self.loss_types!r} 不受支持——"
                "其它 trl loss_type（hinge/ipo/aot/...）没有接入 δ 的 margin 计算。"
            )
        if len(self.loss_types) > 1 and self.loss_types[1:] != ["sft"]:
            raise NotImplementedError(
                f"AuxDPOTrainer 只支持 base loss_type 之后再叠加一个 'sft' 项（对应 --rpo_alpha），"
                f"当前 loss_type={self.loss_types!r} 不受支持。"
            )
        self.auxdpo_delta_cap = auxdpo_delta_cap
        self.auxdpo_l2 = auxdpo_l2
        self.auxdpo_lr = auxdpo_lr
        # 训练用的辅助状态，不是模型的一部分：不进 PeftModel、不随 adapter 保存/加载，训练结束后
        # 随 trainer 一起丢弃。用 float32（数值范围很小，O(1)，没必要跟主模型一样上 bf16）。
        self.aux_delta_raw = torch.nn.Parameter(
            torch.zeros(aux_num_train_examples, dtype=torch.float32, device=self.accelerator.device)
        )
        self._auxdpo_optimizer_group_added = False
        logging.warning(
            f"[auxdpo] enabled: {aux_num_train_examples} per-pair 辅助偏移量 δ "
            f"(delta_cap={auxdpo_delta_cap}, l2={auxdpo_l2}, lr={auxdpo_lr})；"
            "训练日志里额外的 aux/* 指标含义见文件头说明第 6 点 (d)。"
        )

    def _set_signature_columns_if_needed(self) -> None:
        super()._set_signature_columns_if_needed()
        if self._signature_columns is not None and "aux_row_id" not in self._signature_columns:
            self._signature_columns = list(self._signature_columns) + ["aux_row_id"]

    def create_optimizer(self):  # noqa: D401
        optimizer = super().create_optimizer()
        if not self._auxdpo_optimizer_group_added:
            optimizer.add_param_group(
                {"params": [self.aux_delta_raw], "lr": self.auxdpo_lr, "weight_decay": 0.0}
            )
            self._auxdpo_optimizer_group_added = True
        return optimizer

    def _compute_loss(self, model, inputs, return_outputs):
        mode = "train" if self.model.training else "eval"
        aux_row_id = inputs.get("aux_row_id")

        _non_model_keys = {"completion_mask", "ref_chosen_logps", "ref_rejected_logps", "aux_row_id"}
        model_kwargs = {k: v for k, v in inputs.items() if k not in _non_model_keys}
        model_kwargs["use_cache"] = False
        outputs = model(**model_kwargs)

        input_ids = inputs["input_ids"]
        completion_mask = inputs["completion_mask"]
        shift_logits = outputs.logits[..., :-1, :]
        shift_labels = input_ids[..., 1:]
        shift_completion_mask = completion_mask[..., 1:]
        per_token_logps = selective_log_softmax(shift_logits, shift_labels)
        per_token_logps[shift_completion_mask == 0] = 0.0
        logps = per_token_logps.sum(dim=1)
        chosen_logps, rejected_logps = logps.chunk(2, dim=0)

        if self.precompute_ref_logps:
            ref_chosen_logps, ref_rejected_logps = inputs["ref_chosen_logps"], inputs["ref_rejected_logps"]
        else:
            # compute_ref_log_probs（父类方法）自己会按 _non_model_keys 过滤 model kwargs，但它不
            # 认识 "aux_row_id"，这里先剥掉，否则会被当成 model.forward 的一个 kwargs 传下去报错。
            ref_inputs = {k: v for k, v in inputs.items() if k != "aux_row_id"}
            ref_chosen_logps, ref_rejected_logps = self.compute_ref_log_probs(ref_inputs)

        chosen_logratios = chosen_logps - ref_chosen_logps
        rejected_logratios = rejected_logps - ref_rejected_logps

        # 长度归一化要用未 shift 的 completion_mask（跟 trl._compute_loss 的 'sigmoid_norm' 分支
        # 完全一致的口径），sft 交叉熵项要用 shift 后的 mask 去对齐 shift_labels，两者不能混用。
        chosen_len_mask, rejected_len_mask = completion_mask.chunk(2, dim=0)
        if self.loss_types[0] == "sigmoid_norm":
            chosen_avg = chosen_logratios / chosen_len_mask.sum(dim=1).clamp(min=1.0)
            rejected_avg = rejected_logratios / rejected_len_mask.sum(dim=1).clamp(min=1.0)
            margin = self.beta * (chosen_avg - rejected_avg)
        else:  # "sigmoid"
            margin = self.beta * (chosen_logratios - rejected_logratios)

        # 不含 δ 的 margin，只用来算 aux/reward_accuracies_no_aux 诊断指标（见文件头说明第 6 点
        # (d)），不参与梯度。
        base_margin = margin.detach().clone()

        reg_loss = margin.new_zeros(())
        aux_delta = None
        if aux_row_id is not None and mode == "train":
            raw = self.aux_delta_raw[aux_row_id.to(self.aux_delta_raw.device)].to(margin.dtype)
            aux_delta = self.auxdpo_delta_cap * torch.tanh(raw)
            margin = margin + aux_delta
            reg_loss = self.auxdpo_l2 * (aux_delta**2).mean()

        loss = -F.logsigmoid(margin).mean() * self.loss_weights[0] + reg_loss

        if len(self.loss_types) > 1:  # "sft" / --rpo_alpha
            chosen_logits, _ = shift_logits.chunk(2, dim=0)
            chosen_labels, _ = shift_labels.chunk(2, dim=0)
            chosen_sft_mask, _ = shift_completion_mask.chunk(2, dim=0)
            sft_loss = F.cross_entropy(chosen_logits[chosen_sft_mask.bool()], chosen_labels[chosen_sft_mask.bool()])
            loss = loss + self.loss_weights[1] * sft_loss

        # ---- 指标记录：跟 trl 原生 DPOTrainer 同名的一部分（rewards/*、logps/*，方便跟不开
        # --use_auxdpo 的历史训练日志直接对比），外加 AuxDPO 专属的 aux/* 诊断指标。没有搬运原生
        # entropy / mean_token_accuracy / logits/* 这几个（跟本文件的判断逻辑无关，见文件头
        # 说明第 6 点：真实退化情况本来就该靠 GenerationDegenerationCallback 的 generate 结果看，
        # 不是靠这些 teacher-forcing 附加指标）。
        chosen_rewards = self.beta * chosen_logratios.detach()
        rejected_rewards = self.beta * rejected_logratios.detach()
        self._metrics[mode]["rewards/chosen"].append(self.accelerator.gather(chosen_rewards).mean().item())
        self._metrics[mode]["rewards/rejected"].append(self.accelerator.gather(rejected_rewards).mean().item())
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        self._metrics[mode]["rewards/accuracies"].append(self.accelerator.gather(reward_accuracies).mean().item())
        reward_margins = chosen_rewards - rejected_rewards
        self._metrics[mode]["rewards/margins"].append(self.accelerator.gather(reward_margins).mean().item())
        self._metrics[mode]["logps/chosen"].append(self.accelerator.gather(chosen_logps).mean().item())
        self._metrics[mode]["logps/rejected"].append(self.accelerator.gather(rejected_logps).mean().item())
        if aux_delta is not None:
            no_aux_accuracy = (base_margin > 0).float()
            self._metrics[mode]["aux/reward_accuracies_no_aux"].append(
                self.accelerator.gather(no_aux_accuracy).mean().item()
            )
            self._metrics[mode]["aux/delta_mean"].append(self.accelerator.gather(aux_delta.detach()).mean().item())
            self._metrics[mode]["aux/delta_abs_mean"].append(
                self.accelerator.gather(aux_delta.detach().abs()).mean().item()
            )
            self._metrics[mode]["aux/reg_loss"].append(self.accelerator.gather(reg_loss.detach()).mean().item())

        return (loss, outputs) if return_outputs else loss


def main() -> None:
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.length_normalize:
        loss_type = list(training_args.loss_type)
        if "sigmoid" in loss_type:
            loss_type = ["sigmoid_norm" if lt == "sigmoid" else lt for lt in loss_type]
            training_args.loss_type = loss_type
            logging.warning(
                f"[dpo] length_normalize=True -> loss_type={training_args.loss_type} "
                "('sigmoid' -> 'sigmoid_norm'，pairwise 项按 completion token 数取平均后再比较，"
                "见文件头'第二个教训'消除长度偏置)。"
            )

    if training_args.rpo_alpha > 0:
        loss_type = list(training_args.loss_type)
        if "sft" not in loss_type:
            base_weights = list(training_args.loss_weights) if training_args.loss_weights else [1.0] * len(loss_type)
            training_args.loss_type = loss_type + ["sft"]
            training_args.loss_weights = base_weights + [training_args.rpo_alpha]
            logging.warning(
                f"[dpo] rpo_alpha={training_args.rpo_alpha} -> loss_type={training_args.loss_type}, "
                f"loss_weights={training_args.loss_weights}"
            )

    if data_args.dry_run:
        adapter_dir = Path(model_args.adapter_name_or_path)
        if not adapter_dir.is_absolute():
            adapter_dir = (_PROJECT_ROOT / adapter_dir).resolve()
        tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), use_fast=False)
        tokenizer.padding_side = "left"
        make_dpo_dataset_module(tokenizer, data_args, training_args.max_length, training_args.seed)
        logging.warning("[dry-run] skipped model loading / training.")
        return

    tokenizer, model = load_policy_model(model_args, training_args)
    data_module = make_dpo_dataset_module(tokenizer, data_args, training_args.max_length, training_args.seed)

    if data_module["eval_dataset"] is not None:
        if training_args.eval_strategy == "no":
            training_args.eval_strategy = "steps"
        logging.warning(f"[dpo] eval_strategy={training_args.eval_strategy}, eval_steps={training_args.eval_steps}")
    else:
        training_args.eval_strategy = "no"

    trainer_cls: type = DPOTrainer
    trainer_kwargs: Dict[str, Any] = {}
    if training_args.use_auxdpo:
        trainer_cls = AuxDPOTrainer
        trainer_kwargs.update(
            aux_num_train_examples=len(data_module["train_dataset"]),
            auxdpo_delta_cap=training_args.auxdpo_delta_cap,
            auxdpo_l2=training_args.auxdpo_l2,
            auxdpo_lr=training_args.auxdpo_lr,
            # 需要显式传自定义 collator：DPOTrainer.__init__ 缺省只会构造不认识 "aux_row_id" 的
            # 原生 DataCollatorForPreference，那样这一列会在 collate 阶段被丢掉。
            data_collator=AuxDataCollatorForPreference(
                pad_token_id=tokenizer.pad_token_id,
                max_length=training_args.max_length,
                truncation_mode=training_args.truncation_mode,
                pad_to_multiple_of=training_args.pad_to_multiple_of,
            ),
        )
        if training_args.world_size > 1 and training_args.num_train_epochs > 1:
            logging.warning(
                "[auxdpo] 警告：多卡 (world_size="
                f"{training_args.world_size}) + num_train_epochs="
                f"{training_args.num_train_epochs} > 1，每个训练样本的 δ 在不同 epoch 可能被 "
                "DistributedSampler 重新洗牌分到不同卡上，而 aux_delta_raw 没有做跨卡同步，训练不会 "
                "报错但对应样本的 δ 会在换卡后从 0 重新学，语义打折——见文件头说明第 6 点 (b)，建议 "
                "改成单卡训练，或接受这个近似。"
            )

    # save_best_by_gen_eval：按 gen_eval score=acc+has_think 保留 top-K，关掉 HF Trainer 按“最近 N 个”的轮转，
    # 否则它会把高分但较早的 checkpoint 删掉（用户上次跑只剩最后 4 个就是这个原因）。
    gen_eval_save_total_limit: Optional[int] = training_args.save_total_limit
    if training_args.save_best_by_gen_eval:
        if training_args.save_total_limit is not None and training_args.save_total_limit > 0:
            logging.warning(
                f"[dpo] save_best_by_gen_eval=True: checkpoint 轮转改为按 gen_eval "
                f"score=acc+has_think 保留 top-{training_args.save_total_limit}"
                f"（关闭 HF Trainer 的最近-N 轮转）。"
            )
        training_args.save_total_limit = None

    trainer = trainer_cls(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=data_module["train_dataset"],
        eval_dataset=data_module["eval_dataset"],
        processing_class=tokenizer,
        **trainer_kwargs,
    )

    if training_args.gen_eval_steps > 0:
        if training_args.gen_eval_data_path:
            gen_eval_pool = load_mam_gen_eval_records(
                training_args.gen_eval_data_path,
                training_args.gen_eval_num_samples,
            )
            gen_eval_shuffle = False
        else:
            gen_eval_pool = data_module["eval_dataset"] or data_module["train_dataset"]
            gen_eval_pool = list(gen_eval_pool)
            gen_eval_shuffle = True
        trainer.add_callback(
            GenerationDegenerationCallback(
                tokenizer,
                gen_eval_pool,
                num_samples=training_args.gen_eval_num_samples,
                max_new_tokens=training_args.gen_eval_max_new_tokens,
                every_n_steps=training_args.gen_eval_steps,
                batch_size=training_args.gen_eval_batch_size,
                seed=training_args.seed,
                shuffle=gen_eval_shuffle,
                save_best_by_gen_eval=training_args.save_best_by_gen_eval,
                save_total_limit=gen_eval_save_total_limit,
            )
        )
        src = training_args.gen_eval_data_path or "DPO eval/train held-out"
        if training_args.save_best_by_gen_eval:
            k_txt = (
                str(gen_eval_save_total_limit)
                if gen_eval_save_total_limit is not None and gen_eval_save_total_limit > 0
                else "unlimited"
            )
            best_msg = (
                f"; save_best_by_gen_eval=True -> keep top-{k_txt} checkpoint-{{step}} "
                f"by score=acc+has_think "
                f"(leaderboard: {training_args.output_dir}/gen_eval_topk.json)"
            )
        else:
            best_msg = ""
        logging.warning(
            f"[dpo] gen_eval enabled: every {training_args.gen_eval_steps} steps, "
            f"{training_args.gen_eval_num_samples} samples from {src} (batch_size="
            f"{training_args.gen_eval_batch_size}), max_new_tokens="
            f"{training_args.gen_eval_max_new_tokens}; predictions written to "
            f"{training_args.output_dir}/gen_eval/step-{{N}}_predictions.jsonl "
            f"(same schema as el/eval_CoT_checkpoint.py mam_val jsonl; see [gen_eval] logs)"
            f"{best_msg}."
        )
    elif training_args.save_best_by_gen_eval:
        logging.warning(
            "[dpo] --save_best_by_gen_eval True 被忽略：需要同时设置 --gen_eval_steps > 0。"
        )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_state()
    save_default_adapter_only(trainer, tokenizer, training_args.output_dir)
    logging.warning(f"[dpo] saved final adapter -> {training_args.output_dir}")


if __name__ == "__main__":
    main()
