#!/usr/bin/env python3
"""
DPO 训练脚本（非 CoT 版）：在普通 SFT LoRA checkpoint（如 result_mammo2/checkpoint-21000）
基础上，用 rl/gen_dpo_data4sft.py 产出的 {prompt, accept, reject, gold} pair 数据继续做偏好
优化，底层同样用 trl.DPOTrainer（本机 trl==1.8.0）。

本文件是 rl/train_dpo.py（CoT-SFT 版）的精简改版，去掉了所有 CoT 专属逻辑（<think> 前缀处理、
has_think 统计）与 AuxDPO；改之前建议先看一遍 rl/train_dpo.py 文件头，那里记录的两条踩坑经验
（prompt/completion tokenize 边界合并 bug、chosen 系统性比 rejected 长导致的长度偏置）里，
第一条在本文件以另一种形式复现（见下面第 1 点），第二条在本文件的数据形态下基本不适用（见第 2 点）。

===================== 关键设计点 =====================

1) prompt/completion 的 tokenize 边界合并问题，在非 CoT 数据下以另一种形式出现：
   rl/train_dpo.py 踩过的坑是 prompt 固定以 "### Response:" 结尾（末尾是 ":" token），CoT 的
   completion 固定以 "<think>\\n" 开头，BPE 把结尾 ":" 和开头 "<" 合并成同一个 token，导致 trl
   "prompt 单独 tokenize 再从 prompt+completion 里切片" 的方式丢字符。

   本文件的数据没有 "<think>"，但 accept（gold，候选实体格式）**同样固定以 "<" 开头**，实测
   （见 checkpoint-21000 的 tokenizer）同样的 ":" + "<" 合并 bug 会发生：
       tokenize("...### Response:")            末尾是 [..., "ĠResponse", ":"]
       tokenize("...### Response:<Shannon...")  末尾变成 [..., "ĠResponse", ":<", "Sh", ...]
   即 ":<" 被合并成了同一个 token（id 与 rl/train_dpo.py 记录的 32352 一致），accept 的这个
   前导 "<" 会从 chosen_ids 里丢字符，重演同一类边界 bug。

   reject（本地模型的原始生成）**不总是**以 "<" 开头：实测发现该 checkpoint 在部分预测错误的
   样本上会先吐出 1 个噪声字符再接真正的候选实体，例如
   `"><sandra Jacobson [DESCRIPTION] None [TYPE] None>"`（用单条 generate、不用 batch/padding
   复现过一模一样的输出，确认是模型本身在该样本上的真实生成行为，不是本项目 batch generate 脚本
   的 padding/解码 bug）。这意味着不能简单套用 rl/train_dpo.py 的“把两边共享的固定前缀挪进
   prompt”方案（那里 chosen/rejected 的前缀字面上完全相同，这里 chosen 恒为 "<...>"，rejected
   前面偶尔多出的噪声字符两边并不共享，如果无脑把 "<" 挪进 prompt，遇到这种 reject 会把一个真实
   不存在的 "<" 字符插进 rejected 的实际内容里，污染数据而不是修复边界）。

   本文件在 build_dpo_records() 里用 `_sanitize_and_split_boundary()` 做两步：
     a) 清洗 reject：找到 reject 文本里第一个 "<" 的位置，把它之前的字符（真正候选实体之外的
        生成噪声）切掉；若整段里根本没有 "<"，整条样本丢弃。清洗后 reject 与 accept 一样固定
        以 "<" 开头。
     b) 把 accept/清洗后 reject 共享的这一个前导 "<" 字符挪进 prompt（prompt = prompt + "<"，
        accept/reject 各自去掉开头这个 "<"），并用实际 tokenizer 逐条验证 tokenize(prompt) 是否
        严格是 tokenize(prompt+accept) 与 tokenize(prompt+reject) 的前缀；仍不满足则整条丢弃
        （dropped_boundary_unfixable，见下面 `_sanitize_and_split_boundary()` 的文档字符串，
        实测存在挪 1 个字符也修不好的更深层损坏生成，例如 reject 是 "<<i>Some Title..."
        这种以连续两个 "<" 开头、并非真正候选实体格式的输出）。训练前还会用
        `_assert_no_prompt_completion_boundary_mismatch()` 对最终保留的数据再做一次抽样自检
        （逻辑不含任何 CoT 假设，原样从 rl/train_dpo.py 搬过来），双重保险。
   这里选择“清洗 reject 里的噪声前缀”而不是直接丢弃这些样本，是因为实测这类样本占错误预测里不
   算小的比例（几个手工核查的例子里接近一半），直接丢弃会损失相当一部分负样本；被切掉的只是生成
   噪声（不影响“模型选错了哪个候选实体”这个 DPO 真正要纠正的信号）。

2) 关于 --length_normalize 和 --rpo_alpha 是否还需要：
   rl/train_dpo.py 默认 --length_normalize True、建议 --rpo_alpha 1.0 起步，动机是那份 CoT
   数据里 chosen（teacher 生成的完整 <think> 推理 + 答案）系统性比 rejected（本地模型的原始
   生成，同样带 <think> 但更短/更容易不收尾）长得多（实测 completion token 长度比 p50=2.57，
   p90=5.81），标准 'sigmoid' pairwise loss 对序列 log-prob 求和、不做长度归一化，天然带一个
   "越长越优" 的虚假梯度信号。

   本文件的 accept/reject 都是**去掉共享前导 "<" 之后的单行候选实体字符串**
   （"Entity [DESCRIPTION] ... [TYPE] ...>"），没有任何推理过程，二者长度差异只来自"这个候选
   实体的名字/描述文本本身有多长"，跟"谁被偏好"没有系统性因果关系（gold 和错误猜测都是从同一批
   候选里选出来的，格式、字段完全一致）——不存在 CoT 数据里那种"chosen 恒定比 rejected 多一段
   推理"的结构性长度偏置。所以：
     - --length_normalize 默认改成 **False**（未强制关闭，仍可用 --length_normalize True 打
       开；但本文件数据形态下没有必须开启的理由，默认关闭、行为等价标准 DPO 'sigmoid' loss）。
     - --rpo_alpha 默认仍是 **0.0**（不变，与 rl/train_dpo.py 一致），保留这个开关：RPO 的"给
       chosen 加一个 NLL 项防止其似然被意外压低"是任何 DPO 训练都可能受益的通用技巧，不是
       CoT 专属的修复手段，所以没有理由把这个选项本身删掉，只是不强制建议开启（没有本文件数据
       特有的、必须开启它的理由；如果实测发现 chosen 似然被明显压低，可以照常打开试 1.0）。
   建议：先用默认（两者都关）跑一次，用 --gen_eval_steps 的真实 generate 准确率做判断；如果观察
   到明显的长度/退化问题再按需打开，不要一上来就叠加。

3) 不支持 AuxDPO：这是 rl/train_dpo.py 针对"CoT 数据长度失衡导致的 DPO misspecification"问题
   的一个工程近似修复（见其文件头说明第 6 点），既然第 2 点已经说明本文件数据形态下没有那种系统性
   长度失衡，AuxDPO 要解决的问题在这里没有对应的病灶，直接不实现（trainer 固定用 trl 原生
   DPOTrainer，没有 --use_auxdpo 开关）。

4) 其余设计（continue 训练同一个 LoRA adapter 当 reference model、tokenizer 必须从 adapter
   目录加载、显式按 --max_length 过滤超长样本而不依赖 trl 静默截断、GenerationDegenerationCallback
   做训练中的真实 generate 监控、--save_best_by_gen_eval 按分数保留 top-K checkpoint）跟
   rl/train_dpo.py 完全一致，只是把 CoT 专属的 has_think 统计去掉，score 只用 acc；
   GenerationDegenerationCallback 用的 build_prompt / 候选约束 / 归一化函数都换成 el/sft.py
   的非 CoT 版本（build_prompt、_apply_candidate_constraint、_normalize_pred_text），与
   el/eval_checkpoint.py 的线下评测口径保持一致。

===================== 用法示例 =====================
  conda activate tablellama-fa

  # 0) dry-run：只做数据加载 + 边界修复 + 长度过滤统计，不加载模型
  python rl/train_dpo4sft.py --output_dir result_dpo_sft_checkpoint-21000 \\
      --data_path /DATA1/khli/t&m/rl_sft_checkpoint-21000_all.jsonl --dry_run

  # 1) 单卡试跑
  python rl/train_dpo4sft.py \\
      --output_dir result_dpo_sft_checkpoint-21000 \\
      --data_path /DATA1/khli/t&m/rl_sft_checkpoint-21000_all.jsonl \\
      --bf16 True \\
      --per_device_train_batch_size 4 --gradient_accumulation_steps 4 \\
      --learning_rate 5e-6 --num_train_epochs 1 \\
      --eval_ratio 0.05 --eval_steps 50

  # 2) 多卡（每卡 1 进程）+ 训练中真实 generate 监控（跟 el/eval_checkpoint.py 同口径）：
  torchrun --nproc_per_node=2 rl/train_dpo4sft.py \\
      --output_dir result_dpo_sft_checkpoint-21000 \\
      --data_path /DATA1/khli/t&m/rl_sft_checkpoint-21000_all.jsonl \\
      --bf16 True \\
      --per_device_train_batch_size 4 --gradient_accumulation_steps 2 \\
      --learning_rate 5e-6 --num_train_epochs 1 \\
      --eval_ratio 0.05 --eval_steps 50 \\
      --gen_eval_steps 50 --gen_eval_num_samples 64 --gen_eval_batch_size 16 \\
      --gen_eval_max_new_tokens 128 \\
      --save_best_by_gen_eval True --save_total_limit 3

  # 3) 断点续训
  python rl/train_dpo4sft.py --output_dir result_dpo_sft_checkpoint-21000 \\
      --data_path /DATA1/khli/t&m/rl_sft_checkpoint-21000_all.jsonl \\
      --resume_from_checkpoint result_dpo_sft_checkpoint-21000/checkpoint-200

训练产物：--output_dir 下是一个可以像 checkpoint-21000 一样直接被
el/eval_checkpoint.py / rl/gen_dpo_data4sft.py 的 --model 参数加载的 LoRA adapter 目录。
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
import transformers
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState

from trl import DPOConfig, DPOTrainer

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# 复用 el/eval_checkpoint.py 同款的非 CoT 解析/评分逻辑，让训练中的 gen_eval 监控跟线下生成式
# 评测用的是同一套候选约束 / 归一化规则，结果才可比。
from el.sft import (  # noqa: E402
    _apply_candidate_constraint,
    _apply_eval_table_compression,
    _normalize_description_marker,
    _normalize_pred_text,
    build_prompt,
)

DEFAULT_ADAPTER = "result_mammo2/checkpoint-21000"
DEFAULT_DATA_PATH = "/DATA1/khli/t&m/rl_sft_checkpoint-21000_all.jsonl"
# prompt+completion 拼接后的 token 数上限，与 el/sft.py DEFAULT_MODEL_MAX_LENGTH /
# rl/gen_dpo_data4sft.py --max_input_length 保持同一量级。
DEFAULT_MAX_LENGTH = 2560


@dataclass
class ModelArguments:
    adapter_name_or_path: str = field(
        default=DEFAULT_ADAPTER,
        metadata={
            "help": "要继续训练的 LoRA checkpoint 目录（相对路径相对项目根），默认 "
            f"{DEFAULT_ADAPTER}。DPO 会在这个 adapter 上继续训练，同时把它训练开始时刻的 "
            "权重快照当作 reference model。"
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
            "help": "rl/gen_dpo_data4sft.py 输出的 DPO pair jsonl，每行含 "
            "prompt/accept/reject/gold 字段（脚本内部把 accept/reject 重命名为 "
            "trl 约定的 chosen/rejected，并做边界修复，见文件头说明第 1 点）。"
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
            "eval（仅用于监控 eval_loss / rewards/accuracies 等指标，不能替代 --gen_eval_steps "
            "的真实生成式准确率评测）。"
        },
    )
    debug_max_examples: Optional[int] = field(
        default=None,
        metadata={"help": "调试用：只取过滤后的前 N 条做 smoke test，不建议在正式训练时设置。"},
    )
    dry_run: bool = field(
        default=False,
        metadata={"help": "只加载数据、跑边界修复 + 长度过滤统计并打印，不加载模型、不训练。"},
    )
    max_chosen_reject_len_ratio: Optional[float] = field(
        default=None,
        metadata={
            "help": "缺省 None（不过滤，只打印统计）。设置为正数时，丢弃 "
            "token_len(chosen)/token_len(rejected) 超过该比例的 pair。见文件头说明第 2 点：本文件 "
            "数据形态下 chosen/rejected 都是单行候选实体字符串，理论上不存在系统性长度失衡，这个 "
            "选项主要用于诊断（先 --dry_run 看日志打印的 ratio 分布），一般不需要真的设置。"
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
            "'sft'）并设置对应的 loss_weights=[..., rpo_alpha]，即 RPO/Llama-3 论文里的 "
            "rpo_alpha：给 chosen 额外加一个交叉熵 NLL 项，防止 chosen 似然被训练压低。见文件头 "
            "说明第 2 点：这是通用 DPO 技巧，不是 CoT 专属修复，本文件数据形态下没有必须开启的 "
            "理由，默认关闭，可按需打开。"
        },
    )
    length_normalize: bool = field(
        default=False,
        metadata={
            "help": "默认关闭（与 rl/train_dpo.py 的默认相反）。True 时把 loss_type 里的基础 "
            "pairwise 项从 'sigmoid' 换成 trl 内置的 'sigmoid_norm'（chosen/rejected 按各自 "
            "completion token 数取平均 log-prob 后再比较）。见文件头说明第 2 点：本文件数据的 "
            "chosen/rejected 都是单行候选实体字符串，不存在 CoT 数据里那种'chosen 恒定多一段 "
            "推理导致系统性更长'的结构性偏置，默认不需要长度归一化；如果实测发现长度分布仍有明显 "
            "偏差，可显式传 --length_normalize True 打开。"
        },
    )
    gen_eval_steps: int = field(
        default=0,
        metadata={
            "help": "默认 0（关闭）。>0 时每训练这么多 step，就用当前 policy 在 eval_dataset（若 "
            "没有则用训练集前若干条）上抽样做一次真实 generate（贪心解码，长度用 "
            "--gen_eval_max_new_tokens，跟线下 el/eval_checkpoint.py 的默认口径一致），统计 "
            "抽取式准确率并写进 trainer 日志（键名 gen_eval/*）。只在 world process 0 上跑，用 "
            "try/except 包裹，失败不影响训练。"
        },
    )
    gen_eval_data_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "可选：独立的生成式评测集（mammotab 风格 JSON list，字段含 "
            "instruction/input_seg/question/output，与 el/eval_checkpoint.py --eval_data_path "
            "同格式）。设置后 gen_eval 不再从 DPO eval/train pair 里抽样本，而是用该文件前 "
            "--gen_eval_num_samples 条（固定顺序），prompt 用非 CoT build_prompt（不预填任何 "
            "候选括号），保证和线下 eval_checkpoint.py 评测可比。"
        },
    )
    gen_eval_num_samples: int = field(
        default=32,
        metadata={
            "help": "每次 --gen_eval_steps 触发时做真实 generate 的样本数。若设置了 "
            "--gen_eval_data_path，表示取该文件的前 N 条（不 shuffle）。"
        },
    )
    gen_eval_max_new_tokens: int = field(
        default=128,
        metadata={
            "help": "退化监控 generate 时的 max_new_tokens，默认 128，跟 el/eval_checkpoint.py "
            "非 CoT 评测脚本的默认值保持一致（不是 CoT 版本的 768，本文件模型不输出 <think> "
            "推理，答案本身很短）。"
        },
    )
    gen_eval_batch_size: int = field(
        default=16,
        metadata={"help": "退化监控一次 generate() 调用喂多少条样本（左 padding 后一起生成）。"},
    )
    save_best_by_gen_eval: bool = field(
        default=False,
        metadata={
            "help": "默认 False。True 时按 gen_eval 分数 score=acc（非 CoT 没有 has_think 维度）"
            "保留 top-K 个 checkpoint-{step}（K=--save_total_limit，默认 3），score 相同时保留 "
            "step 更小的；会关闭 HuggingFace Trainer 自带的“按最近 N 个”轮转。每次更新写 "
            "{output_dir}/gen_eval_topk.json。需要 --gen_eval_steps > 0。"
        },
    )

    # ---- 下面几个字段只是覆盖 DPOConfig 的默认值，字段本身继承自 DPOConfig/TrainingArguments ----
    max_length: Optional[int] = field(
        default=DEFAULT_MAX_LENGTH,
        metadata={"help": "prompt+completion 拼接后的 token 数上限，见文件头说明。"},
    )
    learning_rate: float = field(
        default=5e-6,
        metadata={"help": "LoRA + DPO 常用学习率比全量微调 DPO（trl 默认 1e-6）高一个量级。"},
    )
    num_train_epochs: float = field(
        default=1.0,
        metadata={"help": "DPO 很容易 1~2 epoch 内就 reward 饱和/过拟合，默认只跑 1 epoch。"},
    )
    per_device_train_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=4)
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
    """粗略估计 tokenize(prompt + completion) 的 token 数（+1 给自动补的 eos 留余量）。"""
    return len(tokenizer(prompt + completion).input_ids) + 1


def _sanitize_and_split_boundary(
    tokenizer, prompt: str, chosen_raw: str, rejected_raw: str,
) -> Optional[Tuple[str, str, str, bool]]:
    """见文件头说明第 1 点。返回 (new_prompt, chosen, rejected, was_cleaned)；以下情况返回
    None（调用方丢弃该样本）：
      - chosen_raw 不以 "<" 开头（数据不变式被打破，不应该发生）；
      - rejected_raw 里完全找不到 "<"（生成完全跑飞，不可解析）；
      - 清洗、挪前导 "<" 之后，仍无法让 tokenize(new_prompt) 同时是
        tokenize(new_prompt+chosen) 和 tokenize(new_prompt+rejected) 的严格前缀
        （实测存在更深层损坏的生成，例如 reject 是 "<<i>Small Sacrifices..."
        这种以连续两个 "<" 开头、并非真正候选实体格式的输出：挪 1 个 "<" 进 prompt 后
        剩余文本仍以 "<" 开头，导致 "Response:<" 与 "Response:" + "<<i>..." 在这两种整体
        tokenize 方式下产生不同的 BPE 分组，边界依然对不齐；这类样本本身也不构成"看起来像
        一个候选实体的错误答案"，直接丢弃比强行修复更合理）。
    这里直接用实际 tokenizer 逐条验证对齐，而不是只满足于"挪 1 个字符"这个启发式一定成立
    ——能覆盖所有未预见到的边界问题，而不仅是已知的这一种。
    """
    if not chosen_raw.startswith("<"):
        return None
    entity_start = rejected_raw.find("<")
    if entity_start == -1:
        return None
    was_cleaned = entity_start > 0
    rejected_clean = rejected_raw[entity_start:]

    new_prompt = prompt + "<"
    chosen = chosen_raw[1:]
    rejected = rejected_clean[1:]

    prompt_ids = tokenizer(new_prompt, add_special_tokens=False).input_ids
    for completion in (chosen, rejected):
        full_ids = tokenizer(new_prompt + completion, add_special_tokens=False).input_ids
        if full_ids[: len(prompt_ids)] != prompt_ids:
            return None
    return new_prompt, chosen, rejected, was_cleaned


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
    max_chosen_reject_len_ratio: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """把 {prompt, accept, reject, ...} 转成 trl 约定的 {prompt, chosen, rejected}，做边界
    修复（见文件头说明第 1 点）并按 --max_length 显式丢弃超长样本，打印漏斗统计。
    """
    kept: List[Dict[str, Any]] = []
    dropped_missing_field = 0
    dropped_boundary_unfixable = 0
    dropped_too_long = 0
    dropped_len_ratio = 0
    cleaned_leading_noise = 0
    len_ratios: List[float] = []
    for rec in raw_records:
        prompt = rec.get("prompt")
        chosen_raw = rec.get("accept")
        rejected_raw = rec.get("reject")
        if not prompt or not chosen_raw or not rejected_raw:
            dropped_missing_field += 1
            continue
        # SFT 训练时 el/sft.py 的 SupervisedDataset 会把 question/output 里的
        # "[DESCRIPTION]" 统一规范化成 "[DESC]"，模型实际学到/见到的候选实体标记是
        # "[DESC]"。rl/gen_dpo_data4sft.py 早期生成的数据（以及未来若不规范化直接落盘的
        # 数据）prompt/accept/reject/gold 里可能还是原始的 "[DESCRIPTION]"，这里统一做一次
        # 同样的规范化，确保 DPO 训练看到的候选实体标记跟 SFT 基座模型训练时一致，不引入
        # 额外的分布偏移。对已经是 "[DESC]" 的数据这是无操作（replace 找不到子串）。
        prompt = _normalize_description_marker(prompt)
        chosen_raw = _normalize_description_marker(chosen_raw)
        rejected_raw = _normalize_description_marker(rejected_raw)
        split = _sanitize_and_split_boundary(tokenizer, prompt, chosen_raw, rejected_raw)
        if split is None:
            dropped_boundary_unfixable += 1
            continue
        new_prompt, chosen, rejected, was_cleaned = split
        if was_cleaned:
            cleaned_leading_noise += 1

        chosen_completion_len = len(tokenizer(chosen, add_special_tokens=False).input_ids)
        rejected_completion_len = len(tokenizer(rejected, add_special_tokens=False).input_ids)
        len_ratio = chosen_completion_len / max(1, rejected_completion_len)
        len_ratios.append(len_ratio)
        if max_chosen_reject_len_ratio is not None and len_ratio > max_chosen_reject_len_ratio:
            dropped_len_ratio += 1
            continue

        chosen_len = _combined_token_len(tokenizer, new_prompt, chosen)
        rejected_len = _combined_token_len(tokenizer, new_prompt, rejected)
        if max(chosen_len, rejected_len) > max_length:
            dropped_too_long += 1
            continue
        kept.append(
            {
                "idx": rec.get("idx"),
                "prompt": new_prompt,
                "chosen": chosen,
                "rejected": rejected,
                "gold": _normalize_description_marker(rec.get("gold", "")),
            }
        )

    total = len(raw_records)
    dropped_total = dropped_missing_field + dropped_boundary_unfixable + dropped_too_long + dropped_len_ratio
    ratio = (dropped_total / total) if total else 0.0
    logging.warning(
        f"[{dataset_name}] kept {len(kept)}/{total}; dropped {dropped_total}/{total} ({ratio:.2%}) "
        f"(missing_field={dropped_missing_field}, "
        f"boundary_unfixable(no '<' in reject, or still misaligned after fix)={dropped_boundary_unfixable}, "
        f"exceeds_max_length_{max_length}={dropped_too_long}, "
        f"exceeds_len_ratio_{max_chosen_reject_len_ratio}={dropped_len_ratio}); "
        f"cleaned_leading_noise_before_first_'<'={cleaned_leading_noise}/{total} "
        "(reject 里第一个 '<' 之前有生成噪声、已切掉的样本数，见文件头说明第 1 点)。"
    )
    if len_ratios:
        logging.warning(
            f"[{dataset_name}] chosen/rejected completion token-len ratio: "
            f"p50={_percentile(len_ratios, 0.50):.2f}, p90={_percentile(len_ratios, 0.90):.2f}, "
            f"p99={_percentile(len_ratios, 0.99):.2f}, max={max(len_ratios):.2f} "
            "(本文件数据理论上应接近 1，明显偏离时说明本文件头说明第 2 点的假设不成立，可考虑打开 "
            "--length_normalize / --max_chosen_reject_len_ratio)。"
        )
    return kept


def _assert_no_prompt_completion_boundary_mismatch(
    records: List[Dict[str, Any]], tokenizer, *, dataset_name: str, sample_size: int = 200
) -> None:
    """训练前抽样自检：tokenize(prompt) 必须是 tokenize(prompt+chosen/rejected) 的严格前缀。
    build_dpo_records 已经通过把共享的前导 "<" 挪进 prompt 来规避边界合并 bug（见文件头说明第 1
    点），这里再做一次实测校验，防止以后数据格式变了又踩同一个坑却没人发现。
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
            "comment #1). Fix the prompt/completion split before training."
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
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

    logging.warning(f"[model] loading trainable LoRA adapter from {adapter_dir}")
    model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=True)
    model.config.use_cache = False
    return tokenizer, model


_QUESTION_RE = re.compile(r"### Question:\n(.*?)\n\n### Response:", flags=re.S)


def _extract_question_from_prompt(prompt: str) -> str:
    """从 build_dpo_records 产出的 prompt（"...### Question:\\n{question}\\n\\n### Response:"
    + 挪进来的前导 "<"）里抠出原始 question 文本，喂给 el.sft._apply_candidate_constraint 做
    候选实体约束抽取。正则不依赖 "### Response:" 后面跟什么，挪进来的 "<" 不影响匹配。
    """
    m = _QUESTION_RE.search(prompt or "")
    return m.group(1).strip() if m else ""


def load_mam_gen_eval_records(path: str, num_samples: int) -> List[Dict[str, Any]]:
    """加载 mammotab JSON（与 el/eval_checkpoint.py 同格式），取前 num_samples 条，
    转成 gen_eval 需要的 {idx, prompt, gold, question}。prompt 是纯 build_prompt 输出
    （不预填任何候选括号），与线下推理完全一致。
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
        # 与 SFT 训练时 SupervisedDataset 的预处理一致（见 el/sft.py），把 question/output
        # 里的 "[DESCRIPTION]" 规范化成 "[DESC]"，否则 gen_eval 喂给模型的 prompt 跟训练分布
        # 不一致，测出来的 gen_accuracy 会偏低、且不能跟线下 eval_checkpoint.py 的结果直接比较
        # （eval_checkpoint.py 目前也没做这个规范化，是另一个待修的问题，这里先保证本文件内部
        # 训练 <-> gen_eval 的候选格式自洽）。
        for key in ("question", "output"):
            if key in aligned:
                aligned[key] = _normalize_description_marker(aligned[key])
        records.append(
            {
                "idx": i,
                "prompt": build_prompt(aligned),
                "gold": aligned.get("output") or "",
                "question": aligned.get("question") or "",
                # 与线下 eval 一致：模型自己从零生成，prompt 不预填任何前导字符。
                "_prompt_has_entity_bracket_prefix": False,
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
    （贪心解码，max_new_tokens=--gen_eval_max_new_tokens），统计抽取式准确率（非 CoT 没有
    has_think 维度，score 就是 acc 本身）。

    动机同 rl/train_dpo.py：DPOTrainer 内部的 eval_loss / rewards/accuracies 全是
    teacher-forcing 指标，看不出模型自由生成时的真实准确率，只有真的调用一次 model.generate()
    才能看到。

    每次 gen_eval 还会把逐条结果写到
    ``{output_dir}/gen_eval/step-{global_step}_predictions.jsonl``，字段与
    el/eval_checkpoint.py 产出的 val_predictions jsonl 一致：
    idx / prompt / raw_prediction / prediction / gold / correct。

    ``save_best_by_gen_eval=True`` 时，按 gen_eval score=acc 保留 top-K 个 ``checkpoint-{step}``
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
        seed: int = 0,
        shuffle: bool = True,
        save_best_by_gen_eval: bool = False,
        save_total_limit: Optional[int] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.every_n_steps = every_n_steps
        self.max_new_tokens = max_new_tokens
        self.batch_size = max(1, batch_size)
        self.save_best_by_gen_eval = save_best_by_gen_eval
        self.save_total_limit = save_total_limit
        self.step_metrics: Dict[int, Dict[str, Any]] = {}
        pool = list(records)
        if shuffle:
            random.Random(seed).shuffle(pool)
        self.samples = pool[: max(1, num_samples)]

    @staticmethod
    def _rank_score(metrics: Optional[Dict[str, Any]]) -> float:
        """top-K 排名分：acc（非 CoT 没有 has_think 维度）。无分数视为 -1。"""
        if metrics is None:
            return -1.0
        return float(metrics.get("acc") or 0.0)

    def on_train_begin(
        self,
        args: "TrainingArguments",
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
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
        entries: List[Dict[str, Any]] = []
        for step, metrics in self.step_metrics.items():
            score = self._rank_score(metrics)
            entries.append(
                {
                    "global_step": step,
                    "gen_eval_score": score,
                    "gen_eval_acc": metrics["acc"],
                    "gen_eval_n": metrics.get("n"),
                    "gen_eval_n_correct": metrics.get("n_correct"),
                }
            )
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
                "n": entry.get("gen_eval_n"),
                "n_correct": entry.get("gen_eval_n_correct"),
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
        return (score, -global_step) > (worst_score, -worst_step)

    def _save_step_checkpoint(
        self,
        model,
        *,
        output_dir: str,
        global_step: int,
        metrics: Dict[str, Any],
    ) -> Path:
        ckpt_dir = Path(output_dir) / f"checkpoint-{global_step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        unwrapped = _unwrap_model(model)
        save_kwargs: Dict[str, Any] = {}
        peft_config = getattr(unwrapped, "peft_config", None)
        if peft_config is not None and "default" in peft_config:
            save_kwargs["selected_adapters"] = ["default"]
        unwrapped.save_pretrained(str(ckpt_dir), **save_kwargs)
        self.tokenizer.save_pretrained(str(ckpt_dir))
        meta = {
            "global_step": global_step,
            "gen_eval_score": self._rank_score(metrics),
            "gen_eval_acc": metrics["acc"],
            "gen_eval_n": metrics["n"],
            "gen_eval_n_correct": metrics["n_correct"],
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
        limit = self.save_total_limit
        if limit is None or limit <= 0:
            self._write_topk_meta(output_dir)
            return

        ckpts = self._list_numeric_checkpoints(output_dir)
        if len(ckpts) <= limit:
            self._write_topk_meta(output_dir)
            return

        scored: List[Tuple[float, float, int, Path]] = []
        for step, path in ckpts:
            metrics = self.step_metrics.get(step)
            score = self._rank_score(metrics)
            acc = float(metrics["acc"]) if metrics is not None else -1.0
            scored.append((score, acc, step, path))
        scored.sort(key=lambda x: (-x[0], x[2]))

        def _fmt(score: float, acc: float, step: int) -> str:
            if score < 0:
                return f"checkpoint-{step}(no_gen_eval)"
            return f"checkpoint-{step}(score={score:.4f},acc={acc:.2%})"

        pruned: List[str] = []
        for score, acc, step, path in scored[limit:]:
            shutil.rmtree(path, ignore_errors=True)
            pruned.append(_fmt(score, acc, step))

        self._write_topk_meta(output_dir)
        keep_txt = ", ".join(_fmt(score, acc, step) for score, acc, step, _ in scored[:limit])
        logging.warning(
            f"[gen_eval] top-{limit} by score=acc kept: [{keep_txt}]; "
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
                f"[gen_eval] top-K candidate score={score:.4f} (acc={metrics['acc']:.2%}) "
                f"at step={global_step} -> saved {ckpt_dir}"
            )
        else:
            logging.warning(
                f"[gen_eval] step={global_step} score={score:.4f} (acc={metrics['acc']:.2%}) "
                f"not in top-{self.save_total_limit}, skip saving checkpoint"
            )
        self._rotate_checkpoints_by_gen_eval(output_dir)

    @torch.no_grad()
    def _run(self, model, *, global_step: int, output_dir: str) -> Optional[Dict[str, Any]]:
        was_training = model.training
        model.eval()
        use_cache_before = getattr(model.config, "use_cache", None)
        model.config.use_cache = True
        prev_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        device = next(model.parameters()).device
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
                prompt_len = inputs["input_ids"].shape[1]
                for local_i, (rec, out_row) in enumerate(zip(batch, output_ids)):
                    gen_text = self.tokenizer.decode(out_row[prompt_len:], skip_special_tokens=True)
                    # DPO pair 的 prompt 已经把共享的前导 "<" 挪进去了（见文件头说明第 1 点），
                    # 需要拼回来才是完整 raw_prediction；mam_val gen_eval 的 prompt 不预填任何
                    # 前导字符，模型自己从零生成（与线下一致）。
                    prompt_has_bracket = rec.get(
                        "_prompt_has_entity_bracket_prefix",
                        rec["prompt"].endswith("<"),
                    )
                    raw_pred = ("<" + gen_text) if prompt_has_bracket else gen_text
                    question = rec.get("question") or _extract_question_from_prompt(rec["prompt"])
                    pred = _apply_candidate_constraint(question, raw_pred)
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
                            "correct": correct,
                        }
                    )
                    n_total += 1
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
        logging.warning(
            f"[gen_eval] step={global_step} n={n_total}, acc={n_correct}/{n_total} ({acc:.2%}), "
            f"total_time={time.monotonic() - t_start:.1f}s, wrote {len(rows)} rows -> {out_path} "
            "(抽取式准确率，跟 el/eval_checkpoint.py 的 Accuracy 同口径)。"
        )
        return {"n": n_total, "acc": acc, "n_correct": n_correct}


def save_default_adapter_only(trainer: DPOTrainer, tokenizer, output_dir: str) -> None:
    """训练结束后只保存 "default" adapter（删除 trl 自动创建的 "ref" adapter 快照），
    产物目录结构与 --adapter_name_or_path 的 checkpoint 一致，可直接被
    el/eval_checkpoint.py / rl/gen_dpo_data4sft.py 的 --model 参数加载。
    """
    unwrapped = trainer.accelerator.unwrap_model(trainer.model)
    peft_config = getattr(unwrapped, "peft_config", None)
    if peft_config is not None and "ref" in peft_config:
        unwrapped.delete_adapter("ref")
    trainer.save_model(output_dir=output_dir)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(output_dir)


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
                "('sigmoid' -> 'sigmoid_norm')。"
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

    gen_eval_save_total_limit: Optional[int] = training_args.save_total_limit
    if training_args.save_best_by_gen_eval:
        if training_args.save_total_limit is not None and training_args.save_total_limit > 0:
            logging.warning(
                f"[dpo] save_best_by_gen_eval=True: checkpoint 轮转改为按 gen_eval score=acc "
                f"保留 top-{training_args.save_total_limit}（关闭 HF Trainer 的最近-N 轮转）。"
            )
        training_args.save_total_limit = None

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=data_module["train_dataset"],
        eval_dataset=data_module["eval_dataset"],
        processing_class=tokenizer,
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
                f"; save_best_by_gen_eval=True -> keep top-{k_txt} checkpoint-{{step}} by score=acc "
                f"(leaderboard: {training_args.output_dir}/gen_eval_topk.json)"
            )
        else:
            best_msg = ""
        logging.warning(
            f"[dpo] gen_eval enabled: every {training_args.gen_eval_steps} steps, "
            f"{training_args.gen_eval_num_samples} samples from {src} (batch_size="
            f"{training_args.gen_eval_batch_size}), max_new_tokens="
            f"{training_args.gen_eval_max_new_tokens}; predictions written to "
            f"{training_args.output_dir}/gen_eval/step-{{N}}_predictions.jsonl{best_msg}."
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
