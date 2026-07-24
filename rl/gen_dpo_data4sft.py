#!/usr/bin/env python3
"""
RL (DPO) 训练数据生成脚本 —— 直接用 SFT（非 CoT）模型生成 reject，gold 作为 accept。

与 rl/gen_dpo_data.py 的区别：
  - rl/gen_dpo_data.py 面向 CoT-SFT 模型：本地推理错误的样本要交给 deepseek-v4-pro
    现场生成 "<think>...</think>\\n答案" 作为 accept（因为 CoT 模型的 reject 里带
    think 过程，需要一个同样带 think 的、但答案正确的 accept 来配对）。
  - 本脚本面向普通 SFT（非 CoT）模型：模型输出本身就是纯答案，没有 <think> 推理，
    所以完全不需要调用 DeepSeek —— 推理错误样本的 accept 直接用数据里的 gold 答案
    （规范化为候选实体格式），reject 用本地模型的原始生成（去掉结尾 </s>）。

流程：
  1) 用本地 SFT LoRA checkpoint 对 --input_data_file 里的数据做推理，推理逻辑与
     el/eval_checkpoint.py 一致（_apply_eval_table_compression -> 规范化 "[DESCRIPTION]"
     -> "[DESC]"（与 el/sft.py SupervisedDataset 训练时的预处理一致，见 _prepare_ex_for_prompt）
     -> build_prompt -> greedy generate -> _apply_candidate_constraint / _normalize_pred_text），
     本地生成按 --gen_batch_size 做批量 batch generate。
  2) 找出 prediction（去掉结尾 </s>）与 gold 不匹配的样本：
       reject = 本地模型的原始生成（去掉结尾 </s>）
       accept = gold 答案（规范化为候选实体格式 "<...>"，标记已是 "[DESC]"）
  3) 组成 DPO pair 写入 /DATA1/khli/t&m/rl_sft_{model}_{num}.jsonl：
       {"prompt": ..., "accept": "<Entity [DESC] ... [TYPE] ...>", "reject": raw_prediction}

多卡（torchrun）时按“round”（每 round = --gen_batch_size 条样本）轮转分配给各 rank，
round 之间各 rank 完全独立并行处理，只在每 --sync_every_rounds 个 round 做一次同步。

用法示例：
  conda activate tablellama-fa
  python rl/gen_dpo_data4sft.py --num 200
  python rl/gen_dpo_data4sft.py --model result_mammo2/checkpoint-21000 \\
      --input_data_file /DATA1/khli/t&m/merged_SFT_70k&35k.json --num 500
  python rl/gen_dpo_data4sft.py --num 200 --resume   # 断点续跑（多卡时从 .rank*.jsonl 分片恢复）

  # 多 GPU（每卡 1 进程，与 eval_checkpoint.py 一致）：
  torchrun --nproc_per_node=2 rl/gen_dpo_data4sft.py --num 200
  torchrun --nproc_per_node=2 rl/gen_dpo_data4sft.py --num 200 --resume  # 续跑读主文件 + rank 分片
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.distributed as dist
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

try:
    import ijson
except ImportError:  # pragma: no cover
    ijson = None

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# 复用现有推理 / 评估逻辑（与 el/eval_checkpoint.py 一致，均为“非 CoT”版本）
from el.sft import (
    _apply_candidate_constraint,
    _apply_eval_table_compression,
    _normalize_description_marker,
    _normalize_pred_text,
    build_prompt,
)

# DEFAULT_MODEL = "result_mammo2/checkpoint-21000"
DEFAULT_MODEL = "result_dpo_sft/checkpoint-150"
DEFAULT_INPUT = "/DATA1/khli/t&m/merged_SFT_70k&35k.json"


def _strip_trailing_eos(text: str) -> str:
    """去掉 decode 结果末尾的 </s> 等 EOS 残留，保留完整生成作为 reject。"""
    if not isinstance(text, str):
        return ""
    return re.sub(r"</?s>\s*$", "", text.strip(), flags=re.I).strip()


def _prepare_ex_for_prompt(ex: Dict[str, Any]) -> Dict[str, Any]:
    """与 el/sft.py SupervisedDataset 的预处理顺序一致：先压缩表格，再对 question/output 做
    [DESCRIPTION] -> [DESC] 规范化。SFT 训练时模型实际学习/见到的候选实体标记是 "[DESC]"
    （见 SupervisedDataset.__init__ 里对 question/output 的 _normalize_description_marker
    调用），如果这里不做同样的规范化，本地模型看到的推理 prompt 就是训练分布之外的
    "[DESCRIPTION]" 格式，生成的 reject 以及作为 accept 的 gold 也会跟训练格式不一致。
    """
    compressed = _apply_eval_table_compression(ex)
    normalized = dict(compressed)
    for key in ("question", "output"):
        if key in normalized:
            normalized[key] = _normalize_description_marker(normalized[key])
    return normalized


# ----------------------------- distributed -----------------------------
def setup_dist() -> Tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_distributed = world_size > 1

    if is_distributed:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            dist.init_process_group(
                backend="nccl",
                device_id=torch.device(f"cuda:{local_rank}"),
                timeout=timedelta(hours=6),
            )
        else:
            dist.init_process_group(backend="gloo", timeout=timedelta(hours=6))

    return is_distributed, world_size, rank, local_rank


def dist_barrier(*, is_distributed: bool, local_rank: int) -> None:
    """多卡时用于同步点（每 sync_every_rounds 个 round 一次），保证 collective 各 rank 同时进入。"""
    if not is_distributed:
        return
    if torch.cuda.is_available():
        dist.barrier(device_ids=[local_rank])
    else:
        dist.barrier()


def broadcast_resume_state(
    start_idx: int,
    collected_baseline: int,
    *,
    is_distributed: bool,
    device: torch.device,
) -> Tuple[int, int]:
    if not is_distributed:
        return start_idx, collected_baseline
    payload = torch.tensor([start_idx, collected_baseline], dtype=torch.long, device=device)
    dist.broadcast(payload, src=0)
    return int(payload[0].item()), int(payload[1].item())


def all_reduce_sum(value: int, device: torch.device, is_distributed: bool) -> int:
    if not is_distributed:
        return value
    t = torch.tensor([value], dtype=torch.long, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return int(t.item())


def rank_shard_path(out_path: Path, rank: int) -> Path:
    return out_path.parent / f"{out_path.stem}.rank{rank}{out_path.suffix}"


def load_jsonl_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def merge_rank_outputs(
    out_path: Path,
    world_size: int,
    *,
    existing_records: Optional[List[Dict[str, Any]]] = None,
    target_num: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """合并已有记录与各 rank shard，按 idx 排序并去重（同 idx 保留后者）。"""
    by_idx: Dict[int, Dict[str, Any]] = {}
    for rec in existing_records or []:
        idx = rec.get("idx")
        if isinstance(idx, int):
            by_idx[idx] = rec

    for r in range(world_size):
        shard = rank_shard_path(out_path, r)
        if not shard.exists():
            continue
        for rec in load_jsonl_records(shard):
            idx = rec.get("idx")
            if isinstance(idx, int):
                by_idx[idx] = rec

    merged = [by_idx[k] for k in sorted(by_idx)]
    if target_num is not None and len(merged) > target_num:
        merged = merged[:target_num]
    return merged


def compute_resume_state(
    out_path: Path,
    world_size: int,
    *,
    use_rank_shards: bool,
) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    汇总可恢复的 DPO 记录，返回 (records, pair_count, start_idx)。
    多卡时合并主输出与各 rank 分片（crash 后分片里才有未合并的进度）。
    """
    existing_from_main = load_jsonl_records(out_path)
    if use_rank_shards:
        existing_records = merge_rank_outputs(
            out_path,
            world_size,
            existing_records=existing_from_main,
        )
    else:
        existing_records = existing_from_main

    collected_baseline = len(existing_records)
    start_idx = 0
    if existing_records:
        start_idx = (
            max(
                int(rec["idx"])
                for rec in existing_records
                if isinstance(rec.get("idx"), int)
            )
            + 1
        )
    return existing_records, collected_baseline, start_idx


# ----------------------------- data loading -----------------------------
def iter_json_array(path: Path) -> Iterator[Dict[str, Any]]:
    """流式读取顶层 JSON 数组，避免一次性加载大文件。"""
    if ijson is not None:
        with open(path, "rb") as f:
            yield from ijson.items(f, "item")
        return
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        yield from data


def iter_index_batches(
    iterator: Iterator[Dict[str, Any]],
    start_idx: int,
    batch_size: int,
) -> Iterator[List[Tuple[int, Dict[str, Any]]]]:
    """
    跳过 idx < start_idx 的样本（--resume 续跑），按 batch_size 打包成一个个“round”，
    供本地模型批量 generate 使用；多卡时每个 round 整体轮转分配给一个 rank
    （而不是逐条样本分配），round 之间各 rank 完全独立并行，无需逐条 barrier。
    """
    batch: List[Tuple[int, Dict[str, Any]]] = []
    for global_idx, ex in enumerate(iterator):
        if global_idx < start_idx:
            continue
        batch.append((global_idx, ex))
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


# ----------------------------- model loading -----------------------------
def load_local_model(
    checkpoint_dir: Path,
    *,
    local_rank: int,
    is_distributed: bool,
    log,
):
    adapter_cfg = json.loads((checkpoint_dir / "adapter_config.json").read_text(encoding="utf-8"))
    base_model_name = adapter_cfg["base_model_name_or_path"]
    log(f"[local] base model : {base_model_name}")
    log(f"[local] checkpoint : {checkpoint_dir}")

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir), use_fast=False)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    # 左 padding：batch generate 时保证每条样本的 prompt 结尾对齐在同一列，
    # 新生成的 token 才能用同一个切片位置（inputs["input_ids"].shape[1]）取出。
    tokenizer.padding_side = "left"

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device(f"cuda:{local_rank}" if is_distributed else "cuda")
    else:
        device = torch.device("cpu")
    dtype = torch.bfloat16 if use_cuda else None

    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=dtype,
            attn_implementation="flash_attention_2" if use_cuda else "eager",
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=dtype)

    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, str(checkpoint_dir))
    model.to(device)
    model.eval()
    log(f"[local] device(rank0 view): {device}")
    return tokenizer, model, device


@torch.no_grad()
def local_generate_batch(
    tokenizer, model, device, prompts: List[str], args
) -> List[str]:
    """批量 greedy generate。左 padding 后各样本的新生成 token 起始列一致，可统一切片。"""
    if not prompts:
        return []
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_input_length,
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_id
    output_ids = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        do_sample=False,
        use_cache=True,
        eos_token_id=int(eos_token_id) if eos_token_id is not None else None,
        pad_token_id=pad_token_id,
    )
    prompt_len = inputs["input_ids"].shape[1]
    new_ids = output_ids[:, prompt_len:]
    return [tokenizer.decode(seq, skip_special_tokens=True) for seq in new_ids]


# ----------------------------- args -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate DPO (accept/reject) pairs directly from a plain SFT checkpoint "
        "(accept = gold answer, no teacher/DeepSeek call needed)."
    )
    p.add_argument("--model", type=str, default=DEFAULT_MODEL,
                   help=f"本地 LoRA checkpoint 目录（默认 {DEFAULT_MODEL}，相对路径相对项目根）")
    p.add_argument("--input_data_file", type=str, default=DEFAULT_INPUT,
                   help=f"待推理的数据 JSON 数组文件（默认 {DEFAULT_INPUT}）")
    p.add_argument("--num", type=int, default=None,
                   help="收集到多少条推理错误（DPO pair）后停止；缺省则跑完整个数据集")
    p.add_argument("--out_path", type=str, default="",
                   help="输出 jsonl 路径（缺省 /DATA1/khli/t&m/rl_sft_{model}_{num}.jsonl）")
    p.add_argument(
        "--resume",
        action="store_true",
        help="从已有输出续跑；多卡时读取主 jsonl 与各 rank 分片 (.rank*.jsonl)",
    )

    # 本地模型生成参数（与 eval_checkpoint.py 对齐）
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--min_new_tokens", type=int, default=2)
    p.add_argument("--max_input_length", type=int, default=2560)

    # 批量生成 / 多卡分片参数
    p.add_argument("--gen_batch_size", type=int, default=8,
                   help="本地模型批量生成的 batch size，即每个 round 的样本数（默认 8）")
    p.add_argument("--sync_every_rounds", type=int, default=5,
                   help="多卡时每处理多少个 round 做一次同步 / --num 检查（默认 5）；"
                        "round 之间各 rank 完全独立并行，不再逐条样本 barrier")
    return p.parse_args()


def model_tag_from_path(model: str) -> str:
    """从 --model 路径提取用于文件名的标签（取最后一级目录名）。"""
    name = Path(model).name.strip()
    return name if name else "model"


def resolve_out_path(args: argparse.Namespace) -> Path:
    if args.out_path.strip():
        return Path(args.out_path).expanduser().resolve()
    model_tag = model_tag_from_path(args.model)
    num_tag = str(args.num) if args.num is not None else "all"
    return Path(f"/DATA1/khli/t&m/rl_sft_{model_tag}_{num_tag}.jsonl")


def should_stop_global(
    collected_baseline: int,
    local_new_pairs: int,
    target: Optional[int],
    *,
    device: torch.device,
    is_distributed: bool,
) -> bool:
    if target is None:
        return False
    global_new = all_reduce_sum(local_new_pairs, device, is_distributed)
    return collected_baseline + global_new >= target


def main() -> None:
    args = parse_args()
    if args.num is not None and args.num < 1:
        raise SystemExit("--num must be a positive integer")

    is_distributed, world_size, rank, local_rank = setup_dist()

    def log(msg: str) -> None:
        if rank == 0:
            print(msg)

    # ---- 本地模型 ----
    checkpoint_dir = Path(args.model)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = (_PROJECT_ROOT / checkpoint_dir).resolve()
    if not (checkpoint_dir / "adapter_config.json").is_file():
        raise SystemExit(f"adapter_config.json not found under: {checkpoint_dir}")

    input_path = Path(args.input_data_file).expanduser().resolve()
    if not input_path.is_file():
        raise SystemExit(f"Input data file not found: {input_path}")

    out_path = resolve_out_path(args)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"[dist] WORLD_SIZE={world_size}, rank={rank}, local_rank={local_rank}")

    target = args.num  # None = 跑完整个数据集
    use_rank_shards = is_distributed

    # ---- resume ----
    start_idx = 0
    collected_baseline = 0
    if args.resume:
        if rank == 0:
            _, collected_baseline, start_idx = compute_resume_state(
                out_path,
                world_size,
                use_rank_shards=use_rank_shards,
            )
        start_idx, collected_baseline = broadcast_resume_state(
            start_idx,
            collected_baseline,
            is_distributed=is_distributed,
            device=torch.device(f"cuda:{local_rank}" if is_distributed and torch.cuda.is_available() else "cpu"),
        )
        if rank == 0 and use_rank_shards:
            shard_lines = [
                f"rank{r}={len(load_jsonl_records(rank_shard_path(out_path, r)))}"
                for r in range(world_size)
            ]
            log(f"[resume] shard pairs: {', '.join(shard_lines)}")
        log(f"[resume] existing pairs={collected_baseline}, resume scanning from idx={start_idx}")
    elif rank == 0:
        if out_path.exists():
            print(f"Warning: {out_path} exists and will be overwritten (no --resume).", file=sys.stderr)
        if use_rank_shards:
            for r in range(world_size):
                shard = rank_shard_path(out_path, r)
                if shard.exists():
                    print(
                        f"Warning: {shard} exists and will be overwritten (no --resume).",
                        file=sys.stderr,
                    )

    tokenizer, model, device = load_local_model(
        checkpoint_dir,
        local_rank=local_rank,
        is_distributed=is_distributed,
        log=log,
    )

    write_path = rank_shard_path(out_path, rank) if use_rank_shards else out_path
    if use_rank_shards:
        if args.resume:
            out_mode = "a" if write_path.exists() else "w"
        else:
            write_path.unlink(missing_ok=True)
            out_mode = "w"
    else:
        out_mode = "a" if (args.resume and write_path.exists()) else "w"

    scanned = 0
    wrong_total = 0
    local_new_pairs = 0

    # 按 --gen_batch_size 打包成 round，多卡时整个 round 轮转分配给一个 rank
    # （而不是逐条样本分配），round 之间各 rank 完全独立并行、无需逐条 barrier。
    batch_iterator: Iterator[List[Tuple[int, Dict[str, Any]]]] = iter_index_batches(
        iter_json_array(input_path), start_idx, args.gen_batch_size
    )
    if tqdm is not None and rank == 0:
        batch_iterator = tqdm(batch_iterator, desc="gen-dpo4sft", unit="round")

    with open(write_path, out_mode, encoding="utf-8") as out_f:
        for round_idx, round_items in enumerate(batch_iterator):
            owns_round = (not use_rank_shards) or ((round_idx % world_size) == rank)

            if owns_round and round_items:
                scanned += len(round_items)

                ex_for_prompt_list = [
                    _prepare_ex_for_prompt(ex) for _, ex in round_items
                ]
                prompt_list = [build_prompt(ex) for ex in ex_for_prompt_list]
                raw_preds = local_generate_batch(tokenizer, model, device, prompt_list, args)

                for (global_idx, _), ex_for_prompt, prompt, raw_pred in zip(
                    round_items, ex_for_prompt_list, prompt_list, raw_preds
                ):
                    # 用规范化后的 question/output（"[DESC]"），跟喂给模型的 prompt 里的候选
                    # 格式保持一致，见 _prepare_ex_for_prompt 的说明。
                    question = ex_for_prompt.get("question", "")
                    pred_norm = _apply_candidate_constraint(question, raw_pred)
                    gold_raw = (ex_for_prompt.get("output") or "").strip()
                    gold_norm = _normalize_pred_text(gold_raw)

                    if not gold_norm or pred_norm == gold_norm:
                        continue

                    wrong_total += 1
                    reject_text = _strip_trailing_eos(raw_pred)

                    # 直接用数据里的原始 gold 文本作为 accept（此时已规范化为跟 prompt/question
                    # 里候选列表一致的 "[DESC]" 格式，与 SFT 训练时模型实际看到的标记保持一致）。
                    record = {
                        "idx": global_idx,
                        "prompt": prompt,
                        "accept": gold_raw,
                        "reject": reject_text,
                        "gold": gold_raw,
                        "model_prediction": pred_norm,
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_f.flush()
                    local_new_pairs += 1

            # ---- 多卡同步：不再逐条样本 barrier，而是每 sync_every_rounds 个 round
            # 才同步一次；期间各 rank 完全独立并行处理各自拥有的 round。----
            reached_sync_point = (round_idx + 1) % max(1, args.sync_every_rounds) == 0
            if is_distributed and reached_sync_point:
                dist_barrier(is_distributed=is_distributed, local_rank=local_rank)

            if target is not None and (reached_sync_point or not is_distributed):
                if should_stop_global(
                    collected_baseline,
                    local_new_pairs,
                    target,
                    device=device,
                    is_distributed=is_distributed,
                ):
                    log(f"[stop] reached target num={target}")
                    break

    if is_distributed:
        dist_barrier(is_distributed=is_distributed, local_rank=local_rank)

    # ---- rank0 汇总统计与合并输出 ----
    total_scanned = all_reduce_sum(scanned, device, is_distributed)
    total_wrong = all_reduce_sum(wrong_total, device, is_distributed)
    total_new_pairs = all_reduce_sum(local_new_pairs, device, is_distributed)
    total_pairs = collected_baseline + total_new_pairs

    if rank == 0:
        if use_rank_shards:
            merged = merge_rank_outputs(
                out_path,
                world_size,
                existing_records=load_jsonl_records(out_path),
                target_num=target,
            )
            with out_path.open("w", encoding="utf-8") as f:
                for row in merged:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            total_pairs = len(merged)

        print("=" * 80)
        print(f"Scanned (this run): {total_scanned}")
        print(f"Wrong predictions : {total_wrong}")
        print(f"DPO pairs total   : {total_pairs}")
        print(f"Saved to          : {out_path}")
        if use_rank_shards:
            print(f"Rank shards       : {out_path.stem}.rank*{out_path.suffix}")

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
