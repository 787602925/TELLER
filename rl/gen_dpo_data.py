#!/usr/bin/env python3
"""
RL (DPO) 训练数据生成脚本。

流程：
  1) 用当前阶段的本地模型（SFT + CoT-SFT 后的 LoRA checkpoint）对 --input_data_file
     里的数据做推理，推理逻辑与 el/eval_CoT_checkpoint.py 基本一致
     （_apply_eval_table_compression -> build_prompt -> greedy generate ->
     _apply_cot_candidate_constraint / _normalize_pred_text），但本地生成按
     --gen_batch_size 做批量 batch generate（而不是逐条 batch=1）。
  2) 找出 prediction（去掉结尾 </s>）与 gold 不匹配的样本，取其 raw_prediction 作为 reject。
  3) 把这些“推理错误”的原始样本加工 prompt（把正确答案作为内部校验信息注入，但要求
     DeepSeek 的 <think> 推理不得引用/暗示该答案），交给 deepseek-v4-pro 生成
     “正确答案 + <think>”，作为 accept。这一步通过线程池（--teacher_workers）异步调用，
     与本地 GPU 生成解耦、并行执行，不阻塞下一批样本的推理。
  4) 组成 DPO pair 写入 /DATA1/khli/t&m/rl_{model}_{num}.jsonl：
       {"prompt": ..., "accept": "<think>...</think>\\n<entity>", "reject": "<think>...</think>\\n<entity>"}

多卡（torchrun）时按“round”（每 round = --gen_batch_size 条样本）轮转分配给各 rank，
round 之间各 rank 完全独立并行处理，只在每 --sync_every_rounds 个 round 做一次同步，
不再是逐条样本 barrier（逐条 barrier 会让多卡退化成事实上的单卡串行）。

用法示例：
  conda activate tablellama-fa
  export DEEPSEEK_API_KEY=...
  python rl/gen_dpo_data.py --num 200
  python rl/gen_dpo_data.py --model result_CoT/checkpoint-1000 \\
      --input_data_file /DATA1/khli/t&m/merged_SFT_70k&35k.json --num 500
  python rl/gen_dpo_data.py --num 200 --resume   # 断点续跑（多卡时从 .rank*.jsonl 分片恢复）

  # 多 GPU（每卡 1 进程，与 eval_CoT_checkpoint.py 一致）：
  torchrun --nproc_per_node=2 rl/gen_dpo_data.py --num 200
  torchrun --nproc_per_node=2 rl/gen_dpo_data.py --num 200 --resume  # 续跑读主文件 + rank 分片
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
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

# 复用现有推理 / 评估逻辑（与 el/eval_CoT_checkpoint.py 一致）
from el.sft import (
    _apply_cot_candidate_constraint,
    _apply_eval_table_compression,
    _normalize_pred_text,
)

# 复用 DeepSeek 调用 / 解析 / 清洗工具
from data_aug.augment_ent_link_thinking import (
    DEFAULT_API_KEY_ENV_BY_PROVIDER,
    DEFAULT_BASE_URL,
    DEFAULT_RWTHGPT_BASE_URL,
    MODEL_CONFIGS,
    RATE_LIMIT_WAIT_MODELS,
    build_prompt,
    call_model,
    ensure_angle_brackets,
    extract_entity_from_text,
    format_augmented_output,
    resolve_model_config,
    split_reasoning_and_answer_for_gpt52,
    split_reasoning_and_answer_from_content,
)

DEFAULT_MODEL = "result_CoT_filtered_compressed_data_lora/checkpoint-350"
DEFAULT_INPUT = "/DATA1/khli/t&m/merged_SFT_70k&35k.json"
DEFAULT_REFERENCE_ACCEPT_POOL = "/DATA1/khli/t&m/merged_CoTSFT_all&all_filtered_cot_compressed.json"
DEFAULT_TEACHER_VARIANT = "pro"  # deepseek-v4-pro


def _strip_trailing_eos(text: str) -> str:
    """去掉 decode 结果末尾的 </s> 等 EOS 残留，保留完整 CoT 生成作为 reject。"""
    if not isinstance(text, str):
        return ""
    return re.sub(r"</?s>\s*$", "", text.strip(), flags=re.I).strip()


_CANDIDATE_TEMPLATE_RE = re.compile(r"<[^<>]+?\[DESCRIPTION\][^<>]*?>")
_BULLET_LINE_RE = re.compile(r"^[-*•]\s")


def _is_candidate_dump_line(line: str) -> bool:
    """
    形如 "- <A [DESCRIPTION] .. [TYPE] ..>" 的候选枚举行：信息量低（只是在抄写
    候选列表，不是理由），句数截断时优先跳过，把有限的句数配额留给真正解释推理
    过程的句子，而不是被这类行占满。
    """
    stripped = line.strip()
    return bool(stripped and _BULLET_LINE_RE.match(stripped) and _CANDIDATE_TEMPLATE_RE.search(stripped))


def _truncate_reasoning(
    reasoning: str,
    *,
    max_sentences: int,
    max_chars: int,
) -> str:
    """
    硬性截短一段 <think> reasoning：防止 accept 系统性比 reject（本地模型原始生成）
    长很多，导致 DPO 学到“越长越好”的偏置、推理时生成不完就被截断。
    同时用于两个 accept 来源：deepseek-v4-pro teacher 现场生成、reference pool
    缓存命中（实测发现命中的 CoT-SFT 缓存 reasoning 同样偏长，是长度失衡的主因之一）。

    先按句子数截断（按行再按句末标点断句，因为候选枚举等 list 式文本常常整行没有
    句末标点，若不按行先切分，会被当成一句超长“句子”，让句数截断形同虚设）；
    再按字符数兜底截断，且回退到最近的句末标点/空白处，避免把单词或候选模板
    "<...[DESCRIPTION]...[TYPE]...>" 切到中间，产出不完整的字符残片。
    """
    text = (reasoning or "").strip()
    if not text:
        return text

    if max_sentences > 0:
        chunks = [
            piece.strip()
            for line in text.split("\n")
            for piece in re.split(r"(?<=[.!?。！？])\s+", line)
            if piece.strip()
        ]
        if len(chunks) > max_sentences:
            substantive = [c for c in chunks if not _is_candidate_dump_line(c)]
            kept = substantive if substantive else chunks
            text = "\n".join(kept[:max_sentences]).strip()

    if max_chars > 0 and len(text) > max_chars:
        window = text[:max_chars]
        end_marks = list(re.finditer(r"[.!?。！？]", window))
        if end_marks:
            text = window[: end_marks[-1].end()].rstrip()
        else:
            boundary = max(window.rfind(" "), window.rfind("\n"))
            text = (window[:boundary] if boundary > 0 else window).rstrip()

    return text


def shorten_accept_text(
    accept_text: str,
    *,
    max_sentences: int,
    max_chars: int,
) -> str:
    """
    对完整的 accept 文本（"<think>reasoning</think>\\nanswer" 或纯 answer）做统一
    收短处理：解析出 reasoning/answer，只截短 reasoning，answer 保持原样（避免破坏
    候选实体格式导致训练目标出错）。若解析不到 <think> 块（纯 answer，没有 reasoning
    可截），原样返回。
    """
    reasoning, answer = split_reasoning_and_answer_from_content(accept_text)
    if not reasoning or not answer:
        return accept_text
    short_reasoning = _truncate_reasoning(
        reasoning, max_sentences=max_sentences, max_chars=max_chars
    )
    return format_augmented_output(short_reasoning, answer)


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


# ----------------------------- DeepSeek teacher prompt -----------------------------
def build_teacher_prompt(
    ex: Dict[str, Any],
    gold: str,
    *,
    max_reasoning_sentences: int = 3,
) -> str:
    """
    在原始 EL 题面基础上注入“仅供内部校验、绝不可引用”的正确答案，
    要求 deepseek-v4-pro 像没人给过答案一样，从证据独立推出该答案。

    max_reasoning_sentences 用来把“简洁”要求量化成硬性句数上限，目的是让 teacher
    生成的 accept（<think>reasoning</think>+answer）不再系统性比本地模型的 reject
    （原始生成）长很多；即便模型没完全遵守，后面也会用 _truncate_reasoning
    再做一次硬截断兜底。
    """
    base = build_prompt(ex)  # 与本地模型完全一致的题面，结尾是 "### Response:"
    return (
        f"{base}\n\n"
        "=== SOLUTION CHECK (internal only — NEVER reference in your output) ===\n"
        "The verified correct referent entity for this task is:\n"
        f"{gold}\n"
        "=== END SOLUTION CHECK ===\n\n"
        "You are an expert entity-linking solver. Solve the task above AS IF no answer had "
        "ever been given to you. Treat the table context and the candidates' "
        "[DESCRIPTION]/[TYPE] fields as your ONLY evidence.\n\n"
        "Answer in EXACTLY this format (nothing before <think>):\n"
        "<think>\n<your concise, genuine reasoning>\n</think>\n"
        "<one candidate-format entity>\n\n"
        "Hard rules for the <think> reasoning:\n"
        "1) Derive the answer independently from evidence: page/section/caption, the mention's "
        "row/column context, and the candidates' [DESCRIPTION]/[TYPE].\n"
        "2) NEVER mention, hint, or imply that a correct/gold/provided/verified answer was given "
        "to you. Forbidden words/phrases include (non-exhaustive): 'gold', 'provided answer', "
        "'verified', 'correct answer is', 'given answer', 'since the answer', 'we are told', "
        "'标准答案', '题目给出', '已知答案'.\n"
        "3) Do NOT restate the whole table or copy the candidate list; cite only the minimal "
        "distinguishing evidence that separates the right candidate from close distractors.\n"
        f"4) BE EXTREMELY CONCISE: at most {max_reasoning_sentences} short sentences in total, "
        "no filler/hedging, no restating the question or the full candidate list — every sentence "
        "must carry distinguishing evidence and directly lead to the final entity.\n"
        "5) The final entity after </think> MUST be exactly one candidate and MUST equal the "
        "verified correct entity (in candidate format).\n"
    )


def teacher_reasoning_and_answer(
    client: Any,
    ex: Dict[str, Any],
    gold: str,
    *,
    model_variant: str,
    model_id: str,
    provider: str,
    max_tokens: int,
    reasoning_effort: str,
    temperature: float,
    max_reasoning_sentences: int,
) -> tuple[str, str]:
    """调用 DeepSeek 并解析出 (reasoning, answer)。"""
    prompt = build_teacher_prompt(ex, gold, max_reasoning_sentences=max_reasoning_sentences)
    reasoning, content = call_model(
        client,
        prompt,
        model_variant=model_variant,
        model=model_id,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        temperature=temperature,
        enable_deepseek_thinking=(provider == "deepseek"),
        enable_rate_limit_wait=(model_variant in RATE_LIMIT_WAIT_MODELS),
    )
    content_reasoning, content_answer = split_reasoning_and_answer_from_content(content)
    if not reasoning and content_reasoning:
        reasoning = content_reasoning
    answer_source = content_answer if content_answer else content
    if model_variant == "gpt-5.2":
        gpt52_reasoning, gpt52_answer = split_reasoning_and_answer_for_gpt52(content)
        if not reasoning and gpt52_reasoning:
            reasoning = gpt52_reasoning
        if gpt52_answer:
            answer_source = gpt52_answer
    answer = extract_entity_from_text(answer_source)
    return reasoning, answer


def teacher_worker_task(
    client: Any,
    ex_for_prompt: Dict[str, Any],
    gold_raw: str,
    gold_norm: str,
    *,
    model_variant: str,
    model_id: str,
    provider: str,
    max_tokens: int,
    reasoning_effort: str,
    temperature: float,
    retries: int,
    sleep_s: float,
    max_reasoning_sentences: int,
    max_reasoning_chars: int,
) -> Dict[str, Any]:
    """
    在独立线程中运行：对一个“本地模型推理错误”的样本调用 teacher API（含重试），
    只返回结果、不修改任何跨线程共享状态，避免多线程计数竞争；由主线程统一汇总统计。
    与本地 GPU 批量生成解耦并行执行，不阻塞下一个 round 的推理。
    """
    calls = 0
    matched = False
    reasoning, answer = "", ""
    last_error = ""
    for _ in range(max(1, retries + 1)):
        calls += 1
        try:
            reasoning, answer = teacher_reasoning_and_answer(
                client, ex_for_prompt, gold_raw,
                model_variant=model_variant,
                model_id=model_id,
                provider=provider,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
                temperature=temperature,
                max_reasoning_sentences=max_reasoning_sentences,
            )
        except Exception as e:  # noqa: BLE001
            last_error = str(e)
            reasoning, answer = "", ""
        if reasoning.strip() and _normalize_pred_text(ensure_angle_brackets(answer)) == gold_norm:
            matched = True
            break
        if sleep_s > 0:
            time.sleep(sleep_s)

    # 硬截断兜底：即便 teacher 没完全遵守 prompt 里的句数限制，也强制把 accept 的
    # reasoning 压短，避免 accept 系统性比 reject（本地模型原始生成）长很多。
    short_reasoning = _truncate_reasoning(
        reasoning,
        max_sentences=max_reasoning_sentences,
        max_chars=max_reasoning_chars,
    )
    accept_text = (
        format_augmented_output(short_reasoning, ensure_angle_brackets(gold_raw))
        if matched
        else ""
    )
    return {
        "matched": matched,
        "accept_text": accept_text,
        "calls": calls,
        "error": "" if matched else last_error,
    }


def drain_pending(
    pending: List[Dict[str, Any]],
    out_f,
    *,
    block: bool,
    rank: int,
) -> Dict[str, int]:
    """
    收割 pending 中的 teacher future。block=False 时只收割已完成的（非阻塞，随手清理，
    避免 pending 无限堆积）；block=True 时等待全部完成（用于收尾，保证一条不丢）。
    命中的样本立即写入 out_f 并 flush，写入格式与之前完全一致（便于 --resume 续跑兼容）。
    """
    stats = {"new_pairs": 0, "teacher_fail": 0, "teacher_api_calls": 0}
    still_pending: List[Dict[str, Any]] = []
    for item in pending:
        future = item["future"]
        if not block and not future.done():
            still_pending.append(item)
            continue
        result = future.result()
        stats["teacher_api_calls"] += result["calls"]
        if result["matched"]:
            record = {
                "idx": item["idx"],
                "prompt": item["prompt"],
                "accept": result["accept_text"],
                "reject": item["reject_text"],
                "gold": item["gold_raw"],
                "model_prediction": item["pred_norm"],
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            stats["new_pairs"] += 1
        else:
            stats["teacher_fail"] += 1
            if result["error"]:
                print(
                    f"[teacher-error] rank={rank} idx={item['idx']}: {result['error']}",
                    file=sys.stderr,
                )
    pending[:] = still_pending
    return stats


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


def _norm_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip()
    return str(v).strip()


def build_example_key(ex: Dict[str, Any]) -> str:
    """
    用 instruction + input_seg(+fallback input) + question 作为“原样本”匹配键。
    """
    instruction = _norm_text(ex.get("instruction"))
    input_seg = _norm_text(ex.get("input_seg", ex.get("input", "")))
    question = _norm_text(ex.get("question"))
    return "\u241f".join((instruction, input_seg, question))


def load_reference_pool(path: Path) -> Dict[str, list[str]]:
    """
    加载可复用 accept 池（键: 样本键，值: 该样本可选 output 列表）。
    同一键可能存在多条记录（例如重复样本或多版本输出），故保留列表。
    """
    pool: Dict[str, list[str]] = {}
    for ex in iter_json_array(path):
        key = build_example_key(ex)
        out = _norm_text(ex.get("output"))
        if not key or not out:
            continue
        if key not in pool:
            pool[key] = [out]
        else:
            pool[key].append(out)
    return pool


def last_processed_idx(path: Path) -> int:
    """读取已有 jsonl 中最大的 idx，用于 --resume 续跑。"""
    if not path.exists():
        return -1
    last = -1
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            idx = rec.get("idx")
            if isinstance(idx, int) and idx > last:
                last = idx
    return last


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
    p = argparse.ArgumentParser(description="Generate DPO (accept/reject) pairs for the RL stage.")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL,
                   help=f"本地 LoRA checkpoint 目录（默认 {DEFAULT_MODEL}，相对路径相对项目根）")
    p.add_argument("--input_data_file", type=str, default=DEFAULT_INPUT,
                   help=f"待推理的数据 JSON 数组文件（默认 {DEFAULT_INPUT}）")
    p.add_argument("--num", type=int, default=None,
                   help="收集到多少条推理错误（DPO pair）后停止；缺省则跑完整个数据集")
    p.add_argument("--out_path", type=str, default="",
                   help="输出 jsonl 路径（缺省 /DATA1/khli/t&m/rl_{model}_{num}.jsonl）")
    p.add_argument(
        "--resume",
        action="store_true",
        help="从已有输出续跑；多卡时读取主 jsonl 与各 rank 分片 (.rank*.jsonl)",
    )
    p.add_argument(
        "--reference_accept_file",
        type=str,
        default=DEFAULT_REFERENCE_ACCEPT_POOL,
        help=(
            "本地可复用 accept 数据池文件。若命中同一原样本且该 output 与 gold 匹配，"
            "则直接复用、跳过 teacher API（默认 merged_CoTSFT_all&all_filtered_cot_compressed.json，"
            "即 CoT-SFT 训练实际使用的过滤+压缩后版本，保证缓存命中的 accept 本身干净、够短）"
        ),
    )

    # 本地模型生成参数（与 eval_CoT_checkpoint.py 对齐）
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--min_new_tokens", type=int, default=2)
    p.add_argument("--max_input_length", type=int, default=2560)

    # 批量生成 / 并发 / 多卡分片参数
    p.add_argument("--gen_batch_size", type=int, default=8,
                   help="本地模型批量生成的 batch size，即每个 round 的样本数（默认 8）")
    p.add_argument("--teacher_workers", type=int, default=4,
                   help="每个 rank 并发调用 teacher API 的线程数（默认 4），"
                        "与本地 GPU 批量生成解耦、异步并行执行")
    p.add_argument("--sync_every_rounds", type=int, default=5,
                   help="多卡时每处理多少个 round 做一次同步 / --num 检查（默认 5）；"
                        "round 之间各 rank 完全独立并行，不再逐条样本 barrier")

    # DeepSeek teacher 参数
    p.add_argument("--teacher_model", type=str, default=DEFAULT_TEACHER_VARIANT,
                   choices=tuple(MODEL_CONFIGS.keys()),
                   help="teacher 模型 variant（默认 pro -> deepseek-v4-pro）")
    p.add_argument("--teacher_base_url", type=str, default="")
    p.add_argument("--teacher_api_key_env", type=str, default="")
    p.add_argument("--teacher_max_tokens", type=int, default=8192)
    p.add_argument("--teacher_reasoning_effort", type=str, default="high",
                   choices=("low", "high", "none"))
    p.add_argument("--teacher_temperature", type=float, default=0.6)
    p.add_argument("--teacher_retries", type=int, default=2,
                   help="teacher 最终实体与 gold 不一致时的重试次数")
    p.add_argument("--accept_max_reasoning_sentences", type=int, default=3,
                   help="accept 的 <think> reasoning 最多保留的句数（默认 3）。"
                        "对 teacher(deepseek-v4-pro) 现场生成的 accept：既写进 prompt 引导模型少写，"
                        "也用于生成后硬截断；对 reference pool 缓存命中的 accept：直接硬截断"
                        "（实测发现缓存里的 reasoning 同样明显偏长）。目的是让 accept 不再"
                        "系统性比 reject（本地模型原始生成）长很多")
    p.add_argument("--accept_max_reasoning_chars", type=int, default=480,
                   help="accept 的 <think> reasoning 硬截断的字符数上限"
                        "（句数截断后仍过长时的兜底，对 teacher 生成和 reference pool 缓存"
                        "命中的 accept 都生效，默认 480）")
    p.add_argument("--sleep", type=float, default=0.0, help="每次 teacher 调用后 sleep 秒数")
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
    return Path(f"/DATA1/khli/t&m/rl_{model_tag}_{num_tag}.jsonl")


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
    reference_path = Path(args.reference_accept_file).expanduser().resolve()
    if not reference_path.is_file():
        raise SystemExit(f"Reference accept file not found: {reference_path}")

    # ---- DeepSeek teacher ----
    teacher_cfg = resolve_model_config(args.teacher_model)
    provider = teacher_cfg["provider"]
    model_id = teacher_cfg["model_id"]
    if args.teacher_base_url.strip():
        base_url = args.teacher_base_url.strip().rstrip("/")
    else:
        base_url = DEFAULT_BASE_URL if provider == "deepseek" else DEFAULT_RWTHGPT_BASE_URL
    api_key_env = (
        args.teacher_api_key_env.strip()
        if args.teacher_api_key_env.strip()
        else DEFAULT_API_KEY_ENV_BY_PROVIDER[provider]
    )
    api_key = os.environ.get(api_key_env, "").strip()
    if not api_key:
        raise SystemExit(f"Please set environment variable {api_key_env}")
    try:
        from openai import OpenAI
    except ImportError as e:
        raise SystemExit("openai package is required: pip install openai") from e
    client = OpenAI(api_key=api_key, base_url=base_url)
    log(
        f"[teacher] --teacher_model {args.teacher_model} -> {model_id} "
        f"(provider={provider}, base_url={base_url}, api_key_env={api_key_env})"
    )

    out_path = resolve_out_path(args)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"[dist] WORLD_SIZE={world_size}, rank={rank}, local_rank={local_rank}")

    log(f"[reference] loading pool from: {reference_path}")
    reference_pool = load_reference_pool(reference_path)
    log(f"[reference] loaded keys={len(reference_pool)} (file={reference_path.name})")

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
    reference_hit_total = 0
    reference_hit_correct = 0
    reference_hit_wrong = 0
    teacher_fail = 0
    teacher_api_calls_total = 0
    local_new_pairs = 0

    # 按 --gen_batch_size 打包成 round，多卡时整个 round 轮转分配给一个 rank
    # （而不是逐条样本分配），round 之间各 rank 完全独立并行、无需逐条 barrier。
    batch_iterator: Iterator[List[Tuple[int, Dict[str, Any]]]] = iter_index_batches(
        iter_json_array(input_path), start_idx, args.gen_batch_size
    )
    if tqdm is not None and rank == 0:
        batch_iterator = tqdm(batch_iterator, desc="gen-dpo", unit="round")

    executor = ThreadPoolExecutor(
        max_workers=max(1, args.teacher_workers), thread_name_prefix="teacher"
    )
    pending: List[Dict[str, Any]] = []

    with open(write_path, out_mode, encoding="utf-8") as out_f:
        for round_idx, round_items in enumerate(batch_iterator):
            owns_round = (not use_rank_shards) or ((round_idx % world_size) == rank)

            if owns_round and round_items:
                scanned += len(round_items)

                ex_for_prompt_list = [
                    _apply_eval_table_compression(ex) for _, ex in round_items
                ]
                prompt_list = [build_prompt(ex) for ex in ex_for_prompt_list]
                raw_preds = local_generate_batch(tokenizer, model, device, prompt_list, args)

                for (global_idx, ex), ex_for_prompt, prompt, raw_pred in zip(
                    round_items, ex_for_prompt_list, prompt_list, raw_preds
                ):
                    question = ex.get("question", "")
                    pred_norm = _apply_cot_candidate_constraint(question, raw_pred)
                    gold_raw = (ex.get("output") or "").strip()
                    gold_norm = _normalize_pred_text(gold_raw)

                    if not gold_norm or pred_norm == gold_norm:
                        continue

                    wrong_total += 1
                    reject_text = _strip_trailing_eos(raw_pred)

                    # ---- 先查 reference pool：命中且正确则直接复用，不调 API ----
                    accept_text = ""
                    ex_key = build_example_key(ex)
                    cached_outputs = reference_pool.get(ex_key, [])
                    if cached_outputs:
                        reference_hit_total += 1
                        for cached_out in cached_outputs:
                            cached_final = _apply_cot_candidate_constraint(question, cached_out)
                            if cached_final == gold_norm:
                                # 实测发现 reference pool 缓存里的 reasoning 同样明显
                                # 偏长（甚至比 teacher 现场生成的更长），同样做硬截断，
                                # 避免这条“复用缓存”路径重新引入 accept 系统性偏长的问题。
                                accept_text = shorten_accept_text(
                                    cached_out,
                                    max_sentences=args.accept_max_reasoning_sentences,
                                    max_chars=args.accept_max_reasoning_chars,
                                )
                                reference_hit_correct += 1
                                break
                        if not accept_text:
                            reference_hit_wrong += 1

                    if accept_text:
                        record = {
                            "idx": global_idx,
                            "prompt": prompt,
                            "accept": accept_text,
                            "reject": reject_text,
                            "gold": gold_raw,
                            "model_prediction": pred_norm,
                        }
                        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        out_f.flush()
                        local_new_pairs += 1
                        continue

                    # ---- reference 不可用时，异步交给线程池调用 teacher API，
                    # 不阻塞本 rank 对下一个 round 的本地 GPU 生成 ----
                    future = executor.submit(
                        teacher_worker_task,
                        client, ex_for_prompt, gold_raw, gold_norm,
                        model_variant=teacher_cfg["variant"],
                        model_id=model_id,
                        provider=provider,
                        max_tokens=args.teacher_max_tokens,
                        reasoning_effort=args.teacher_reasoning_effort,
                        temperature=args.teacher_temperature,
                        retries=args.teacher_retries,
                        sleep_s=args.sleep,
                        max_reasoning_sentences=args.accept_max_reasoning_sentences,
                        max_reasoning_chars=args.accept_max_reasoning_chars,
                    )
                    pending.append({
                        "future": future,
                        "idx": global_idx,
                        "prompt": prompt,
                        "reject_text": reject_text,
                        "gold_raw": gold_raw,
                        "pred_norm": pred_norm,
                    })

                # 顺手（非阻塞）收割已完成的 teacher 结果，避免 pending 无限堆积，
                # 同时让输出尽早落盘。
                delta = drain_pending(pending, out_f, block=False, rank=rank)
                local_new_pairs += delta["new_pairs"]
                teacher_fail += delta["teacher_fail"]
                teacher_api_calls_total += delta["teacher_api_calls"]

                if args.sleep > 0:
                    time.sleep(args.sleep)

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

        # ---- 收尾：阻塞等待所有还没跑完的 teacher 请求，全部落盘，一条不丢 ----
        delta = drain_pending(pending, out_f, block=True, rank=rank)
        local_new_pairs += delta["new_pairs"]
        teacher_fail += delta["teacher_fail"]
        teacher_api_calls_total += delta["teacher_api_calls"]

    executor.shutdown(wait=True)

    if is_distributed:
        dist_barrier(is_distributed=is_distributed, local_rank=local_rank)

    # ---- rank0 汇总统计与合并输出 ----
    total_scanned = all_reduce_sum(scanned, device, is_distributed)
    total_wrong = all_reduce_sum(wrong_total, device, is_distributed)
    total_ref_hit = all_reduce_sum(reference_hit_total, device, is_distributed)
    total_ref_hit_correct = all_reduce_sum(reference_hit_correct, device, is_distributed)
    total_ref_hit_wrong = all_reduce_sum(reference_hit_wrong, device, is_distributed)
    total_teacher_fail = all_reduce_sum(teacher_fail, device, is_distributed)
    total_teacher_api_calls = all_reduce_sum(teacher_api_calls_total, device, is_distributed)
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
        print(
            f"Reference hits    : total={total_ref_hit}, "
            f"correct={total_ref_hit_correct}, wrong={total_ref_hit_wrong}"
        )
        print(f"teacher_api_calls_total: {total_teacher_api_calls}")
        print(f"Teacher failures  : {total_teacher_fail}")
        print(f"DPO pairs total   : {total_pairs}")
        print(f"Saved to          : {out_path}")
        if use_rank_shards:
            print(f"Rank shards       : {out_path.stem}.rank*{out_path.suffix}")

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
