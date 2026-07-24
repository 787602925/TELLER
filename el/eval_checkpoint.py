import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
import transformers
from peft import PeftModel
from tqdm import tqdm

from sft import (
    _apply_candidate_constraint,
    _apply_eval_table_compression,
    _load_eval_records,
    _normalize_description_marker,
    _normalize_pred_text,
    build_prompt,
)


def _str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    v = value.lower()
    if v in {"true", "1", "yes", "y"}:
        return True
    if v in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def _load_base_model_name(checkpoint_dir: Path, fallback: Optional[str]) -> str:
    cfg_path = checkpoint_dir / "adapter_config.json"
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        base = cfg.get("base_model_name_or_path")
        if base:
            return str(base)
    if fallback:
        return fallback
    raise ValueError(
        "Cannot infer base model from adapter_config.json. "
        "Please provide --base_model_name_or_path."
    )


def _init_distributed() -> Dict[str, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_distributed = world_size > 1

    if is_distributed and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")

    return {
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "is_distributed": int(is_distributed),
    }


def _cleanup_distributed(is_distributed: bool):
    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()


@torch.no_grad()
def run_eval(
    checkpoint_dir: Path,
    eval_data_path: str,
    output_dir: Path,
    step: int,
    model_max_length: int = 2048,
    max_new_tokens: int = 128,
    eval_data_limit: Optional[int] = None,
    use_flash_attn: bool = True,
    bf16: bool = True,
    base_model_name_or_path: Optional[str] = None,
):
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    dist_info = _init_distributed()
    rank = dist_info["rank"]
    world_size = dist_info["world_size"]
    local_rank = dist_info["local_rank"]
    is_distributed = bool(dist_info["is_distributed"])

    base_model_name = _load_base_model_name(checkpoint_dir, base_model_name_or_path)

    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            str(checkpoint_dir),
            model_max_length=model_max_length,
            padding_side="right",
            use_fast=False,
        )
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

        use_cuda = torch.cuda.is_available()
        if use_cuda and is_distributed:
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cuda:0" if use_cuda else "cpu")
        dtype = torch.bfloat16 if (bf16 and use_cuda) else None
        attn_impl = "flash_attention_2" if (use_flash_attn and use_cuda) else "eager"

        model = transformers.AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=dtype,
            attn_implementation=attn_impl,
        )
        # Match embedding shapes with training-time resized tokenizer before loading adapter.
        model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(model, str(checkpoint_dir))
        model.to(device)
        model.eval()

        eval_records = _load_eval_records(eval_data_path, limit=eval_data_limit)
        eval_records = [_apply_eval_table_compression(ex) for ex in eval_records]
        # 与 sft.py SupervisedDataset 的预处理顺序一致（先压缩表格，再规范化候选实体标记），
        # 否则这里喂给模型的 prompt 里候选实体标记是训练时从未见过的 "[DESCRIPTION]"，
        # 测出来的准确率跟训练分布不一致。
        for ex in eval_records:
            for key in ("question", "output"):
                if key in ex:
                    ex[key] = _normalize_description_marker(ex[key])
        indexed_records = list(enumerate(eval_records))
        local_records = indexed_records[rank::world_size]

        eos_token_id = tokenizer.eos_token_id
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_id

        rows: List[Dict[str, Any]] = []
        local_desc = f"ValGenEval step {step} rank {rank}/{world_size}"
        for ex_idx, ex in tqdm(local_records, desc=local_desc, ncols=120):
            prompt = build_prompt(ex)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=False).to(device)
            seq_len = inputs["input_ids"].shape[1]
            max_pos = getattr(model.config, "max_position_embeddings", 131072)
            if seq_len + max_new_tokens > max_pos:
                keep = max_pos - max_new_tokens
                inputs = {k: v[:, -keep:] for k, v in inputs.items()}

            autocast_ctx = (
                torch.autocast(device_type=device.type, dtype=torch.bfloat16)
                if (device.type in {"cuda", "cpu"} and bf16)
                else torch.no_grad()
            )
            with autocast_ctx:
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=2,
                    do_sample=False,
                    use_cache=True,
                    eos_token_id=int(eos_token_id) if eos_token_id is not None else None,
                    pad_token_id=pad_token_id,
                )

            new_ids = output_ids[0, inputs["input_ids"].shape[1]:]
            raw_pred = tokenizer.decode(new_ids, skip_special_tokens=True)
            prediction = _apply_candidate_constraint(ex.get("question", ""), raw_pred)
            gold = _normalize_pred_text(ex.get("output") or "")

            rows.append(
                {
                    "_example_idx": ex_idx,
                    "table": ex.get("table"),
                    "cell": ex.get("cell"),
                    "prompt": prompt,
                    "gold": gold,
                    "raw_prediction": raw_pred,
                    "prediction": prediction,
                    "correct": int(prediction == gold),
                }
            )

        if is_distributed:
            gathered_rows: List[List[Dict[str, Any]]] = [None for _ in range(world_size)]  # type: ignore
            dist.all_gather_object(gathered_rows, rows)
            all_rows = [row for shard in gathered_rows for row in shard]
        else:
            all_rows = rows

        if rank == 0:
            all_rows.sort(key=lambda r: r["_example_idx"])
            for row in all_rows:
                row.pop("_example_idx", None)

            output_dir.mkdir(parents=True, exist_ok=True)
            pred_path = output_dir / f"val_predictions_step_{step:06d}.jsonl"
            with pred_path.open("w", encoding="utf-8") as f:
                for row in all_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

            n_total = len(all_rows)
            n_correct = sum(r["correct"] for r in all_rows)
            acc = (n_correct / n_total) if n_total else 0.0

            metrics_path = output_dir / f"val_metrics_step_{step:06d}.json"
            with metrics_path.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "step": step,
                        "accuracy": acc,
                        "n_correct": n_correct,
                        "n_total": n_total,
                        "predictions_file": str(pred_path),
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            print(f"[ValGenEval step {step}] accuracy={acc:.4f} ({n_correct}/{n_total})")
            print(f"[ValGenEval step {step}] saved predictions -> {pred_path}")
            print(f"[ValGenEval step {step}] saved metrics -> {metrics_path}")
    finally:
        _cleanup_distributed(is_distributed)


def main():
    parser = argparse.ArgumentParser(description="Run offline val generation eval from a saved PEFT checkpoint.")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--eval_data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="result_mammo/val_eval_offline")
    parser.add_argument("--step", type=int, default=17484)
    parser.add_argument("--model_max_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--eval_data_limit", type=int, default=None)
    parser.add_argument("--use_flash_attn", type=_str2bool, default=True)
    parser.add_argument("--bf16", type=_str2bool, default=True)
    parser.add_argument("--base_model_name_or_path", type=str, default=None)
    args = parser.parse_args()

    run_eval(
        checkpoint_dir=Path(args.checkpoint_dir),
        eval_data_path=os.path.expanduser(args.eval_data_path),
        output_dir=Path(args.output_dir),
        step=args.step,
        model_max_length=args.model_max_length,
        max_new_tokens=args.max_new_tokens,
        eval_data_limit=args.eval_data_limit,
        use_flash_attn=args.use_flash_attn,
        bf16=args.bf16,
        base_model_name_or_path=args.base_model_name_or_path,
    )


if __name__ == "__main__":
    main()
