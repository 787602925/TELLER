#!/usr/bin/env python3
"""
Evaluate first N samples from ent_link_test_simplified.json
using a local LoRA checkpoint (no vLLM, no multiple backends).
"""

from __future__ import annotations

import argparse
import json
import re
import signal
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from peft import AutoPeftModelForCausalLM
from tqdm import tqdm
from transformers import AutoTokenizer

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import metrics as metrics_mod


DEFAULT_DATA = Path("/DATA1/khli/tablellama/ent_link_test_simplified.json")
DEFAULT_CHECKPOINT = Path("/home/khli/tableLlama/result/checkpoint-3030")
DEFAULT_LIMIT = 200
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_TIMEOUT_SEC = 120
DEFAULT_SAVE_DIR = _ROOT / "result" / "eval_simplified"


class SampleTimeoutError(TimeoutError):
    pass


def load_data(path: Path, limit: int) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}")
    return data[:limit]


def build_prompt(ex: Dict[str, Any]) -> str:
    input_seg = ex.get("input_seg", ex.get("input", ""))
    question = ex.get("question", "")
    return (
        "### Instruction:\n"
        "entity linking task. choose only the correct one from the referent entity candidates. "
        "In the Input below, the table content only contains the caption (if any), all column headers, "
        "and the cells from the same row and same column as the selected entity mention.\n\n"
        "### Input:\n"
        f"{input_seg}\n\n"
        "### Question:\n"
        f"{question}\n\n"
        "### Response:"
    )


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return " ".join(text.split()).strip()


def extract_candidates(ex: Dict[str, Any]) -> List[str]:
    candidates = ex.get("candidates_entity_desc_list")
    if isinstance(candidates, list) and candidates:
        return candidates
    question = ex.get("question", "")
    if not isinstance(question, str):
        return []
    return re.findall(r"<[^<>]+>", question)


def extract_prediction(raw_text: str, candidates: List[str]) -> str:
    text = normalize_text(raw_text)
    if not text:
        return ""

    for cand in candidates:
        if cand and cand.strip() in text:
            return cand.strip()

    patterns = [
        r"<[^>]+\[DESC\][^>]+\[TYPE\][^>]*>",
        r"<[^>]+\[DESCRIPTION\][^>]+\[TYPE\][^>]*>",
        r"<[^>]+>",
    ]
    for p in patterns:
        m = re.search(p, text)
        if m:
            return m.group(0).strip().replace("[DESCRIPTION]", "[DESC]")
    return ""


def with_timeout(fn, prompt: str, timeout_sec: int) -> str:
    def _alarm_handler(signum, frame):
        raise SampleTimeoutError(f"single sample timeout after {timeout_sec}s")

    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.setitimer(signal.ITIMER_REAL, float(timeout_sec))
    try:
        return fn(prompt)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)


def load_model_and_tokenizer(checkpoint_path: Path):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {checkpoint_path}")
    if not (checkpoint_path / "adapter_config.json").exists():
        raise FileNotFoundError(f"Missing adapter_config.json in {checkpoint_path}")
    if not (
        (checkpoint_path / "adapter_model.safetensors").exists()
        or (checkpoint_path / "adapter_model.bin").exists()
    ):
        raise FileNotFoundError(
            f"Missing adapter model weights in {checkpoint_path} "
            "(need adapter_model.safetensors or adapter_model.bin)"
        )

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))
    model = AutoPeftModelForCausalLM.from_pretrained(str(checkpoint_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model


def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    formatted = (
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{prompt}"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    inputs = tokenizer(
        formatted,
        return_tensors="pt",
        add_special_tokens=True,
        padding=False,
        truncation=False,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_id
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
    new_ids = outputs[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def run_eval(
    data: List[Dict[str, Any]],
    tokenizer,
    model,
    max_new_tokens: int,
    timeout_sec: int,
):
    rows = []
    y_true: List[str] = []
    y_pred: List[str] = []

    for idx, ex in enumerate(tqdm(data, desc="Evaluating")):
        prompt = build_prompt(ex)
        raw_pred = ""
        error = ""
        try:
            raw_pred = with_timeout(
                lambda p: generate(model, tokenizer, p, max_new_tokens=max_new_tokens),
                prompt,
                timeout_sec=timeout_sec,
            )
        except Exception as e:
            error = str(e)

        pred = extract_prediction(raw_pred, extract_candidates(ex))
        gold = (ex.get("output") or "").strip()
        correct = int(pred == gold)
        y_true.append(gold)
        y_pred.append(pred)
        rows.append(
            {
                "index": idx,
                "id": ex.get("id"),
                "gold": gold,
                "prediction": pred,
                "raw_prediction": raw_pred,
                "correct": correct,
                "error": error,
            }
        )

    n_total = len(rows)
    n_correct = sum(r["correct"] for r in rows)
    acc = n_correct / n_total if n_total else 0.0
    f1 = metrics_mod.f1_score(y_true, y_pred) if n_total else 0.0
    n_errors = sum(1 for r in rows if r["error"])
    return rows, {"n_total": n_total, "n_correct": n_correct, "n_errors": n_errors, "accuracy": acc, "f1": f1}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate simplified EL with local checkpoint only")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA))
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--timeout_sec", type=int, default=DEFAULT_TIMEOUT_SEC)
    parser.add_argument("--save_dir", type=str, default=str(DEFAULT_SAVE_DIR))
    args = parser.parse_args()

    if args.limit <= 0:
        raise ValueError("--limit must be > 0")
    if args.max_new_tokens <= 0:
        raise ValueError("--max_new_tokens must be > 0")
    if args.timeout_sec <= 0:
        raise ValueError("--timeout_sec must be > 0")

    data_path = Path(args.data)
    checkpoint_path = Path(args.checkpoint)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(data_path, args.limit)
    tokenizer, model = load_model_and_tokenizer(checkpoint_path)
    rows, summary = run_eval(
        data=data,
        tokenizer=tokenizer,
        model=model,
        max_new_tokens=args.max_new_tokens,
        timeout_sec=args.timeout_sec,
    )

    stem = f"checkpoint_{checkpoint_path.name}_first{len(data)}"
    pred_path = save_dir / f"predictions_{stem}.jsonl"
    metrics_path = save_dir / f"metrics_{stem}.json"
    with open(pred_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": str(checkpoint_path),
                "data": str(data_path),
                **summary,
                "predictions_path": str(pred_path),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(json.dumps({"metrics_path": str(metrics_path), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
