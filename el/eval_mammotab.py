#!/usr/bin/env python3
"""
MammoTab (SemTab 2024) 评估脚本 — 训练阶段中间评估。

数据集: ~/DATA/mammotab/mammotab_dataset_semtab/mammotab_2024_prompts_50.jsonl
Prompt 结构与 el/token_estimate.py 的 build_prompt 保持一致（### Instruction / Input / Question / Response）。

用法:
    python -m el.eval_mammotab -b llama --limit 100 --data ~/DATA/mammotab/my_custom.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests
import torch

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import metrics as metrics_mod

SAVE_DIR = Path(__file__).resolve().parent / "results"
MAX_NEW_TOKENS = 4096
TEMPERATURE = 0.6
TOP_P = 0.9
VLLM_API_URL = "http://localhost:8000/v1/completions"
BACKENDS = ("llama", "llama_r1", "qwen", "qwen_r1")

DATA_PATH = (
    Path.home()
    / "DATA"
    / "mammotab"
    / "mammotab_dataset_semtab"
    / "mammotab_2024_prompts_50.jsonl"
)


# ---------- 数据加载 ----------

def load_data(path: Path = DATA_PATH, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """逐行读取 JSONL 文件，返回样本列表。"""
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data[:limit] if limit is not None else data


# ---------- Prompt 构建 ----------

def build_prompt(ex: Dict[str, Any]) -> str:
    """
    与训练脚本 `el/sft.py` 对齐：
    - Instruction 使用固定模板
    - Input 优先 input_seg，回退 input
    """
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


def normalize_entity_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return " ".join(text.split()).strip()


def ensure_wrapped_in_angle_brackets(s: str) -> str:
    if not s or not isinstance(s, str):
        return s
    s = s.strip()
    if not s.startswith("<"):
        s = "<" + s
    if not s.endswith(">"):
        s = s + ">"
    return s


# ---------- 本文件内置后端推理 ----------

def _load_hf_or_peft_model(model_name_or_path: str):
    # 若路径是 LoRA adapter（含 adapter_config.json），优先按 PEFT 加载。
    if (Path(model_name_or_path) / "adapter_config.json").exists():
        model = AutoPeftModelForCausalLM.from_pretrained(model_name_or_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    return model


def _llama_load_model(model_name: str = "meta-llama/Llama-3.1-8B"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for {model_name}: {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = _load_hf_or_peft_model(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model


def _llama_chat_prompt(user_content: str) -> str:
    return (
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_content}"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def _resolve_generation_token_ids(tokenizer):
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = eos_token_id
    return eos_token_id, pad_token_id


def _llama_generate(model, tokenizer, prompt: str) -> str:
    full_prompt = _llama_chat_prompt(prompt)
    inputs = tokenizer(
        full_prompt,
        return_tensors="pt",
        add_special_tokens=True,
        padding=False,
        truncation=False,
    )
    device = model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    eos_token_id, pad_token_id = _resolve_generation_token_ids(tokenizer)
    with torch.no_grad():
        gen_kwargs = {
            "max_new_tokens": MAX_NEW_TOKENS,
            "eos_token_id": eos_token_id,
            "pad_token_id": pad_token_id,
        }
        if TEMPERATURE > 0.0:
            gen_kwargs.update(do_sample=True, temperature=TEMPERATURE, top_p=TOP_P)
        outputs = model.generate(**inputs, **gen_kwargs)
    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def _llama_r1_load_model(model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for {model_name}: {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = _load_hf_or_peft_model(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model


def _llama_r1_generate(model, tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    device = model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    eos_token_id, pad_token_id = _resolve_generation_token_ids(tokenizer)
    with torch.no_grad():
        gen_kwargs = {
            "max_new_tokens": MAX_NEW_TOKENS,
            "eos_token_id": eos_token_id,
            "pad_token_id": pad_token_id,
        }
        if TEMPERATURE > 0.0:
            gen_kwargs.update(do_sample=True, temperature=TEMPERATURE, top_p=TOP_P)
        outputs = model.generate(**inputs, **gen_kwargs)
    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def _qwen_call(prompt: str, model_name: str = "Qwen/Qwen2.5-1.5B") -> str:
    body = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": 1024,
        "min_tokens": 2,
        "temperature": 0,
        "top_p": 1.0,
    }
    resp = requests.post(VLLM_API_URL, json=body, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    choice = (data.get("choices") or [{}])[0]
    return choice.get("text") or choice.get("content") or ""


def _qwen_r1_call(prompt: str, model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B") -> str:
    body = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": 1024,
        "min_tokens": 2,
        "temperature": 0.0,
        "top_p": 1.0,
    }
    resp = requests.post(VLLM_API_URL, json=body, timeout=120)
    resp.raise_for_status()
    return resp.json()["choices"][0]["text"]


def get_predict_fn_and_meta(backend: str, model_path: Optional[str] = None):
    if backend == "llama":
        model_name = model_path or "meta-llama/Llama-3.1-8B"
        tok, mod = _llama_load_model(model_name=model_name)
        return (
            lambda prompt: _llama_generate(mod, tok, prompt),
            SAVE_DIR / "predictions_llama.jsonl",
            model_name,
            "Entity linking (Llama-3.1-8B)",
        )
    if backend == "llama_r1":
        model_name = model_path or "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
        tok, mod = _llama_r1_load_model(model_name=model_name)
        return (
            lambda prompt: _llama_r1_generate(mod, tok, prompt),
            SAVE_DIR / "predictions_llama_r1.jsonl",
            model_name,
            "Entity linking (DeepSeek R1 Llama)",
        )
    if backend == "qwen":
        model_name = model_path or "Qwen/Qwen2.5-1.5B"
        return (
            lambda prompt: _qwen_call(prompt, model_name=model_name),
            SAVE_DIR / "predictions_qwen.jsonl",
            model_name,
            "Entity linking (Qwen)",
        )
    if backend == "qwen_r1":
        model_name = model_path or "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        return (
            lambda prompt: _qwen_r1_call(prompt, model_name=model_name),
            SAVE_DIR / "predictions_qwen_r1.jsonl",
            model_name,
            "Entity linking (Qwen R1)",
        )
    raise ValueError(f"Unknown backend: {backend}. Choose from: {BACKENDS}")


# ---------- 候选实体抽取 ----------

def extract_candidates_from_question(question: str) -> List[str]:
    """从 question 字段中解析出所有 <...> 形式的候选实体字符串。"""
    return re.findall(r"<[^<>]+>", question)


def _collapse_ws(s: str) -> str:
    return " ".join(s.split())


def _slice_before_prompt_echo(raw: str) -> str:
    """
    模型常把整段 ### Input / ### Question 再生成一遍；若在其上匹配 <...>，
    会误命中「题目里的第一个候选」。只保留第一次出现回显标记之前的内容。
    """
    if not raw:
        return raw
    cut = len(raw)
    for sep in ("\n### Input:\n", "\n### Question:\n"):
        i = raw.find(sep)
        if i != -1:
            cut = min(cut, i)
    i = raw.find("\n[TLE] ")
    if i != -1:
        cut = min(cut, i)
    return raw[:cut] if cut < len(raw) else raw


def _canonical_candidate(c: str) -> str:
    c = (c or "").strip()
    if not c:
        return ""
    if c.startswith("<") and c.endswith(">"):
        return c
    return ensure_wrapped_in_angle_brackets(c)


def extract_entity_from_response(raw_text: str, candidates: List[str]) -> str:
    """
    从 raw_prediction 中解析与 gold 一致的 `<Name [DESC] ... [TYPE] ...>`。

    改进点：
    - 截断回显的 prompt，避免匹配到题目里的候选；
    - 除整段尖括号外，用「去掉 <> 后的 inner」做子串匹配（应对引号内、无尖括号输出）；
    - 从引号内、尖括号内抓取 `[DESC]...[TYPE]` 再与候选表对齐。
    """
    snippet = _slice_before_prompt_echo(raw_text)
    text = normalize_entity_text(snippet)
    if not text:
        return ""

    canon_set = set()
    inner_rows: List[tuple[int, str, str]] = []
    for c in candidates:
        canon = _canonical_candidate(c)
        if not canon or len(canon) < 3:
            continue
        inner = canon[1:-1]
        canon_set.add(canon)
        inner_rows.append((len(inner), inner, canon))
    inner_rows.sort(key=lambda x: x[0], reverse=True)

    text_flat = _collapse_ws(text)

    # 1) 候选 inner 出现在截断后的文本中（按最长 inner 优先，减少误匹配短名）
    for _ln, inner, canon in inner_rows:
        inner_variants = {
            inner,
            inner.replace("[DESC]", "[DESCRIPTION]"),
            inner.replace("[DESCRIPTION]", "[DESC]"),
        }
        inner_variants_flat = {_collapse_ws(v) for v in inner_variants}
        if any(v in text for v in inner_variants) or any(v in text_flat for v in inner_variants_flat):
            return canon

    # 2) 尖括号实体（仅含 [DESC]/[TYPE] 的典型 mammotab 形），且必须在候选表里
    for m in re.finditer(r"<([^>]*\[DESC\][^>]*\[TYPE\][^>]*)>", text):
        wrapped = "<" + m.group(1).strip() + ">"
        if wrapped in canon_set:
            return wrapped

    # 3) 引号内的「无尖括号」实体描述（直引号与常见弯引号）
    _quoted_patterns = (
        r"'([^']*?\[DESC\].*?\[TYPE\][^']*)'",
        r'"([^"]*?\[DESC\].*?\[TYPE\][^"]*)"',
        r"\u2018([^\u2019]*?\[DESC\].*?\[TYPE\][^\u2019]*)\u2019",
        r"\u201c([^\u201d]*?\[DESC\].*?\[TYPE\][^\u201d]*)\u201d",
    )
    for qp in _quoted_patterns:
        for m in re.finditer(qp, text, flags=re.DOTALL):
            inner_guess = _collapse_ws(m.group(1).strip())
            if not inner_guess:
                continue
            for _ln, inner, canon in inner_rows:
                if inner_guess == _collapse_ws(inner) or inner_guess == inner:
                    return canon

    # 4) 行内裸的 [DESC]...[TYPE]...（如 ### Output:\\nName [DESC] ... [TYPE] ...）
    loose = re.compile(
        r"(?<![<\w])([^\n<]{1,4000}?\[DESC\].*?\[TYPE\][^\n,<]{0,2000})(?=\s*$|\s*[\n#]|\s*###|\Z)",
        re.MULTILINE,
    )
    for m in loose.finditer(text):
        inner_guess = _collapse_ws(m.group(1).strip().rstrip(".,;:"))
        if not inner_guess or "[DESC]" not in inner_guess:
            continue
        for _ln, inner, canon in inner_rows:
            if inner_guess == _collapse_ws(inner) or inner_guess == inner:
                return canon

    # 5) 与 inference 一致：支持 [DESCRIPTION] 标记
    match = re.search(r"<[^>]+\[DESCRIPTION\][^>]+\[TYPE\][^>]*>", text)
    if match:
        m = match.group(0).strip()
        if m in canon_set:
            return m
        m_desc = m.replace("[DESCRIPTION]", "[DESC]")
        if m_desc in canon_set:
            return m_desc

    return ""


# ---------- 主评估循环 ----------

def run_eval(
    data: List[Dict[str, Any]],
    predict_fn,
    predictions_path: Path,
    results_title: str,
    desc: str = "MammoTab eval",
) -> Dict[str, Any]:
    Path(predictions_path).parent.mkdir(parents=True, exist_ok=True)
    y_true: List[str] = []
    y_pred: List[str] = []
    results = []

    for ex in tqdm(data, desc=desc):
        prompt = build_prompt(ex)
        try:
            raw_pred = predict_fn(prompt)
        except Exception as e:
            print(f"Inference error: {e}")
            raw_pred = ""

        candidates = extract_candidates_from_question(ex.get("question", ""))
        pred = extract_entity_from_response(raw_pred, candidates)
        gold = (ex.get("output") or "").strip()

        y_true.append(gold)
        y_pred.append(pred)
        results.append({
            "table": ex.get("table"),
            "cell": ex.get("cell"),
            "prompt": prompt,
            "gold": gold,
            "prediction": pred,
            "raw_prediction": raw_pred,
            "correct": int(pred == gold),
        })

    n_correct = sum(r["correct"] for r in results)
    n_total = len(results)
    acc = n_correct / n_total if n_total else 0.0
    f1 = metrics_mod.f1_score(y_true, y_pred)

    print(f"\n====== RESULTS ({results_title}) ======")
    print(f"Accuracy : {acc:.4f}  ({n_correct}/{n_total})")
    print(f"F1 (exact): {f1:.4f}")

    with open(predictions_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Predictions saved → {predictions_path}")

    return {"accuracy": acc, "f1": f1, "n_correct": n_correct, "n_total": n_total}


# ---------- CLI 入口 ----------

def main() -> None:
    parser = argparse.ArgumentParser(description="MammoTab SemTab 2024 评估")
    parser.add_argument(
        "-b", "--backend",
        default="llama_r1",
        choices=BACKENDS,
        help="模型后端（与 inference.py 相同）",
    )
    parser.add_argument(
        "--data",
        default=str(DATA_PATH),
        help="JSONL 数据集路径（默认 mammotab_2024_prompts_50.jsonl）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="最多评估 N 条样本（默认全部）",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="模型或 LoRA adapter 路径。若不传则使用各 backend 默认模型。",
    )
    args = parser.parse_args()

    data = load_data(Path(args.data), limit=args.limit)
    print(f"Loaded {len(data)} samples from {args.data}")

    predict_fn, _, results_title, _ = get_predict_fn_and_meta(args.backend, model_path=args.model_path)
    predictions_path = SAVE_DIR / f"mammotab_predictions_{args.backend}.jsonl"
    desc = f"MammoTab ({results_title})"

    run_eval(data, predict_fn, predictions_path, results_title=results_title, desc=desc)


if __name__ == "__main__":
    main()
