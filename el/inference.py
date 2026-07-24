#!/usr/bin/env python3
"""
Entity linking 推理：数据加载、prompt 构建、实体抽取、批量评估。
通过 -b / --backend 指定模型：llama | llama_r1 | qwen | qwen_r1。
通过 -p / --prompt 指定 prompt 模式：1 使用原始 TableLlama prompt（默认），2 使用只保留同行同列 cell 的新 prompt。

用法: python el/inference.py -b llama_r1 -p 2
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch
import requests
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from auxiliary.get_prompt_tablellama_test import get_prompt_input

# 从项目根可 import metrics
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import metrics


# ---------- 公共配置与数据 ----------
DATA_PATH = Path.home() / "DATA" / "tablellama" / "ent_link_test.json"
SAVE_DIR = Path(__file__).resolve().parent / "results"
MAX_SAMPLES = 50
MAX_NEW_TOKENS = 4096
TEMPERATURE = 0.6
TOP_P = 0.9
VLLM_API_URL = "http://localhost:8000/v1/completions"

BACKENDS = ("llama", "llama_r1", "qwen", "qwen_r1")


def load_data(limit: int = MAX_SAMPLES, data_path: Path = None):
    path = data_path or DATA_PATH
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data[:limit]


def build_prompt(ex: dict) -> str:
    """
    与 TableLlama 原始 repo 中的 prompt_input 一致。
    """
    instruction = ex.get("instruction", "")
    input_seg = ex.get("input_seg", ex.get("input", ""))
    question = ex.get("question", "")
    return (
        f"### Instruction:\nentity linking task. choose only the correct one from the referent entity candidates.\n\n"
        f"### Input:\n{input_seg}\n\n"
        f"### Question:\n{question}\n\n"
        "### Response:"
    )


def build_prompt2(ex: dict, index: int) -> str:
    """
    新的 prompt 形式：

    ### Instruction: entity linking task

    ### Input: 只包含同行同列的cell, caption和所有header

    ### Candidates: 和原代码格式一样

    ### Question: The entity mention: "A"; the column name "H". From candidates: <>, <>, ..., what is the correct referent entity for the entity mention "A"

    ### Response
    """
    # 使用辅助函数，根据原始 input + entity 获取“只包含同行同列的 cell + caption + header”的新 input
    processed_input = get_prompt_input(DATA_PATH, index)
    question = ex.get("question", "")

    return (
        "### Instruction:\n"
        "entity linking task. choose only the correct one from the referent entity candidates. In the Input below, the table content only contains the caption (if any), all column headers, and the cells from the same row and same column as the selected entity mention.\n\n"
        "### Input:\n"
        f"{processed_input}\n\n"
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


def extract_entity_from_response(raw_text: str, candidates_entity_desc_list):
    text = normalize_entity_text(raw_text)
    if not text:
        return ""
    for cand in candidates_entity_desc_list:
        if cand and cand.strip() in text:
            return ensure_wrapped_in_angle_brackets(cand.strip())
    match = re.search(r"<[^>]+\[DESCRIPTION\][^>]+\[TYPE\][^>]*>", text)
    if match:
        return match.group(0).strip()
    match = re.search(r"<[^>]+>", text)
    if match:
        return match.group(0).strip()
    return ensure_wrapped_in_angle_brackets(text[:500])


def run_entity_linking_eval(
    data,
    predict_fn,
    predictions_path: Path,
    results_title: str,
    desc: str = "Entity linking",
    prompt_mode: int = 1,
):
    Path(predictions_path).parent.mkdir(parents=True, exist_ok=True)
    y_true, y_pred, results = [], [], []
    for idx, ex in enumerate(tqdm(data, desc=desc)):
        if prompt_mode == 2:
            prompt = build_prompt2(ex, idx)
        else:
            prompt = build_prompt(ex)
        try:
            raw_pred = predict_fn(prompt)
        except Exception as e:
            print("Inference error:", e)
            raw_pred = ""
        candidates = ex.get("candidates_entity_desc_list", [])
        pred = extract_entity_from_response(raw_pred, candidates)
        gold = (ex.get("output") or "").strip()
        y_true.append(gold)
        y_pred.append(pred)
        results.append({
            "id": ex.get("id"),
            "prompt": prompt,
            "gold": gold,
            "prediction": pred,
            "raw_prediction": raw_pred,
            "correct": int(pred == gold),
        })
    acc = sum(r["correct"] for r in results) / len(results) if results else 0.0
    f1 = metrics.f1_score(y_true, y_pred)
    print(f"====== RESULTS ({results_title}) ======")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 (exact match): {f1:.4f}")
    with open(predictions_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Predictions saved to {predictions_path}")


# ---------- Llama 3.1 ----------
def _llama_load_model():
    model_name = "meta-llama/Llama-3.1-8B"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for {model_name}: {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model


def _llama_chat_prompt(user_content: str) -> str:
    return (
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_content}"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def _llama_generate(model, tokenizer, prompt: str) -> str:
    full_prompt = _llama_chat_prompt(prompt)
    inputs = tokenizer(full_prompt, return_tensors="pt", add_special_tokens=True, padding=False, truncation=False)
    device = model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        gen_kwargs = {"max_new_tokens": MAX_NEW_TOKENS, "pad_token_id": tokenizer.eos_token_id}
        if TEMPERATURE > 0.0:
            gen_kwargs.update(do_sample=True, temperature=TEMPERATURE, top_p=TOP_P)
        outputs = model.generate(**inputs, **gen_kwargs)
    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


# ---------- Llama R1 ----------
def _llama_r1_load_model():
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for {model_name}: {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model


def _llama_r1_generate(model, tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
    )
    device = model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        gen_kwargs = {"max_new_tokens": MAX_NEW_TOKENS, "pad_token_id": tokenizer.eos_token_id}
        if TEMPERATURE > 0.0:
            gen_kwargs.update(do_sample=True, temperature=TEMPERATURE, top_p=TOP_P)
        outputs = model.generate(**inputs, **gen_kwargs)
    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


# ---------- Qwen vLLM ----------
def _qwen_call(prompt: str) -> str:
    model_name = "Qwen/Qwen2.5-1.5B"
    body = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": 1024,
        "min_tokens": 2,
        "temperature": 0,
        "top_p": 1.0,
    }
    resp = requests.post(VLLM_API_URL, json=body)
    resp.raise_for_status()
    data = resp.json()
    choice = (data.get("choices") or [{}])[0]
    return choice.get("text") or choice.get("content") or ""


# ---------- Qwen R1 vLLM ----------
def _qwen_r1_call(prompt: str) -> str:
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    body = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": 1024,
        "min_tokens": 2,
        "temperature": 0.0,
        "top_p": 1.0,
    }
    resp = requests.post(VLLM_API_URL, json=body)
    resp.raise_for_status()
    return resp.json()["choices"][0]["text"]


# ---------- Backend 派发 ----------
def get_predict_fn_and_meta(backend: str):
    """返回 (predict_fn, predictions_path, results_title, desc)。"""
    if backend == "llama":
        tok, mod = _llama_load_model()
        return (
            lambda prompt: _llama_generate(mod, tok, prompt),
            SAVE_DIR / "predictions_llama.jsonl",
            "Llama-3.1-8B",
            "Entity linking (Llama-3.1-8B)",
        )
    if backend == "llama_r1":
        tok, mod = _llama_r1_load_model()
        return (
            lambda prompt: _llama_r1_generate(mod, tok, prompt),
            SAVE_DIR / "predictions_llama_r1.jsonl",
            "DeepSeek R1 Llama",
            "Entity linking (DeepSeek R1 Llama)",
        )
    if backend == "qwen":
        return (
            _qwen_call,
            SAVE_DIR / "predictions_qwen.jsonl",
            "Qwen2.5-1.5B",
            "Entity linking (Qwen)",
        )
    if backend == "qwen_r1":
        return (
            _qwen_r1_call,
            SAVE_DIR / "predictions_qwen_r1.jsonl",
            "DeepSeek R1 Qwen",
            "Entity linking (Qwen R1)",
        )
    raise ValueError(f"Unknown backend: {backend}. Choose from: {BACKENDS}")


def get_backend_for_single_inference(backend: str):
    """
    供 auxiliary/single_inference.py 使用。
    返回 (load_data, build_prompt, extract_entity_from_response, run_inference, load_msg)。
    """
    load_msg_map = {
        "llama": "Llama-3.1-8B（transformers）",
        "llama_r1": "DeepSeek-R1-Distill-Llama-8B（transformers）",
        "qwen": "Qwen（vLLM API）",
        "qwen_r1": "DeepSeek-R1-Distill-Qwen（vLLM API）",
    }
    load_msg = load_msg_map.get(backend, backend)
    if backend in ("llama", "llama_r1"):
        if backend == "llama":
            tok, mod = _llama_load_model()
            run_inference = lambda prompt: _llama_generate(mod, tok, prompt)
        else:
            tok, mod = _llama_r1_load_model()
            run_inference = lambda prompt: _llama_r1_generate(mod, tok, prompt)
    elif backend == "qwen":
        run_inference = _qwen_call
    elif backend == "qwen_r1":
        run_inference = _qwen_r1_call
    else:
        raise ValueError(f"Unknown backend: {backend}. Choose from: {BACKENDS}")
    return (load_data, build_prompt, extract_entity_from_response, run_inference, load_msg)


def main():
    parser = argparse.ArgumentParser(description="Entity linking 推理，-b 指定模型, -p 指定 prompt 模式")
    parser.add_argument("-b", "--backend", type=str, default="llama", choices=BACKENDS, help="模型后端")
    parser.add_argument(
        "-p",
        "--prompt",
        type=int,
        default=1,
        choices=[1, 2],
        help="Prompt 模式：1 使用原始 TableLlama prompt（默认），2 使用只保留同行同列 cell 的新 prompt",
    )
    args = parser.parse_args()

    data = load_data()
    print(f"Loaded {len(data)} samples from {DATA_PATH}")
    if args.backend in ("qwen", "qwen_r1"):
        print(f"Using vLLM API at {VLLM_API_URL}")
    print(f"Using prompt mode: {args.prompt}")

    predict_fn, predictions_path, results_title, desc = get_predict_fn_and_meta(args.backend)
    run_entity_linking_eval(
        data,
        predict_fn,
        predictions_path,
        results_title,
        desc=desc,
        prompt_mode=args.prompt,
    )


if __name__ == "__main__":
    main()
