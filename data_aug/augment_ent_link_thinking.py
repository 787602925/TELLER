#!/usr/bin/env python3
"""
使用多模型 API（DeepSeek + RWTHGPT）为实体链接训练数据扩充推理链标签。

输入默认：/DATA1/khli/tablellama/ent_link_train_simplified.json
输出默认：同目录下两个文件
  - ent_link_train_simplified_with_think_correct_{flash|pro}.jsonl
  - ent_link_train_simplified_with_think_wrong_{flash|pro}.jsonl
（带 --num 时会追加 _{num}）

运行环境：conda activate tablellama-fa
需设置环境变量（按模型）：
  - deepseek: DEEPSEEK_API_KEY
  - rwthgpt: RWTHGPT_API_KEY（可用 --api_key_env 覆盖）

示例：
  python data_aug/augment_ent_link_thinking.py --num 100 --model pro
  python data_aug/augment_ent_link_thinking.py --model flash
  python data_aug/augment_ent_link_thinking.py --model mistral
  python data_aug/augment_ent_link_thinking.py --model gpt-5.2
  nohup python data_aug/augment_ent_link_thinking.py --model gpt-5.2 --resume > ~/tableLlama/gpt5.2_aug.log 2>&1 &
  python data_aug/augment_ent_link_thinking.py --model gpt-5.4-mini
  python data_aug/augment_ent_link_thinking.py --num 50 --resume --model flash
  --resume 从已有输出 JSONL 行数继续（跳过已处理条数）
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    import ijson
except ImportError:
    ijson = None

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

DEFAULT_INPUT = "/DATA1/khli/tablellama/ent_link_train_single_table.json"
DEFAULT_MODEL_VARIANT = "pro"
MODEL_CONFIGS = {
    "flash": {
        "model_id": "deepseek-v4-flash",
        "provider": "deepseek",
    },
    "pro": {
        "model_id": "deepseek-v4-pro",
        "provider": "deepseek",
    },
    "mistral": {
        "model_id": "mistral-small-4-119b-2603",
        "provider": "rwthgpt",
    },
    "gpt-5.2": {
        "model_id": "gpt-5.2",
        "provider": "rwthgpt",
    },
    "gpt-5.4-mini": {
        "model_id": "gpt-5.4-mini",
        "provider": "rwthgpt",
    },
}
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_RWTHGPT_BASE_URL = "https://chat.kiconnect.nrw/api/v1"
DEFAULT_API_KEY_ENV_BY_PROVIDER = {
    "deepseek": "DEEPSEEK_API_KEY",
    "rwthgpt": "RWTHGPT_API_KEY",
}
RATE_LIMIT_WAIT_MODELS = {"mistral", "gpt-5.2", "gpt-5.4-mini"}
# 与 el/inference.py、schema_aug 等保持一致
THINK_OPEN = "<" + "think>"
THINK_CLOSE = "</" + "think>"


def build_prompt(ex: Dict[str, Any]) -> str:
    """与 el/sft.py build_prompt 一致的推理 prompt。"""
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
        "Reasoning requirements:\n"
        "1) Use context evidence only (page/section/caption + row/column + candidate [DESCRIPTION]/[TYPE]).\n"
        "2) Keep reasoning concise; do NOT restate the full table/question/candidate list.\n"
        "3) Do NOT copy/paste candidate lists from the input; only cite minimal distinguishing evidence.\n"
        "4) The reasoning should support the same final entity you output after </think>.\n"
        "5) Reasoning is mandatory: include non-empty reasoning content before the final entity; do not output answer-only.\n\n"
        "After reasoning, output exactly one referent entity in candidate format "
        "(e.g. <EntityName [DESCRIPTION] ... [TYPE] ...>).\n\n"
        "### Response:"
    )


def sanitize_reasoning(reasoning: str) -> str:
    """清洗冗余/泄露式句子，并压缩为短 reasoning。"""
    if not isinstance(reasoning, str):
        return ""
    text = reasoning.replace("\r\n", "\n").strip()
    if not text:
        return ""

    # 去掉典型泄露/元话术
    banned = [
        r"(?i)\b(the\s+)?provided\s+correct\s+answer\b",
        r"(?i)\bfor\s+verification\b",
        r"(?i)\bgold\s*(answer|label)?\b",
        r"(?i)\bwe\s+are\s+given\s+an?\s+entity\s+linking\s+task\b",
        r"(?i)\bi\s+should\s+output\s+exactly\b",
        r"(?i)^\s*candidates?\s*[:：]\s*$",
    ]
    kept: List[str] = []
    for raw in text.split("\n"):
        line = re.sub(r"\s+", " ", raw).strip()
        if not line:
            continue
        if any(re.search(pat, line) for pat in banned):
            continue
        # 过滤“复读候选列表”的行：一行中出现多个候选模板
        if len(re.findall(r"<[^<>]+?\[DESCRIPTION\][^<>]*?>", line)) >= 2:
            continue
        if re.search(r"(?i)\bthe referent entity candidates are\b", line):
            continue
        kept.append(line)

    return "\n".join(kept).strip()


def format_augmented_output(reasoning: str, answer: str) -> str:
    """拼接为训练用的 thinking + 答案格式（与 el/inference.py 解析兼容）。"""
    reasoning = sanitize_reasoning(reasoning)
    answer = (answer or "").strip()
    if not reasoning:
        return answer
    return f"{THINK_OPEN}\n{reasoning}\n{THINK_CLOSE}\n{answer}"


def split_reasoning_and_answer_from_content(content: str) -> tuple[str, str]:
    """
    从 content 中解析 <think>reasoning</think>answer，作为非 DeepSeek 模型回退。
    若没有 think 块，则返回 ("", 原文)。
    """
    text = (content or "").strip()
    if not text:
        return "", ""
    m = re.search(r"<think>\s*(.*?)\s*</think>\s*(.*)$", text, flags=re.S | re.I)
    if not m:
        return "", text
    reasoning = (m.group(1) or "").strip()
    answer = (m.group(2) or "").strip()
    return reasoning, answer


def split_reasoning_and_answer_for_gpt52(content: str) -> tuple[str, str]:
    """
    gpt-5.2 常见输出形态：先给解释，最后给一个候选实体。
    仅当检测到末尾候选实体时，提取其前文作为 reasoning。
    """
    text = (content or "").strip()
    if not text:
        return "", ""
    matches = list(
        re.finditer(r"<\s*[^<>]+?\s*\[DESCRIPTION\]\s*[^<>]+?\s*\[TYPE\]\s*[^<>]+?>", text, flags=re.S)
    )
    if not matches:
        return "", text
    last = matches[-1]
    answer = normalize_entity_text(last.group(0))
    reasoning = text[: last.start()].strip()
    return reasoning, answer


def iter_json_array(path: Path, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    """流式读取顶层 JSON 数组，避免一次性加载百万条。"""
    if ijson is not None:
        with open(path, "rb") as f:
            gen = ijson.items(f, "item")
            if limit is not None:
                from itertools import islice

                yield from islice(gen, limit)
            else:
                yield from gen
        return

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = data[:limit] if limit is not None else data
    yield from items


def count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def skip_iter(it: Iterator[Dict[str, Any]], n: int) -> Iterator[Dict[str, Any]]:
    from itertools import islice

    return islice(it, n, None)


def is_rate_limit_error(err: Exception) -> bool:
    """判断是否为请求限流/配额上限错误。"""
    status_code = getattr(err, "status_code", None)
    if status_code == 429:
        return True
    response = getattr(err, "response", None)
    if response is not None and getattr(response, "status_code", None) == 429:
        return True
    if err.__class__.__name__.lower() == "ratelimiterror":
        return True
    msg = str(err).lower()
    return any(
        key in msg
        for key in (
            "rate limit",
            "too many requests",
            "quota",
            "requests limit",
            "request limit",
        )
    )


def _parse_seconds_with_unit(raw_num: str, raw_unit: str) -> Optional[float]:
    try:
        value = float(raw_num)
    except (TypeError, ValueError):
        return None
    unit = (raw_unit or "s").lower()
    if unit in ("ms", "millisecond", "milliseconds"):
        return value / 1000.0
    if unit in ("m", "min", "mins", "minute", "minutes"):
        return value * 60.0
    if unit in ("h", "hr", "hrs", "hour", "hours"):
        return value * 3600.0
    return value


def get_rate_limit_wait_seconds(err: Exception) -> float:
    """从错误头/文本中提取等待秒数，提取失败时给保守默认值。"""
    response = getattr(err, "response", None)
    headers = getattr(response, "headers", None) or {}

    retry_after_ms = headers.get("retry-after-ms") if hasattr(headers, "get") else None
    if retry_after_ms:
        try:
            wait_s = float(retry_after_ms) / 1000.0
            if wait_s > 0:
                return wait_s
        except (TypeError, ValueError):
            pass

    retry_after = headers.get("retry-after") if hasattr(headers, "get") else None
    if retry_after:
        try:
            wait_s = float(retry_after)
            if wait_s > 0:
                return wait_s
        except (TypeError, ValueError):
            pass

    reset_value = None
    if hasattr(headers, "get"):
        for key in ("x-ratelimit-reset", "x-rate-limit-reset", "ratelimit-reset"):
            reset_value = headers.get(key)
            if reset_value:
                break
    if reset_value:
        try:
            reset_num = float(reset_value)
            now = time.time()
            # 常见两种含义：绝对时间戳 or 剩余秒数
            wait_s = reset_num - now if reset_num > now + 1 else reset_num
            if wait_s > 0:
                return wait_s
        except (TypeError, ValueError):
            pass

    msg = str(err)
    # 例：Please try again in 120s / 2 minutes / 500ms
    m = re.search(
        r"try again in\s+([0-9]+(?:\.[0-9]+)?)\s*(ms|milliseconds?|s|sec|secs|seconds?|m|min|mins|minutes?|h|hr|hrs|hours?)?",
        msg,
        flags=re.I,
    )
    if m:
        wait_s = _parse_seconds_with_unit(m.group(1), m.group(2) or "s")
        if wait_s is not None and wait_s > 0:
            return wait_s

    m = re.search(
        r"reset(?:s)?\s+in\s+([0-9]+(?:\.[0-9]+)?)\s*(ms|milliseconds?|s|sec|secs|seconds?|m|min|mins|minutes?|h|hr|hrs|hours?)?",
        msg,
        flags=re.I,
    )
    if m:
        wait_s = _parse_seconds_with_unit(m.group(1), m.group(2) or "s")
        if wait_s is not None and wait_s > 0:
            return wait_s

    # 未提供明确窗口时，默认等待 5 分钟再重试
    return 600.0


def call_model(
    client: Any,
    prompt: str,
    *,
    model_variant: str,
    model: str,
    max_tokens: int,
    reasoning_effort: str,
    temperature: float,
    enable_deepseek_thinking: bool,
    enable_rate_limit_wait: bool,
) -> tuple[str, str]:
    """返回 (reasoning_content, content)。每条样本只调用一次 API。"""
    req_kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if enable_deepseek_thinking:
        req_kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
        req_kwargs["reasoning_effort"] = reasoning_effort

    retry_round = 0
    while True:
        try:
            resp = client.chat.completions.create(**req_kwargs)
            msg = resp.choices[0].message
            reasoning = getattr(msg, "reasoning_content", None) or ""
            content = getattr(msg, "content", None) or ""
            return str(reasoning), str(content)
        except Exception as e:
            if not enable_rate_limit_wait or not is_rate_limit_error(e):
                raise
            retry_round += 1
            wait_s = max(1.0, get_rate_limit_wait_seconds(e))
            print(
                (
                    f"[rate-limit] model={model_variant} hit request cap; "
                    f"sleep {wait_s:.1f}s then retry (round={retry_round})"
                ),
                file=sys.stderr,
            )
            time.sleep(wait_s)


def resolve_model_config(variant: str) -> Dict[str, str]:
    """将 --model 映射到模型配置。"""
    key = (variant or "").strip().lower()
    if key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown --model: {variant!r}, choices: {', '.join(MODEL_CONFIGS)}")
    cfg = MODEL_CONFIGS[key]
    return {"variant": key, "model_id": cfg["model_id"], "provider": cfg["provider"]}


def default_output_path(
    input_path: Path, num: Optional[int], model_variant: str, kind: str
) -> Path:
    variant = (model_variant or DEFAULT_MODEL_VARIANT).strip().lower()
    stem = input_path.stem
    if num is not None:
        name = f"{stem}_with_think_{kind}_{variant}_{num}.jsonl"
    else:
        name = f"{stem}_with_think_{kind}_{variant}.jsonl"
    return input_path.parent / name


def extract_entity_from_text(text: str) -> str:
    """从模型输出中提取候选实体字符串。"""
    if not isinstance(text, str):
        return ""
    s = text.strip()
    if not s:
        return ""
    m = re.search(r"<\s*[^<>]+?\s*\[DESCRIPTION\]\s*[^<>]+?\s*\[TYPE\]\s*[^<>]+?>", s, flags=re.S)
    if m:
        return " ".join(m.group(0).split()).strip()
    m = re.search(r"<[^<>]+>", s, flags=re.S)
    if m:
        return " ".join(m.group(0).split()).strip()
    return " ".join(s.split()).strip()


def normalize_entity_text(text: str) -> str:
    return " ".join((text or "").split()).strip()


def canonical_entity_text(text: str) -> str:
    """Normalize for comparison: collapse spaces and ignore outer angle brackets."""
    s = normalize_entity_text(text)
    if s.startswith("<") and s.endswith(">") and len(s) >= 2:
        s = s[1:-1].strip()
    return s


def ensure_angle_brackets(text: str) -> str:
    """Ensure final entity keeps candidate-style angle brackets when possible."""
    s = normalize_entity_text(text)
    if not s:
        return s
    if s.startswith("<") and s.endswith(">"):
        return s
    if "[DESCRIPTION]" in s or "[TYPE]" in s:
        return f"<{s}>"
    return s


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Augment entity-linking data with multi-model thinking traces."
    )
    p.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT,
        help=f"Input JSON array path (default: {DEFAULT_INPUT})",
    )
    p.add_argument(
        "--correct_output",
        type=str,
        default="",
        help="Output JSONL path for correct predictions (auto-generated by default)",
    )
    p.add_argument(
        "--wrong_output",
        type=str,
        default="",
        help="Output JSONL path for wrong predictions (auto-generated by default)",
    )
    p.add_argument(
        "--num",
        type=int,
        default=None,
        metavar="N",
        help="Process first N records only; process all if omitted",
    )
    p.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_VARIANT,
        choices=tuple(MODEL_CONFIGS.keys()),
        help=(
            "Model variant: "
            "flash|pro (DeepSeek), mistral|gpt-5.2|gpt-5.4-mini (RWTHGPT), default: pro"
        ),
    )
    p.add_argument(
        "--base_url",
        type=str,
        default="",
        help="API base URL (auto from --model if omitted)",
    )
    p.add_argument(
        "--api_key_env",
        type=str,
        default="",
        help="Environment variable name for API key (auto from --model provider if omitted)",
    )
    p.add_argument("--max_tokens", type=int, default=8192, help="completion max_tokens")
    p.add_argument(
        "--reasoning_effort",
        type=str,
        default="high",
        choices=("low", "high", "none"),
        help="Thinking depth (high = Think Max)",
    )
    p.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    p.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between requests")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume by skipping already written lines in output JSONL files",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.num is not None and args.num < 1:
        raise SystemExit("--num must be a positive integer")

    model_cfg = resolve_model_config(args.model)
    provider = model_cfg["provider"]
    model_id = model_cfg["model_id"]

    if args.base_url.strip():
        base_url = args.base_url.strip().rstrip("/")
    else:
        base_url = DEFAULT_BASE_URL if provider == "deepseek" else DEFAULT_RWTHGPT_BASE_URL

    api_key_env = (
        args.api_key_env.strip()
        if args.api_key_env.strip()
        else DEFAULT_API_KEY_ENV_BY_PROVIDER[provider]
    )
    api_key = os.environ.get(api_key_env, "").strip()
    if not api_key:
        raise SystemExit(f"Please set environment variable {api_key_env}")

    try:
        from openai import OpenAI
    except ImportError as e:
        raise SystemExit("openai package is required: pip install openai") from e

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        raise SystemExit(f"Input file not found: {input_path}")

    correct_path = (
        Path(args.correct_output).expanduser().resolve()
        if args.correct_output.strip()
        else default_output_path(input_path, args.num, args.model, "correct")
    )
    wrong_path = (
        Path(args.wrong_output).expanduser().resolve()
        if args.wrong_output.strip()
        else default_output_path(input_path, args.num, args.model, "wrong")
    )
    correct_path.parent.mkdir(parents=True, exist_ok=True)
    wrong_path.parent.mkdir(parents=True, exist_ok=True)

    client = OpenAI(api_key=api_key, base_url=base_url)
    print(
        f"Using model: --model {args.model} -> {model_id} "
        f"(provider={provider}, base_url={base_url}, api_key_env={api_key_env})"
    )

    skip_n = count_jsonl_lines(correct_path) + count_jsonl_lines(wrong_path) if args.resume else 0
    if skip_n and not args.resume:
        print(f"Warning: found {skip_n} existing result lines; without --resume outputs will be overwritten", file=sys.stderr)

    items = iter_json_array(input_path, args.num)
    if skip_n:
        items = skip_iter(items, skip_n)
        print(f"resume: skipped first {skip_n} records")

    total_hint = args.num
    if total_hint is None and skip_n:
        print("resume without --num: total record count is unknown")

    out_mode = "a" if args.resume and skip_n else "w"
    correct_count = 0
    wrong_count = 0
    api_error_count = 0

    iterator: Iterator[Dict[str, Any]] = items
    if tqdm is not None:
        iterator = tqdm(iterator, total=total_hint, desc="augment", unit="item")

    with open(correct_path, out_mode, encoding="utf-8") as correct_f, open(
        wrong_path, out_mode, encoding="utf-8"
    ) as wrong_f:
        for idx, ex in enumerate(iterator):
            global_idx = skip_n + idx
            prompt = build_prompt(ex)
            gold = (ex.get("output") or "").strip()
            gold_norm = normalize_entity_text(gold)
            record_base = {
                "instruction": ex.get("instruction"),
                "input_seg": ex.get("input_seg", ex.get("input")),
                "question": ex.get("question"),
                "output_gold": gold,
                "index": global_idx,
            }
            try:
                reasoning, content = call_model(
                    client,
                    prompt,
                    model_variant=model_cfg["variant"],
                    model=model_id,
                    max_tokens=args.max_tokens,
                    reasoning_effort=args.reasoning_effort,
                    temperature=args.temperature,
                    enable_deepseek_thinking=(provider == "deepseek"),
                    enable_rate_limit_wait=(model_cfg["variant"] in RATE_LIMIT_WAIT_MODELS),
                )
                content_reasoning, content_answer = split_reasoning_and_answer_from_content(content)
                if not reasoning and content_reasoning:
                    reasoning = content_reasoning
                answer_source = content_answer if content_answer else content
                if model_cfg["variant"] == "gpt-5.2":
                    gpt52_reasoning, gpt52_answer = split_reasoning_and_answer_for_gpt52(content)
                    if not reasoning and gpt52_reasoning:
                        reasoning = gpt52_reasoning
                    if gpt52_answer:
                        answer_source = gpt52_answer
                pred_entity = extract_entity_from_text(answer_source)
                pred_norm = canonical_entity_text(pred_entity)
                gold_cmp = canonical_entity_text(gold_norm)
                is_correct = bool(gold_cmp) and pred_norm == gold_cmp
                final_answer = ensure_angle_brackets(pred_entity if pred_entity else answer_source.strip())
                augmented = format_augmented_output(reasoning, final_answer)
                out_record = {
                    **{k: ex.get(k) for k in ("instruction", "input_seg", "question")},
                    "input_seg": ex.get("input_seg", ex.get("input")),
                    "output": augmented,
                    "output_gold": gold,
                    "model_content": content.strip(),
                    "model_pred_entity": pred_entity,
                    "is_correct": is_correct,
                }
                if is_correct:
                    correct_f.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                    correct_f.flush()
                    correct_count += 1
                else:
                    wrong_f.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                    wrong_f.flush()
                    wrong_count += 1
            except Exception as e:
                api_error_count += 1
                wrong_count += 1
                fail_record = {
                    **record_base,
                    "error": str(e),
                    "prompt_len": len(prompt),
                    "model_content": "",
                    "model_pred_entity": "",
                    "is_correct": False,
                }
                wrong_f.write(json.dumps(fail_record, ensure_ascii=False) + "\n")
                wrong_f.flush()
                if tqdm is None:
                    print(f"[wrong] index={global_idx} {e}", file=sys.stderr)

            if args.sleep > 0:
                time.sleep(args.sleep)

    print(f"Done: correct={correct_count}, wrong={wrong_count} (API errors={api_error_count})")
    print(f"Correct output: {correct_path}")
    print(f"Wrong output: {wrong_path}")


if __name__ == "__main__":
    main()
