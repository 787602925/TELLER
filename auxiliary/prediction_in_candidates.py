#!/usr/bin/env python3
"""
统计 predictions.jsonl 中 prediction 在 prompt 的 candidates 中的比例。
"""
import json
import re
from pathlib import Path


def extract_candidates_from_prompt(prompt: str) -> list[str]:
    """从 prompt 中解析出 referent entity candidates 列表。"""
    marker = "The referent entity candidates are:"
    if marker not in prompt:
        return []
    start = prompt.index(marker) + len(marker)
    # 截取到 " What is the correct" 或 ". What is the correct" 之前
    rest = prompt[start:]
    end_match = re.search(r"\s+What is the correct referent entity", rest)
    if end_match:
        rest = rest[: end_match.start()]
    # 提取所有 <...> 形式的候选
    candidates = re.findall(r"<[^>]+>", rest)
    return [c.strip() for c in candidates]


def main():
    path = Path(__file__).resolve().parent.parent / "el" / "results" / "predictions_llama_r1.jsonl"
    # path = Path(__file__).resolve().parent.parent / "el" / "results" / "predictions_qwen.jsonl"
    total = 0
    in_candidates = 0
    not_in_candidates_indices: list[int] = []  # 第 n 条（从 1 开始）prediction 不在 candidates 中

    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompt = obj.get("prompt", "")
            prediction = (obj.get("prediction") or "").strip()
            candidates = extract_candidates_from_prompt(prompt)
            total += 1
            if prediction and prediction in candidates:
                in_candidates += 1
            else:
                not_in_candidates_indices.append(idx)

    if total == 0:
        ratio = 0.0
    else:
        ratio = in_candidates / total
    print(f"总条数: {total}")
    print(f"prediction 在 candidates 中的条数: {in_candidates}")
    print(f"比例: {ratio}")
    print(f"prediction 不在 candidates 中的条号列表: {not_in_candidates_indices}")
    return ratio, not_in_candidates_indices


if __name__ == "__main__":
    main()
