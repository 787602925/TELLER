"""
单条数据推理：按索引跑一条样本，打印 id / prompt / gold / prediction / raw_prediction。
通过 -b 选择推理后端（llama / llama_r1 / qwen / qwen_r1）。

用法:
  python single_inference.py -b llama --index 0
  python single_inference.py -b llama_r1 -i 5
  python single_inference.py -b llama_r1 -p 2 -i 5
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from el.inference import BACKENDS, get_backend_for_single_inference, build_prompt2


def _print_result(sample_id, prompt, gold, prediction, raw_prediction):
    sep = "=" * 60
    for label, value in [
        ("id", sample_id),
        ("prompt", prompt),
        ("gold", gold),
        ("prediction", prediction),
        ("raw_prediction", raw_prediction),
    ]:
        print()
        print(sep)
        print(label)
        print(sep)
        print(value)
    print()
    print(sep)
    print("correct:", int(prediction == gold))
    print(sep)


def main():
    parser = argparse.ArgumentParser(
        description="单条实体链接推理，查看 id / prompt / gold / prediction / raw_prediction"
    )
    parser.add_argument("-b", "--backend", type=str, default="llama", choices=BACKENDS, help="推理后端")
    parser.add_argument(
        "-p",
        "--prompt",
        type=int,
        default=1,
        choices=[1, 2],
        help="Prompt 模式：1 使用原始 TableLlama prompt（默认），2 使用只保留同行同列 cell 的新 prompt",
    )
    parser.add_argument("-i", "--index", type=int, default=0, help="数据集中的第几条（0-based）")
    args = parser.parse_args()

    if args.index < 0:
        print("--index 必须 >= 0")
        sys.exit(1)

    load_data, build_prompt, extract_entity_from_response, run_inference, load_msg = get_backend_for_single_inference(
        args.backend
    )

    data = load_data(limit=args.index + 1)
    if args.index >= len(data):
        print(f"索引 {args.index} 超出范围，数据集当前只加载了 {len(data)} 条")
        sys.exit(1)

    ex = data[args.index]
    if args.prompt == 2:
        prompt = build_prompt2(ex, args.index)
    else:
        prompt = build_prompt(ex)

    print(f"加载/使用 {load_msg}...")
    try:
        raw_prediction = run_inference(prompt)
    except Exception as e:
        print("推理错误:", e)
        raw_prediction = ""

    candidates = ex.get("candidates_entity_desc_list", [])
    prediction = extract_entity_from_response(raw_prediction, candidates)
    gold = (ex.get("output") or "").strip()
    sample_id = ex.get("id")

    _print_result(sample_id, prompt, gold, prediction, raw_prediction)


if __name__ == "__main__":
    main()
