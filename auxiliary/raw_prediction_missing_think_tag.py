import json
from pathlib import Path


def get_line_numbers_missing_think_tag(predictions_path: Path) -> list[int]:
    """
    返回 jsonl 文件中 `raw_prediction` 不包含 `</think>` 的行号列表（1-based）。
    """
    line_numbers: list[int] = []

    with predictions_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            raw_pred = record.get("raw_prediction", "")
            if not isinstance(raw_pred, str):
                raw_pred = str(raw_pred)

            if "</think>" not in raw_pred:
                line_numbers.append(idx)

    return line_numbers


def main():
    predictions_path = Path("el/results/predictions_llama_r1_p1.jsonl")

    line_numbers = get_line_numbers_missing_think_tag(predictions_path)
    print(line_numbers)
    print(f"共 {len(line_numbers)} 行 raw_prediction 不含 </think>")


if __name__ == "__main__":
    main()

