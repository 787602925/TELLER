import json
from pathlib import Path


def get_incorrect_line_numbers(predictions_path: Path) -> list[int]:
    """
    返回 jsonl 文件中 `correct` 字段为 0 的行号列表（1-based）。
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

            if record.get("correct") == 0:
                line_numbers.append(idx)

    return line_numbers


def main():
    # 相对项目根目录的路径
    predictions_path = Path("el/results/predictions_llama_r1_p1.jsonl")

    line_numbers = get_incorrect_line_numbers(predictions_path)
    print(line_numbers)
    print(f"共 {len(line_numbers)} 行 correct=0")


if __name__ == "__main__":
    main()

