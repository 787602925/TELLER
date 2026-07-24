#!/usr/bin/env python3
"""
快速冒烟测试：单卡、极小模型、1 次 forward+backward。
使用 transformers 原生 attn_implementation="flash_attention_2"，不再依赖 llama_attn_replace.py。

示例:
  CUDA_VISIBLE_DEVICES=0 python test_llama_attn_replace.py --bf16
  CUDA_VISIBLE_DEVICES=0 python test_llama_attn_replace.py --bf16 --checkpointing
  python test_llama_attn_replace.py --no-flash
"""
from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test native flash attention")
    parser.add_argument(
        "--model",
        default="hf-internal-testing/tiny-random-LlamaForCausalLM",
    )
    parser.add_argument("--no-flash", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--checkpointing", action="store_true")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch", type=int, default=2)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM

    use_flash = not args.no_flash
    if use_flash and not torch.cuda.is_available():
        print("No CUDA, falling back to --no-flash")
        use_flash = False

    dtype = torch.bfloat16 if (args.bf16 and torch.cuda.is_available()) else torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attn_impl = "flash_attention_2" if use_flash else "eager"

    print(f"device={device} attn_impl={attn_impl} dtype={dtype} checkpointing={args.checkpointing}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        attn_implementation=attn_impl,
    ).to(device)
    model.train()

    if args.checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    b, s = args.batch, args.seq_len
    input_ids = torch.randint(0, model.config.vocab_size, (b, s), device=device)
    attention_mask = torch.ones(b, s, device=device, dtype=torch.long)
    labels = input_ids.clone()

    if args.bf16 and device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
    else:
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)

    loss = out.loss
    assert loss.ndim == 0 and torch.isfinite(loss), loss
    loss.backward()
    print(f"OK: forward + backward finished, loss={float(loss.detach().cpu()):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
