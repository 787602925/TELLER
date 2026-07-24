# CoT SFT: fine-tune on chain-of-thought (<think>...</think> + final entity) targets.
#
# 支持两种起点，由 --adapter_name_or_path 是否给出决定：
#
# 1) 在已完成的 SFT LoRA 模型基础上继续训练（--adapter_name_or_path 指向已有 adapter 目录）：
#      base model : meta-llama/Llama-3.1-8B
#      adapter    : 例如 result_mammo/checkpoint-17484 / result_mammo2/checkpoint-21000
#                   (LoRA q/k/v/o + 已 resize 的 embed/lm_head)
#    tokenizer 从 adapter 目录加载（含已扩充的 vocab），继续训练该 LoRA 权重。
#
# 2) 直接从原始 base 模型（如 meta-llama/Llama-3.1-8B）开始做 CoT SFT
#    （不传 --adapter_name_or_path，或传空字符串）：
#      --lora_only False（默认）: 全参数微调，训练 base 模型的全部权重（无 LoRA）。
#      --lora_only True         : 从 base 模型初始化一个全新的 LoRA(q/k/v/o) adapter 并只训练它，
#                                  其余权重（含 --trainable_params 里的 embed/norm）保持冻结。
#
# 训练数据默认 : /DATA1/khli/t&m/merged_CoTSFT_20k&10k.json
#   每条样本含 input_seg / question / output，其中 output 形如:
#       <think>\n...reasoning...\n</think>\n<Entity [DESCRIPTION] ... [TYPE] ...>
# 训练目标是让模型输出对齐训练数据 output 的 value（含 think + 最终实体）。
#
# Prompt 使用 data_aug/augment_ent_link_thinking.py 的 build_prompt（带 Reasoning requirements）。
#
# 运行环境：conda activate tablellama-fa
#
# 示例（单卡，continue 现有 adapter）：
#   python el/sft_CoT.py \
#       --output_dir result_CoT \
#       --bf16 True \
#       --adapter_name_or_path result_mammo/checkpoint-17484 \
#       --per_device_train_batch_size 2 \
#       --gradient_accumulation_steps 8 \
#       --learning_rate 1e-4 \
#       --num_train_epochs 1 \
#       --model_max_length 3072
#
# 示例（多卡，直接从 base 模型全量微调，不 continue 任何 adapter）：
#   torchrun --nproc_per_node=2 el/sft_CoT.py --output_dir result_CoT_from_base --bf16 True \
#       --model_name_or_path meta-llama/Llama-3.1-8B ...
#   （全量微调显存开销大，建议加上 FSDP，见文件末尾说明）
#
# 示例（多卡，直接从 base 模型训练全新 LoRA）：
#   torchrun --nproc_per_node=2 el/sft_CoT.py --output_dir result_CoT_from_base_lora --bf16 True \
#       --model_name_or_path meta-llama/Llama-3.1-8B --lora_only True ...
#
# continue 现有 adapter 时，默认可训练参数 = LoRA(q/k/v/o) + --trainable_params（默认 "embed,norm"）。
# 只想训 LoRA、其余（含 embed/norm）保持冻结时，加上 --lora_only True：
#   torchrun --nproc_per_node=2 el/sft_CoT.py --output_dir result_CoT --bf16 True \
#       --adapter_name_or_path result_mammo/checkpoint-17484 --lora_only True ...
#
# 全量微调 8B 模型显存提示：单卡很难放下 全量 fp32 优化器状态+梯度+权重，
# 建议在多卡命令后追加类似：
#   --fsdp "full_shard auto_wrap" --fsdp_transformer_layer_cls_to_wrap LlamaDecoderLayer
# （这两个参数是 transformers.TrainingArguments 自带的，无需改代码。）

import json
import logging
import math
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import transformers
from torch.utils.data import Dataset, Subset
from transformers import Trainer
from peft import LoraConfig, PeftModel, get_peft_model

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_EL_DIR = Path(__file__).resolve().parent
if str(_EL_DIR) not in sys.path:
    sys.path.insert(0, str(_EL_DIR))

# Prompt 直接复用 data_aug 的带 Reasoning requirements 版本（key 兼容 input_seg/input + question）。
from data_aug.augment_ent_link_thinking import build_prompt  # noqa: E402

# 复用 el/sft.py 中与 build_prompt 无关的工具，避免重复实现。
from el.sft import (  # noqa: E402
    IGNORE_INDEX,
    DEFAULT_EOS_TOKEN,
    DEFAULT_PAD_TOKEN,
    DEFAULT_BOS_TOKEN,
    DEFAULT_UNK_TOKEN,
    jload,
    preprocess,
    smart_tokenizer_and_embedding_resize,
    _tokenize_lens_no_trunc,
    _normalize_description_marker,
    _normalize_optional_limit,
    _load_eval_records,
    _apply_eval_table_compression,
    _has_think_tags,
    DataCollatorForSupervisedDataset,
    ValidationGenerativeEvalCallback as _BaseValGenEvalCallback,
)

DEFAULT_BASE_MODEL = "meta-llama/Llama-3.1-8B"
DEFAULT_ADAPTER_PATH = str(_PROJECT_ROOT / "result_mammo" / "checkpoint-17484")
DEFAULT_DATA_PATH = "/DATA1/khli/t&m/merged_CoTSFT_20k&10k.json"
# CoT 目标比纯答案长很多，默认上调最大长度（input_seg 已简化，3072 通常够用）。
DEFAULT_MODEL_MAX_LENGTH = 3072
DEFAULT_MAX_OUTPUT_TOKENS = 512
THINK_OPEN_TAG = "<think>"
THINK_CLOSE_TAG = "</think>"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=DEFAULT_BASE_MODEL,
        metadata={"help": "Base model to load before applying the SFT adapter."},
    )
    adapter_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional path to an already-trained SFT LoRA adapter to continue from "
            "(e.g. result_mammo/checkpoint-17484). If omitted (or empty), training starts "
            "directly from --model_name_or_path: full-parameter fine-tuning by default, or "
            "a freshly-initialized LoRA adapter when --lora_only True is also passed."
        },
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default=DEFAULT_DATA_PATH,
        metadata={"help": "Path to the CoT training data (json array)."},
    )
    eval_data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional JSON/JSONL path for generative validation."},
    )
    eval_data_limit: Optional[int] = field(
        default=None,
        metadata={"help": "Optional max samples for eval_data_path evaluation."},
    )
    max_output_tokens: int = field(
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        metadata={"help": "Drop samples whose output token length is greater than this value."},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=DEFAULT_MODEL_MAX_LENGTH,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    use_flash_attn: bool = field(
        default=True,
        metadata={"help": "Whether use flash attention for training."},
    )
    trainable_params: str = field(
        default="embed,norm",
        metadata={"help": "Additional trainable parameters except LoRA weights."},
    )
    lora_only: bool = field(
        default=False,
        metadata={
            "help": "If True, only train LoRA adapter weights and keep everything else "
            "(including --trainable_params, e.g. embed/norm) frozen. When continuing from "
            "--adapter_name_or_path, this just skips unfreezing --trainable_params. When "
            "--adapter_name_or_path is omitted, this instead initializes a fresh LoRA(q/k/v/o) "
            "adapter on top of --model_name_or_path and trains only that adapter. Default False: "
            "when continuing an adapter it also unfreezes --trainable_params (existing behavior); "
            "when no adapter is given it means full-parameter fine-tuning from the base model."
        },
    )
    eval_strategy: str = field(default="no")
    eval_steps: int = field(default=50)
    logging_steps: int = field(default=20)
    save_strategy: str = field(default="steps")
    save_steps: int = field(default=2000)
    ddp_find_unused_parameters: bool = field(default=False)


class CoTSupervisedDataset(Dataset):
    """Supervised dataset for CoT SFT.

    与 el/sft.py 的 SupervisedDataset 等价，但使用带 Reasoning requirements 的
    thinking prompt，且 target 为含 <think>...</think> 的完整 output。
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        max_output_tokens: int,
        dataset_name: str = "dataset",
        compress_input_for_eval_prompt: bool = False,
    ):
        super().__init__()
        if max_output_tokens <= 0:
            raise ValueError(f"max_output_tokens must be > 0, got {max_output_tokens}")
        logging.warning(f"[{dataset_name}] Loading data from {data_path} ...")
        list_data_dict = jload(os.path.expanduser(data_path))

        if compress_input_for_eval_prompt:
            list_data_dict = [_apply_eval_table_compression(ex) for ex in list_data_dict]

        # 与训练/评测保持一致：[DESCRIPTION] -> [DESC]
        for ex in list_data_dict:
            for key in ("question", "output"):
                if key in ex:
                    ex[key] = _normalize_description_marker(ex[key])

        logging.warning("Formatting inputs (thinking prompt)...")
        sources = [build_prompt(example) for example in list_data_dict]
        targets = [f"{example['output']}{DEFAULT_EOS_TOKEN}" for example in list_data_dict]
        target_full_lens = _tokenize_lens_no_trunc(targets, tokenizer)

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        max_len = int(tokenizer.model_max_length)
        all_input_ids = data_dict["input_ids"]
        all_labels = data_dict["labels"]
        all_full_lens = data_dict["example_full_lens"]

        self.input_ids: List[torch.Tensor] = []
        self.labels: List[torch.Tensor] = []
        dropped_all_ignored = 0
        dropped_truncated_target = 0
        dropped_output_too_long = 0
        total = len(all_input_ids)

        for input_ids, labels, full_len, output_len in zip(
            all_input_ids, all_labels, all_full_lens, target_full_lens
        ):
            if output_len > max_output_tokens:
                dropped_output_too_long += 1
                continue
            if full_len > max_len:
                dropped_truncated_target += 1
                continue
            if not torch.any(labels.ne(IGNORE_INDEX)):
                dropped_all_ignored += 1
                continue
            self.input_ids.append(input_ids)
            self.labels.append(labels)

        dropped_total = dropped_all_ignored + dropped_truncated_target + dropped_output_too_long
        kept = len(self.input_ids)
        filtered_ratio = (dropped_total / total) if total else 0.0
        logging.warning(
            f"[{dataset_name}] kept {kept}/{total}; filtered {dropped_total}/{total} ({filtered_ratio:.2%}) "
            f"(output_gt_{max_output_tokens}={dropped_output_too_long}, "
            f"truncated_target={dropped_truncated_target}, all_ignore={dropped_all_ignored})"
        )
        if kept == 0:
            raise ValueError(
                f"[{dataset_name}] no valid samples left after filtering in {os.path.expanduser(data_path)}; "
                f"consider increasing --model_max_length (current={max_len})."
            )

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


class CoTValidationGenerativeEvalCallback(_BaseValGenEvalCallback):
    """复用基类的生成评测逻辑，但用 thinking prompt 构造输入，且放宽 max_new_tokens。"""

    def __init__(
        self,
        eval_data: List[Dict[str, Any]],
        tokenizer: transformers.PreTrainedTokenizer,
        save_dir: str,
        output_dir: str,
        save_total_limit: int = 4,
        max_new_tokens: int = 128,
    ):
        super().__init__(
            eval_data=eval_data,
            tokenizer=tokenizer,
            save_dir=save_dir,
            output_dir=output_dir,
            save_total_limit=save_total_limit,
            max_new_tokens=max_new_tokens,
        )

    def _checkpoint_dir(self, step: int) -> Path:
        return self.output_dir / f"checkpoint-{step}"

    def _save_checkpoint(self, model, step: int, metadata: Dict[str, Any]) -> Path:
        ckpt_dir = self._checkpoint_dir(step)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(ckpt_dir))
        self.tokenizer.save_pretrained(str(ckpt_dir))
        with (ckpt_dir / "checkpoint_meta.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return ckpt_dir

    def _delete_checkpoint_dir(self, step: int) -> None:
        ckpt_dir = self._checkpoint_dir(step)
        if ckpt_dir.is_dir():
            shutil.rmtree(ckpt_dir)

    def _update_val_checkpoints(
        self,
        model,
        step: int,
        acc: float,
        n_correct: int,
        n_total: int,
    ) -> None:
        val_slots = max(0, self.save_total_limit - 1)
        if val_slots == 0:
            return

        self._val_ckpt_records = [(a, s) for a, s in self._val_ckpt_records if s != step]
        self._val_ckpt_records.append((acc, step))
        self._val_ckpt_records.sort(key=lambda x: (-x[0], -x[1]))
        self._val_ckpt_records = self._val_ckpt_records[:val_slots]
        kept_steps = {s for _, s in self._val_ckpt_records}

        for rank, (record_acc, record_step) in enumerate(self._val_ckpt_records, start=1):
            if record_step not in kept_steps:
                continue
            meta = {
                "type": "validation",
                "step": record_step,
                "accuracy": record_acc,
                "rank": rank,
            }
            if record_step == step:
                meta["n_correct"] = n_correct
                meta["n_total"] = n_total
            if record_step not in self._saved_ckpt_steps or record_step == step:
                ckpt_dir = self._save_checkpoint(model, record_step, meta)
                self._saved_ckpt_steps.add(record_step)
                print(
                    f"[CoTValGenEval step {step}] kept validation checkpoint "
                    f"(rank={rank}, acc={record_acc:.4f}) -> {ckpt_dir}"
                )

        for old_step in list(self._saved_ckpt_steps):
            if old_step not in kept_steps:
                self._delete_checkpoint_dir(old_step)
                self._saved_ckpt_steps.discard(old_step)
                print(f"[CoTValGenEval step {step}] removed evicted validation checkpoint-{old_step}")

    @torch.no_grad()
    def _run_and_save(self, model, step: int):
        # 直接复用基类实现，差异仅在 build_prompt（本模块已 import thinking 版本，
        # 但基类引用的是其自身模块内的 build_prompt），因此这里重写以注入 thinking prompt。
        import contextlib
        from tqdm import tqdm
        from el.sft import _apply_cot_candidate_constraint, _normalize_pred_text

        model.eval()
        had_gradient_ckpt = getattr(model, "is_gradient_checkpointing", False)
        if had_gradient_ckpt:
            model.gradient_checkpointing_disable()
        device = next(model.parameters()).device
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = eos_token_id

        rows: List[Dict[str, Any]] = []
        for ex in tqdm(self.eval_data, desc=f"CoTValGenEval step {step}", leave=False, ncols=120):
            prompt = build_prompt(ex)
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=False).to(device)
            seq_len = inputs["input_ids"].shape[1]
            max_pos = getattr(model.config, "max_position_embeddings", 131072)
            if seq_len + self.max_new_tokens > max_pos:
                keep = max_pos - self.max_new_tokens
                inputs = {k: v[:, -keep:] for k, v in inputs.items()}
            autocast_ctx = (
                torch.autocast(device_type=device.type, dtype=torch.bfloat16)
                if device.type in {"cuda", "cpu"}
                else contextlib.nullcontext()
            )
            with autocast_ctx:
                # 纯 greedy：不要用 repetition_penalty / no_repeat_ngram_size。
                # 正确答案就是 prompt 里逐字列出的候选之一，抑制重复会禁止模型照抄，
                # 逼它改写成非标准格式（如 DESCRIPTION="..."），导致解析失败、准确率虚假为 0。
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    min_new_tokens=2,
                    do_sample=False,
                    use_cache=True,
                    eos_token_id=int(eos_token_id) if eos_token_id is not None else None,
                    pad_token_id=pad_token_id,
                )
            new_ids = output_ids[0, inputs["input_ids"].shape[1]:]
            raw_pred = self.tokenizer.decode(new_ids, skip_special_tokens=True)
            # 优先提取 </think> 之后的实体；无闭合 tag 时回退到直接解析整段原始输出。
            # 是否有 <think> 只作为统计量（has_think），不再影响 correct 的判定。
            prediction = _apply_cot_candidate_constraint(ex.get("question", ""), raw_pred)
            gold = _normalize_pred_text(ex.get("output") or "")
            prediction_match = int(prediction == gold)
            has_think = int(_has_think_tags(raw_pred))
            rows.append(
                {
                    "table": ex.get("table"),
                    "cell": ex.get("cell"),
                    "prompt": prompt,
                    "gold": gold,
                    "raw_prediction": raw_pred,
                    "prediction": prediction,
                    "prediction_match": prediction_match,
                    "has_think": has_think,
                    "correct": prediction_match,
                }
            )

        self.save_dir.mkdir(parents=True, exist_ok=True)
        pred_path = self.save_dir / f"val_predictions_step_{step:06d}.jsonl"
        with pred_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        n_total = len(rows)
        n_correct = sum(r["correct"] for r in rows)
        n_prediction_match = sum(r["prediction_match"] for r in rows)
        n_has_think = sum(r["has_think"] for r in rows)
        acc = (n_correct / n_total) if n_total else 0.0
        metrics_path = self.save_dir / f"val_metrics_step_{step:06d}.json"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "step": step,
                    "accuracy": acc,
                    "n_correct": n_correct,
                    "n_total": n_total,
                    "n_prediction_match": n_prediction_match,
                    "n_has_think": n_has_think,
                    "predictions_file": str(pred_path),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(
            f"[CoTValGenEval step {step}] accuracy={acc:.4f} ({n_correct}/{n_total}); "
            f"prediction_match={n_prediction_match}/{n_total}, has_think={n_has_think}/{n_total}"
        )
        print(f"[CoTValGenEval step {step}] saved predictions -> {pred_path}")
        self._log_accuracy_to_wandb(step=step, acc=acc, n_correct=n_correct, n_total=n_total)
        self._update_val_checkpoints(
            model=model,
            step=step,
            acc=acc,
            n_correct=n_correct,
            n_total=n_total,
        )

        if had_gradient_ckpt:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        model.train()
        return acc

    def _log_accuracy_to_wandb(self, step: int, acc: float, n_correct: int, n_total: int) -> None:
        """Log generative validation accuracy to Weights & Biases when enabled."""
        try:
            import wandb  # type: ignore
        except Exception:
            return
        if getattr(wandb, "run", None) is None:
            return
        wandb.log(
            {
                "val_gen/accuracy": acc,
                "val_gen/n_correct": n_correct,
                "val_gen/n_total": n_total,
            },
            step=step,
        )

    @torch.no_grad()
    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        # 仅在 rank0 跑生成式评测；其它 rank 在下面的 broadcast 处同步等待，
        # 既避免重复生成，又能把 accuracy 同步给所有进程用于 best-model 选择。
        acc: Optional[float] = None
        if model is not None and self.eval_data and state.is_world_process_zero:
            step = int(state.global_step or 0)
            if self._last_logged_step != step:
                self._last_logged_step = step
                acc = self._run_and_save(model=model, step=step)

        if dist.is_available() and dist.is_initialized():
            device = (
                next(model.parameters()).device if model is not None else torch.device("cpu")
            )
            acc_tensor = torch.tensor(
                [acc if acc is not None else -1.0], dtype=torch.float32, device=device
            )
            dist.broadcast(acc_tensor, src=0)
            broadcasted = float(acc_tensor.item())
            acc = broadcasted if broadcasted >= 0 else None

        # 写回 Trainer 的 metrics，供日志/监控使用。
        if metrics is not None and acc is not None:
            metrics["eval_accuracy"] = acc

    @torch.no_grad()
    def on_train_end(self, args, state, control, model=None, **kwargs):
        if not state.is_world_process_zero or model is None:
            return
        step = int(state.global_step or 0)
        ckpt_dir = self._save_checkpoint(
            model,
            step,
            {
                "type": "final",
                "step": step,
            },
        )
        self._saved_ckpt_steps.add(step)
        print(f"[CoTValGenEval] saved final checkpoint -> {ckpt_dir}")


def make_supervised_data_module(tokenizer, data_args) -> Dict:
    max_output_tokens = int(getattr(data_args, "max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS))
    train_dataset = CoTSupervisedDataset(
        tokenizer=tokenizer,
        data_path=data_args.data_path,
        max_output_tokens=max_output_tokens,
        dataset_name="train",
    )
    eval_dataset = None
    if getattr(data_args, "eval_data_path", None):
        eval_dataset = CoTSupervisedDataset(
            tokenizer=tokenizer,
            data_path=os.path.expanduser(data_args.eval_data_path),
            max_output_tokens=max_output_tokens,
            dataset_name="eval",
            compress_input_for_eval_prompt=True,
        )
        eval_limit = _normalize_optional_limit(getattr(data_args, "eval_data_limit", None))
        if eval_limit is not None and eval_limit < len(eval_dataset):
            eval_dataset = Subset(eval_dataset, list(range(eval_limit)))
            logging.warning(f"Applied eval_data_limit={eval_limit} to eval dataset.")
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    adapter_path: Optional[str] = None
    if model_args.adapter_name_or_path:
        adapter_path = os.path.expanduser(model_args.adapter_name_or_path)
        if not os.path.isdir(adapter_path):
            raise SystemExit(f"Adapter dir not found: {adapter_path}")

    if adapter_path is not None:
        # tokenizer 从 adapter checkpoint 加载：已含 [PAD]/<unk> 等扩充 token，
        # 保证与已保存的 embed/lm_head（vocab 已 resize）尺寸一致。
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            adapter_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
    else:
        # 没有 adapter：直接从 base 模型加载 tokenizer，后面按需扩充特殊 token。
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    attn_impl = "flash_attention_2" if training_args.use_flash_attn else "eager"
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        dtype=torch.bfloat16 if training_args.bf16 else None,
        attn_implementation=attn_impl,
    )

    if adapter_path is not None:
        # 先 resize 到与 adapter 保存的 embedding 一致（不要 pad 到 64 的倍数，否则尺寸不匹配）。
        if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
            # Adapter 中会覆盖 embed/lm_head，这里只做尺寸对齐，关闭 mean_resizing 可避免无意义告警。
            model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

        # 加载已训练好的 SFT adapter，并以可训练模式继续训练。
        logging.warning(f"Loading SFT adapter (trainable) from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)

        # 继续放开 embed/norm 等额外可训练参数（与原 SFT 一致）；
        # --lora_only True 时跳过这一步，只训练 LoRA adapter 权重，其余（含 embed/norm）保持冻结。
        if training_args.lora_only:
            logging.warning(
                f"lora_only=True: skip unfreezing --trainable_params={training_args.trainable_params!r}; "
                "only LoRA adapter weights are trainable."
            )
        else:
            extra_keys = [k for k in training_args.trainable_params.split(",") if k]
            if extra_keys:
                for n, p in model.named_parameters():
                    if any(k in n for k in extra_keys):
                        p.requires_grad_(True)
    else:
        # 没有 adapter：直接在 base 模型上开始训练，先按需扩充特殊 token 并 resize embedding
        # （与 el/sft.py 的 smart_tokenizer_and_embedding_resize 行为一致，新增行做均值初始化）。
        special_tokens_dict: Dict[str, str] = {}
        if tokenizer.pad_token is None:
            special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
        if tokenizer.eos_token is None:
            special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
        if tokenizer.bos_token is None:
            special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
        if tokenizer.unk_token is None:
            special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=special_tokens_dict,
            tokenizer=tokenizer,
            model=model,
        )

        if training_args.lora_only:
            logging.warning(
                f"No --adapter_name_or_path given: initializing a fresh LoRA(q/k/v/o) adapter "
                f"on top of {model_args.model_name_or_path}; only LoRA weights are trainable "
                f"(--trainable_params={training_args.trainable_params!r} stays frozen)."
            )
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)
        else:
            logging.warning(
                f"No --adapter_name_or_path given and lora_only=False: doing full-parameter "
                f"fine-tuning of {model_args.model_name_or_path} (all weights trainable)."
            )

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "config"):
        model.config.use_cache = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logging.warning(f"Trainable params: {trainable:,} / {total:,} ({trainable / total:.4%})")

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    if data_module["eval_dataset"] is None and getattr(training_args, "eval_strategy", "no") != "no":
        logging.warning("No eval dataset (--eval_data_path missing); setting eval_strategy='no'.")
        training_args.eval_strategy = "no"

    callbacks = []
    if data_args.eval_data_path:
        eval_limit = _normalize_optional_limit(data_args.eval_data_limit)
        val_gen_data = _load_eval_records(data_args.eval_data_path, limit=eval_limit)
        # 与训练 prompt 对齐，消除 train/inference skew：
        #   1) 压缩 input_seg（与训练 eval dataset 的 compress_input_for_eval_prompt 一致）
        #   2) question/output 做 [DESCRIPTION] -> [DESC] 归一化（训练数据也做了同样处理）
        aligned_val_gen_data = []
        for ex in val_gen_data:
            ex = _apply_eval_table_compression(ex)
            for key in ("question", "output"):
                if key in ex:
                    ex[key] = _normalize_description_marker(ex[key])
            aligned_val_gen_data.append(ex)
        val_gen_data = aligned_val_gen_data
        if val_gen_data:
            if training_args.save_total_limit is None:
                training_args.save_total_limit = 4
            # Checkpoint 由 callback 管理：save_total_limit-1 个 val top + 1 个 final。
            training_args.save_strategy = "no"
            callbacks.append(
                CoTValidationGenerativeEvalCallback(
                    eval_data=val_gen_data,
                    tokenizer=tokenizer,
                    save_dir=os.path.join(training_args.output_dir, "val_eval"),
                    output_dir=training_args.output_dir,
                    save_total_limit=training_args.save_total_limit,
                    max_new_tokens=768,
                )
            )
            logging.warning(
                f"CoTValidationGenerativeEvalCallback: {len(val_gen_data)} samples from {data_args.eval_data_path}; "
                f"checkpoint policy: top {training_args.save_total_limit - 1} validation + 1 final "
                f"(save_total_limit={training_args.save_total_limit})"
            )

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        callbacks=callbacks,
        **data_module,
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_state()
    if not callbacks:
        trainer.save_model(output_dir=training_args.output_dir)
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    train()
