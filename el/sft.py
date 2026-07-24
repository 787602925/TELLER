# Some code based on https://github.com/epfml/landmark-attention
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import os
import re
import sys
import copy
import json
import math
import logging
import contextlib
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import transformers
from torch.utils.data import Dataset, Subset
from transformers import Trainer, TrainerCallback
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_EL_DIR = Path(__file__).resolve().parent
if str(_EL_DIR) not in sys.path:
    sys.path.insert(0, str(_EL_DIR))

try:
    from data_clean.single_data_train import _simplify_input_seg as _compress_input_seg_like_eval
except Exception:
    _compress_input_seg_like_eval = None


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

DEFAULT_MODEL_NAME = "meta-llama/Llama-3.1-8B"
DEFAULT_DATA_PATH = str(Path.home() / "DATA" / "t&m" / "merged_SFT_70k&35k.json")
DEFAULT_MODEL_MAX_LENGTH = 2048


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


def build_prompt(ex: Dict[str, Any]) -> str:
    """与 el/token_estimate.build_prompt 相同版式：Input 为 `input_seg`（或回退 `input`）。"""
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


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _normalize_description_marker(text: str) -> str:
    if not isinstance(text, str):
        return text
    return text.replace("[DESCRIPTION]", "[DESC]")


def _extract_post_think_candidate(text: str) -> Optional[str]:
    """
    For CoT outputs, prefer extracting the final entity candidate after </think>.
    This avoids mistakenly taking a candidate mention inside reasoning as the answer.
    """
    if not isinstance(text, str):
        return None
    s = text.strip()
    if not s:
        return None
    lower_s = s.lower()
    close_tag = "</think>"
    close_idx = lower_s.rfind(close_tag)
    if close_idx == -1:
        return None
    post = s[close_idx + len(close_tag) :].strip()
    if not post:
        return None
    post = _normalize_description_marker(post)
    post = re.sub(r"\[\s*desc\s*\]", "[DESC]", post, flags=re.I)
    post = re.sub(r"\[\s*description\s*\]", "[DESC]", post, flags=re.I)
    post = re.sub(r"\[\s*type\s*\]", "[TYPE]", post, flags=re.I)
    post = re.sub(r"\s+", " ", post).strip()
    patterns = [
        r"<\s*[^<>]+?\s*\[DESC\]\s*[^<>]+?\s*\[TYPE\]\s*[^<>]+?>",
        r"<\s*[^<>]+?\s*\[DESC\]\s*[^<>]+?>",
        r"<\s*[^<>]+?>",
    ]
    for pat in patterns:
        matches = list(re.finditer(pat, post, flags=re.I | re.S))
        if matches:
            # Use the last candidate after </think> to stay close to "final answer".
            return matches[-1].group(0)
    return None


def _normalize_pred_text(text: str) -> str:
    """
    Canonicalize prediction text with format-only normalization:
    - normalize special markers/casing/spacing
    - extract candidate-like payload from noisy generations
    - remove boundary angle-bracket artifacts
    - remove quote artifacts around entity span
    - output canonical outer "<...>" form
    """
    if not isinstance(text, str):
        return ""
    s = text.strip()
    s = re.sub(r"</?s>", " ", s, flags=re.I)

    # CoT-aware path: prefer the final entity after </think>.
    post_think_candidate = _extract_post_think_candidate(s)
    if post_think_candidate is not None:
        s = post_think_candidate

    s = _normalize_description_marker(s)
    s = re.sub(r"\[\s*desc\s*\]", "[DESC]", s, flags=re.I)
    s = re.sub(r"\[\s*description\s*\]", "[DESC]", s, flags=re.I)
    s = re.sub(r"\[\s*type\s*\]", "[TYPE]", s, flags=re.I)
    s = re.sub(r"\s+", " ", s).strip()

    # Prefer extracting a full candidate shape to avoid trailing XML-ish noise.
    patterns = [
        r"<\s*[^<>]+?\s*\[DESC\]\s*[^<>]+?\s*\[TYPE\]\s*[^<>]+?>",
        r"<\s*[^<>]+?\s*\[DESC\]\s*[^<>]+?>",
        r"<\s*[^<>]+?>",
    ]
    extracted = None
    for pat in patterns:
        m = re.search(pat, s, flags=re.I | re.S)
        if m:
            extracted = m.group(0)
            break

    if extracted is not None:
        s = extracted
    elif "<" in s:
        # Fallback: keep content from first "<" and force close bracket.
        s = s[s.find("<") :].strip()
        if not s.endswith(">"):
            s = f"{s}>"

    s = re.sub(r"\s+", " ", s).strip()
    core = s.strip(" <>")
    if not core:
        return ""
    marker = re.search(r"\[(DESC|DESCRIPTION|TYPE)\]", core, flags=re.I)
    if marker:
        entity = core[: marker.start()].strip()
        tail = core[marker.start() :].strip()
    else:
        entity = core.strip()
        tail = ""

    # Remove quote artifacts around entity mention only.
    if len(entity) >= 2 and (
        (entity[0] == "'" and entity[-1] == "'")
        or (entity[0] == '"' and entity[-1] == '"')
    ):
        entity = entity[1:-1].strip()
    else:
        if entity.startswith(("'", '"')):
            entity = entity[1:].strip()
        if entity.endswith(("'", '"')):
            entity = entity[:-1].strip()

    merged = f"{entity} {tail}".strip() if tail else entity
    merged = re.sub(r"\s+", " ", merged).strip()
    if not merged:
        return ""
    merged = re.sub(r"\s+\[DESC\]", " [DESC]", merged)
    merged = re.sub(r"\s+\[TYPE\]", " [TYPE]", merged)
    return f"<{merged}>"


THINK_OPEN_TAG = "<think>"
THINK_CLOSE_TAG = "</think>"


def _has_think_tags(raw_pred: str) -> bool:
    """True when raw generation contains both CoT think open/close tags."""
    return isinstance(raw_pred, str) and THINK_OPEN_TAG in raw_pred and THINK_CLOSE_TAG in raw_pred


def _normalize_cot_pred_text(text: str) -> str:
    """
    CoT prediction extraction: prefer the text after `</think>`. If the closing tag
    is missing (or there's nothing after it), fall back to parsing the entity pattern
    directly from the raw generation, so a correct final answer can still be credited
    even when the <think> block is malformed/absent. Whether the think tags were
    present is tracked separately (see `_has_think_tags`), not baked into extraction.
    """
    if not isinstance(text, str):
        return ""
    s = text.strip()
    if not s:
        return ""
    close_idx = s.lower().rfind(THINK_CLOSE_TAG.lower())
    if close_idx != -1:
        post = s[close_idx + len(THINK_CLOSE_TAG) :].strip()
        if post:
            return _normalize_pred_text(post)
    # No closing tag (or empty tail): fall back to parsing the whole raw text.
    return _normalize_pred_text(s)


def _apply_candidate_constraint(question: str, prediction: str) -> str:
    """
    Apply format-only normalization and optional exact candidate canonicalization.

    Important: this function does NOT do semantic correction or fuzzy matching.
    It only resolves pure formatting differences.
    """
    parts = _split_question_candidates(question)
    pred_norm = _normalize_pred_text(prediction)
    if parts is None or not pred_norm:
        return pred_norm
    _, candidates, _ = parts
    if not candidates:
        return pred_norm

    # Canonicalize to an existing candidate only when format-normalized forms
    # are exactly equal; avoid any content-level remapping.
    candidate_map: Dict[str, str] = {}
    for c in candidates:
        c_norm = _normalize_pred_text(c)
        if c_norm and c_norm not in candidate_map:
            candidate_map[c_norm] = c_norm
    return candidate_map.get(pred_norm, pred_norm)


def _apply_cot_candidate_constraint(question: str, raw_prediction: str) -> str:
    """Like _apply_candidate_constraint, but only reads prediction after </think>."""
    parts = _split_question_candidates(question)
    pred_norm = _normalize_cot_pred_text(raw_prediction)
    if parts is None or not pred_norm:
        return pred_norm
    _, candidates, _ = parts
    if not candidates:
        return pred_norm

    candidate_map: Dict[str, str] = {}
    for c in candidates:
        c_norm = _normalize_pred_text(c)
        if c_norm and c_norm not in candidate_map:
            candidate_map[c_norm] = c_norm
    return candidate_map.get(pred_norm, pred_norm)


def _split_question_candidates(question: str):
    if not isinstance(question, str) or not question:
        return None
    lower_q = question.lower()
    start_key = "the referent entity candidates are:"
    start_idx = lower_q.find(start_key)
    if start_idx == -1:
        return None
    start_idx += len(start_key)
    end_idx = lower_q.find("what is", start_idx)
    if end_idx == -1:
        end_idx = len(question)
    prefix = question[:start_idx]
    segment = question[start_idx:end_idx]
    suffix = question[end_idx:]
    candidates = re.findall(r"<[^>]+>", segment)
    if not candidates:
        return None
    return prefix, candidates, suffix


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=DEFAULT_MODEL_NAME)


@dataclass
class DataArguments:
    data_path: str = field(
        default=DEFAULT_DATA_PATH,
        metadata={"help": "Path to the training data."},
    )
    eval_data_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional JSON path for validation/evaluation (recommended: pre-compressed val/eval file). "
            "When set, use --evaluation_strategy steps and --eval_steps (or epoch) so Trainer runs eval and "
            "logs eval_loss (e.g. with --report_to wandb)."
        },
    )
    eval_data_limit: Optional[int] = field(
        default=None,
        metadata={"help": "Optional max samples for eval_data_path evaluation (loss + val generation)."},
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
    low_rank_training: bool = field(
        default=True,
        metadata={"help": "Whether use low rank adaptation for training."},
    )
    trainable_params: str = field(
        default="embed,norm",
        metadata={"help": "Additional trainable parameters except LoRA weights, if low rank training."},
    )
    eval_strategy: str = field(
        default="steps",
        metadata={"help": "Run evaluation on a step schedule by default."},
    )
    eval_steps: int = field(
        default=50,
        metadata={"help": "Evaluate every N steps."},
    )
    logging_steps: int = field(
        default=20,
        metadata={"help": "Log training metrics every N steps."},
    )
    ddp_find_unused_parameters: bool = field(
        default=False,
        metadata={"help": "Disable DDP unused parameter search for better performance."},
    )

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _tokenize_lens_no_trunc(
    strings: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> List[int]:
    lengths: List[int] = []
    for text in strings:
        ids = tokenizer(
            text,
            return_tensors="pt",
            truncation=False,
        ).input_ids[0]
        lengths.append(int(ids.shape[0]))
    return lengths


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    example_full_lens = _tokenize_lens_no_trunc(examples, tokenizer)
    return dict(
        input_ids=input_ids,
        labels=labels,
        example_full_lens=example_full_lens,
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        dataset_name: str = "dataset",
        compress_input_for_eval_prompt: bool = False,
    ):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = jload(os.path.expanduser(data_path))

        if compress_input_for_eval_prompt:
            list_data_dict = [_apply_eval_table_compression(ex) for ex in list_data_dict]

        for ex in list_data_dict:
            for key in ("question", "output"):
                if key in ex:
                    ex[key] = _normalize_description_marker(ex[key])

        logging.warning("Formatting inputs...")
        sources = [build_prompt(example) for example in list_data_dict]
        targets = [f"{example['output']}{DEFAULT_EOS_TOKEN}" for example in list_data_dict]

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
        total = len(all_input_ids)

        for input_ids, labels, full_len in zip(all_input_ids, all_labels, all_full_lens):
            if full_len > max_len:
                dropped_truncated_target += 1
                continue
            if not torch.any(labels.ne(IGNORE_INDEX)):
                dropped_all_ignored += 1
                continue
            self.input_ids.append(input_ids)
            self.labels.append(labels)

        dropped_total = dropped_all_ignored + dropped_truncated_target
        kept = len(self.input_ids)
        filtered_ratio = (dropped_total / total) if total else 0.0
        logging.warning(
            f"[{dataset_name}] kept {kept}/{total}; filtered {dropped_total}/{total} ({filtered_ratio:.2%}) "
            f"(truncated_target={dropped_truncated_target}, all_ignore={dropped_all_ignored})"
        )
        if kept == 0:
            raise ValueError(
                f"[{dataset_name}] no valid samples left after filtering in {os.path.expanduser(data_path)}"
            )

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


class ValidationGenerativeEvalCallback(TrainerCallback):
    """On each trainer evaluate event: generate predictions on eval_data_path and save JSONL."""

    def __init__(
        self,
        eval_data: List[Dict[str, Any]],
        tokenizer: transformers.PreTrainedTokenizer,
        save_dir: str,
        output_dir: str,
        save_total_limit: int = 4,
        max_new_tokens: int = 128,
    ):
        self.eval_data = [_apply_eval_table_compression(ex) for ex in eval_data]
        self.tokenizer = tokenizer
        self.save_dir = Path(save_dir).expanduser().resolve()
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.save_total_limit = max(1, int(save_total_limit))
        self.max_new_tokens = max_new_tokens
        self._last_logged_step: Optional[int] = None
        self._val_ckpt_records: List[Tuple[float, int]] = []
        self._saved_ckpt_steps: set[int] = set()

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
                    f"[ValGenEval step {step}] kept validation checkpoint "
                    f"(rank={rank}, acc={record_acc:.4f}) -> {ckpt_dir}"
                )

        for old_step in list(self._saved_ckpt_steps):
            if old_step not in kept_steps:
                self._delete_checkpoint_dir(old_step)
                self._saved_ckpt_steps.discard(old_step)
                print(f"[ValGenEval step {step}] removed evicted validation checkpoint-{old_step}")

    def _eos_token_id_for_generation(self, eos_token_id: Optional[int]):
        # Keep generation stop criterion conservative: only stop at model EOS.
        # Treating ">" as EOS can truncate malformed outputs and reinforce "<<..." patterns.
        return int(eos_token_id) if eos_token_id is not None else None

    @torch.no_grad()
    def _run_and_save(self, model, step: int):
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
        for ex in tqdm(self.eval_data, desc=f"ValGenEval step {step}", leave=False, ncols=120):
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
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    min_new_tokens=2,
                    do_sample=False,
                    use_cache=True,
                    eos_token_id=self._eos_token_id_for_generation(eos_token_id),
                    pad_token_id=pad_token_id,
                )
            new_ids = output_ids[0, inputs["input_ids"].shape[1]:]
            raw_pred = self.tokenizer.decode(new_ids, skip_special_tokens=True)
            prediction = _apply_candidate_constraint(ex.get("question", ""), raw_pred)
            gold = _normalize_pred_text(ex.get("output") or "")
            rows.append(
                {
                    "table": ex.get("table"),
                    "cell": ex.get("cell"),
                    "prompt": prompt,
                    "gold": gold,
                    "raw_prediction": raw_pred,
                    "prediction": prediction,
                    "correct": int(prediction == gold),
                }
            )

        self.save_dir.mkdir(parents=True, exist_ok=True)
        pred_path = self.save_dir / f"val_predictions_step_{step:06d}.jsonl"
        with pred_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        n_total = len(rows)
        n_correct = sum(r["correct"] for r in rows)
        acc = (n_correct / n_total) if n_total else 0.0
        metrics_path = self.save_dir / f"val_metrics_step_{step:06d}.json"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "step": step,
                    "accuracy": acc,
                    "n_correct": n_correct,
                    "n_total": n_total,
                    "predictions_file": str(pred_path),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"[ValGenEval step {step}] accuracy={acc:.4f} ({n_correct}/{n_total})")
        print(f"[ValGenEval step {step}] saved predictions -> {pred_path}")
        print(f"[ValGenEval step {step}] saved metrics -> {metrics_path}")
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
        if not state.is_world_process_zero or model is None or not self.eval_data:
            return
        step = int(state.global_step or 0)
        if self._last_logged_step == step:
            return
        self._last_logged_step = step
        acc = self._run_and_save(model=model, step=step)
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
        print(f"[ValGenEval] saved final checkpoint -> {ckpt_dir}")


def _load_jsonl(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    data = []
    with open(os.path.expanduser(path), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data[:limit] if limit is not None else data


def _normalize_optional_limit(limit: Optional[int]) -> Optional[int]:
    if limit is None:
        return None
    limit = int(limit)
    return limit if limit > 0 else None


def _extract_mam_row_cells(row_body: str) -> List[str]:
    parts = row_body.split("|")
    if len(parts) >= 2:
        return [c.strip() for c in parts[1:-1]]
    return [c.strip() for c in parts if c.strip()]


def _infer_mam_entity_from_input_and_question(
    input_seg: Any,
    question: Any,
) -> Optional[List[Any]]:
    if not isinstance(input_seg, str) or not isinstance(question, str):
        return None

    m_mention = re.search(
        r"selected entity mention in the table cell is:\s*['\"]?(.*?)['\"]?\.\s*The column name",
        question,
        flags=re.I | re.S,
    )
    m_col = re.search(r"The column name for .*? is\s*([^\.]+)\.", question, flags=re.I | re.S)
    if not m_mention or not m_col:
        return None

    mention = _norm_text(m_mention.group(1)).strip("'\"")
    col_name = _norm_text(m_col.group(1))
    if not mention or not col_name:
        return None

    m_col_idx = re.fullmatch(r"col\s*(\d+)", col_name, flags=re.I)
    if not m_col_idx:
        return None
    col_idx = int(m_col_idx.group(1))

    mention_lower = mention.lower()
    for part in input_seg.split("[SEP]"):
        m_row = re.search(r"row\s+(\d+)\s*:\s*(.*)$", part.strip(), flags=re.I | re.S)
        if not m_row:
            continue
        row_num = int(m_row.group(1))
        cells = _extract_mam_row_cells(m_row.group(2))
        if col_idx >= len(cells):
            continue
        cell_text = _norm_text(cells[col_idx]).lower()
        if mention_lower and mention_lower in cell_text:
            return [[row_num - 1, col_idx], mention]
    return None


def _apply_eval_table_compression(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build an in-memory eval view where only input_seg is compressed for prompting.
    candidates/question remain unchanged.
    """
    compressed = dict(example)
    if _compress_input_seg_like_eval is None:
        return compressed

    raw_input_seg = example.get("input_seg", example.get("input", ""))
    if not isinstance(raw_input_seg, str):
        raw_input_seg = str(raw_input_seg)

    entity_for_compression = example.get("entity")
    if entity_for_compression is None:
        entity_for_compression = _infer_mam_entity_from_input_and_question(
            raw_input_seg,
            example.get("question", ""),
        )

    simplified_input_seg = _compress_input_seg_like_eval(
        raw_input_seg,
        entity_for_compression,
        example.get("question", ""),
        drop_on_infer_failure=False,
    )
    if simplified_input_seg:
        compressed["input_seg"] = simplified_input_seg
    return compressed


def _load_eval_records(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    expanded = os.path.expanduser(path)
    if expanded.endswith(".jsonl"):
        return _load_jsonl(expanded, limit=limit)
    loaded = jload(expanded)
    if not isinstance(loaded, list):
        raise ValueError(f"Expected list in eval file: {expanded}")
    return loaded[:limit] if limit is not None else loaded


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(
        tokenizer=tokenizer,
        data_path=data_args.data_path,
        dataset_name="train",
    )
    eval_dataset = None
    if getattr(data_args, "eval_data_path", None):
        eval_dataset = SupervisedDataset(
            tokenizer=tokenizer,
            data_path=os.path.expanduser(data_args.eval_data_path),
            dataset_name="eval",
            compress_input_for_eval_prompt=True,
        )
        eval_limit = _normalize_optional_limit(getattr(data_args, "eval_data_limit", None))
        if eval_limit is not None and eval_limit < len(eval_dataset):
            eval_dataset = Subset(eval_dataset, list(range(eval_limit)))
            logging.warning(
                f"Applied eval_data_limit={eval_limit} to eval dataset from {data_args.eval_data_path}"
            )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    # Load model — use native flash_attention_2 (transformers >= 4.36)
    attn_impl = "flash_attention_2" if training_args.use_flash_attn else "eager"
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16 if training_args.bf16 else None,
        attn_implementation=attn_impl,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    special_tokens_dict = dict()
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

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    if data_module["eval_dataset"] is None and getattr(training_args, "eval_strategy", "no") != "no":
        logging.warning(
            "No eval dataset available (--eval_data_path not provided). Set eval_strategy='no'."
        )
        training_args.eval_strategy = "no"

    if training_args.low_rank_training:
        config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        # enable trainable params
        [p.requires_grad_() for n, p in model.named_parameters() if any([k in n for k in training_args.trainable_params.split(",")])]

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    callbacks = []
    if data_args.eval_data_path:
        eval_limit = _normalize_optional_limit(data_args.eval_data_limit)
        val_gen_data = _load_eval_records(data_args.eval_data_path, limit=eval_limit)
        if val_gen_data:
            if training_args.save_total_limit is None:
                training_args.save_total_limit = 4
            # Checkpoint 由 callback 管理：save_total_limit-1 个 val top + 1 个 final。
            training_args.save_strategy = "no"
            callbacks.append(
                ValidationGenerativeEvalCallback(
                    eval_data=val_gen_data,
                    tokenizer=tokenizer,
                    save_dir=os.path.join(training_args.output_dir, "val_eval"),
                    output_dir=training_args.output_dir,
                    save_total_limit=training_args.save_total_limit,
                )
            )
            logging.warning(
                f"ValidationGenerativeEvalCallback: {len(val_gen_data)} samples from {data_args.eval_data_path}; "
                f"checkpoint policy: top {training_args.save_total_limit - 1} validation + 1 final "
                f"(save_total_limit={training_args.save_total_limit})"
            )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        callbacks=callbacks,
        **data_module,
    )
    trainer.train()
    trainer.save_state()
    if not callbacks:
        trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()