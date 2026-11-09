"""LoRA fine-tuning on your own feedback.

Stage 1 (SFT): learn the edits you accepted / actually made.
Stage 2 (DPO, if enough pairs): prefer what you did over what you rejected.

Every run trains a fresh adapter on ALL your feedback (newest first, capped),
so the model never drifts away from what you liked earlier.

Run:  python -m mine.train --data <dir with sft.jsonl, dpo.jsonl> --out <adapter dir>
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer

BASE_MODEL = os.environ.get("MINE_BASE_MODEL", "continuedev/instinct")
MIN_DPO_PAIRS = int(os.environ.get("MINE_MIN_DPO_PAIRS", "20"))
MAX_LEN = int(os.environ.get("MINE_TRAIN_MAX_LEN", "6144"))
LORA_R = int(os.environ.get("MINE_LORA_R", "16"))


def count_lines(p: Path) -> int:
    return sum(1 for l in open(p) if l.strip()) if p.exists() else 0


def load_model(base: str):
    cuda = torch.cuda.is_available()
    kwargs = {}
    if cuda:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = {"": 0}
        kwargs["dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(base, **kwargs)
    model.config.use_cache = False
    if cuda:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_R * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    return get_peft_model(model, lora)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=BASE_MODEL)
    args = ap.parse_args()

    data, out = Path(args.data), Path(args.out)
    cuda = torch.cuda.is_available()
    n_sft, n_dpo = count_lines(data / "sft.jsonl"), count_lines(data / "dpo.jsonl")
    print(f"[train] base={args.base} sft={n_sft} dpo={n_dpo} cuda={cuda}", flush=True)
    if n_sft == 0:
        raise SystemExit("no training examples")

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = load_model(args.base)
    model.print_trainable_parameters()

    # Small datasets get more epochs; big ones fewer.
    epochs = 3 if n_sft < 200 else 2 if n_sft < 1000 else 1
    common = dict(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        bf16=cuda,
        gradient_checkpointing=cuda,
        lr_scheduler_type="cosine",
        warmup_steps=2,
    )

    # ---- Stage 1: SFT (loss only on the assistant completion)
    sft_ds = load_dataset("json", data_files=str(data / "sft.jsonl"), split="train")
    sft_cfg = SFTConfig(
        output_dir=str(data / "sft_ckpt"),
        num_train_epochs=epochs,
        learning_rate=1e-4,
        max_length=MAX_LEN,
        completion_only_loss=True,
        **common,
    )
    SFTTrainer(model=model, args=sft_cfg, train_dataset=sft_ds, processing_class=tok).train()

    # ---- Stage 2: DPO on (what you did) vs (what you rejected)
    if n_dpo >= MIN_DPO_PAIRS:
        print(f"[train] DPO on {n_dpo} preference pairs", flush=True)
        dpo_ds = load_dataset("json", data_files=str(data / "dpo.jsonl"), split="train")
        dpo_cfg = DPOConfig(
            output_dir=str(data / "dpo_ckpt"),
            num_train_epochs=1,
            learning_rate=5e-6,
            beta=0.1,
            max_length=MAX_LEN,
            **common,
        )
        # ref_model=None + PEFT model => the reference is the model with the adapter disabled
        DPOTrainer(model=model, ref_model=None, args=dpo_cfg, train_dataset=dpo_ds, processing_class=tok).train()
    else:
        print(f"[train] skipping DPO (need {MIN_DPO_PAIRS} pairs, have {n_dpo})", flush=True)

    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(str(tmp), selected_adapters=["default"])
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)
    (out / "mine_meta.json").write_text(json.dumps({"base": args.base, "sft": n_sft, "dpo": n_dpo}))
    print(f"[train] saved adapter to {out}", flush=True)


if __name__ == "__main__":
    main()
