from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import Trainer, TrainingArguments

from tsrlm.data.dataset import TSReportDataset
from tsrlm.data.collator import TSDataCollator
from tsrlm.models.ts_llm import TSReportLM


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_jsonl", type=str, required=True)
    p.add_argument("--eval_jsonl", type=str, default=None)
    p.add_argument("--llm_name_or_path", type=str, required=True)

    # TS encoder
    p.add_argument("--ts_num_vars", type=int, default=1)
    p.add_argument("--ts_patch_len", type=int, default=16)
    p.add_argument("--ts_d_model", type=int, default=256)
    p.add_argument("--ts_layers", type=int, default=4)
    p.add_argument("--ts_heads", type=int, default=8)

    # Bridge
    p.add_argument("--prefix_tokens", type=int, default=32)

    # Training
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--num_train_epochs", type=float, default=3.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--per_device_eval_batch_size", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--max_text_length", type=int, default=512)
    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")

    # Ablations
    p.add_argument("--freeze_llm", action="store_true")
    p.add_argument("--use_revin", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    model = TSReportLM(
        llm_name_or_path=args.llm_name_or_path,
        ts_num_vars=args.ts_num_vars,
        ts_patch_len=args.ts_patch_len,
        ts_d_model=args.ts_d_model,
        ts_layers=args.ts_layers,
        ts_heads=args.ts_heads,
        prefix_tokens=args.prefix_tokens,
        freeze_llm=args.freeze_llm,
        use_revin=args.use_revin,
        trust_remote_code=True,
    )

    train_ds = TSReportDataset(args.train_jsonl)
    eval_ds = TSReportDataset(args.eval_jsonl) if args.eval_jsonl else None

    collator = TSDataCollator(tokenizer=model.tokenizer, max_text_length=args.max_text_length)

    training_args = TrainingArguments(
        output_dir=str(outdir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        evaluation_strategy="steps" if eval_ds is not None else "no",
        eval_steps=args.eval_steps if eval_ds is not None else None,
        save_total_limit=3,
        fp16=args.fp16,
        bf16=args.bf16,
        remove_unused_columns=False,  # important: we pass custom tensors
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(str(outdir / "final_model"))

    # save tokenizer too (useful for generation)
    model.tokenizer.save_pretrained(str(outdir / "final_model"))


if __name__ == "__main__":
    main()
