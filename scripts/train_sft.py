from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from transformers import Trainer, TrainingArguments

from tsrlm.config import TSRLMConfig
from tsrlm.data import TSSFTDataset, TSSFTCollator
from tsrlm.models import TSReportLM


def parse_args():
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--train_jsonl", type=str, required=True)
    p.add_argument("--eval_jsonl", type=str, default=None)

    # llm
    p.add_argument("--llm_name_or_path", type=str, required=True)
    p.add_argument("--trust_remote_code", action="store_true")

    # architecture
    p.add_argument("--encoder_type", type=str, default="patchtst", choices=["patchtst", "chronos2"])
    p.add_argument("--bridge_type", type=str, default="prefix", choices=["prefix", "xattn"])
    p.add_argument("--chronos_model_path", type=str, default="amazon/chronos-2")

    p.add_argument("--ts_num_vars", type=int, default=1)
    p.add_argument("--ts_patch_len", type=int, default=16)
    p.add_argument("--ts_d_model", type=int, default=256)
    p.add_argument("--ts_layers", type=int, default=4)
    p.add_argument("--ts_heads", type=int, default=8)

    p.add_argument("--prefix_tokens", type=int, default=32)
    p.add_argument("--resampler_layers", type=int, default=2)
    p.add_argument("--resampler_heads", type=int, default=8)

    p.add_argument("--n_stat_tokens", type=int, default=2)
    p.add_argument("--use_revin", action="store_true")
    p.add_argument("--prefix_alpha_init", type=float, default=0.1)

    # prompting
    p.add_argument("--default_prompt", type=str, default="You are a time-series captioning assistant. Describe the given time series.")

    # training
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--num_train_epochs", type=float, default=3.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--per_device_eval_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--max_text_length", type=int, default=512)
    p.add_argument("--max_prompt_length", type=int, default=256)
    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    # precision
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")

    # ablations / finetune control
    p.add_argument("--freeze_llm", action="store_true")

    # LoRA (optional)
    p.add_argument("--lora_r", type=int, default=0, help="0 disables LoRA; >0 enables LoRA on the LLM")
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated. Adjust for your LLM architecture if needed.",
    )

    return p.parse_args()


def maybe_apply_lora(model: TSReportLM, args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    """Apply LoRA to model.llm and return the LoRA config dict (to save with checkpoint)."""
    if args.lora_r <= 0:
        return None
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except Exception as e:
        raise ImportError("peft is required for LoRA. Install: pip install peft") from e

    target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model.llm = get_peft_model(model.llm, lora_cfg)
    model.llm.print_trainable_parameters()

    # serialize minimal config
    return {
        "r": args.lora_r,
        "alpha": args.lora_alpha,
        "dropout": args.lora_dropout,
        "target_modules": target_modules,
    }


def main():
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # config stored with the model
    cfg = TSRLMConfig(
        llm_name_or_path=args.llm_name_or_path,
        encoder_type=args.encoder_type,
        bridge_type=args.bridge_type,
        chronos_model_path=args.chronos_model_path,
        ts_num_vars=args.ts_num_vars,
        ts_patch_len=args.ts_patch_len,
        ts_d_model=args.ts_d_model,
        ts_layers=args.ts_layers,
        ts_heads=args.ts_heads,
        prefix_tokens=args.prefix_tokens,
        resampler_layers=args.resampler_layers,
        resampler_heads=args.resampler_heads,
        n_stat_tokens=args.n_stat_tokens,
        use_revin=args.use_revin,
        prefix_alpha_init=args.prefix_alpha_init,
        default_prompt=args.default_prompt,
        max_text_length=args.max_text_length,
    )

    # dtype selection
    torch_dtype: Optional[torch.dtype] = None
    if args.bf16:
        torch_dtype = torch.bfloat16
    elif args.fp16:
        torch_dtype = torch.float16

    model = TSReportLM(
        config=cfg,
        freeze_llm=args.freeze_llm,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch_dtype,
    )

    lora_cfg = maybe_apply_lora(model, args)

    # grad ckpt (LLM side)
    if args.gradient_checkpointing:
        try:
            model.llm.gradient_checkpointing_enable()
        except Exception:
            pass
        model.llm.config.use_cache = False

    train_ds = TSSFTDataset(args.train_jsonl)
    eval_ds = TSSFTDataset(args.eval_jsonl) if args.eval_jsonl else None

    collator = TSSFTCollator(
        tokenizer=model.tokenizer,
        max_text_length=args.max_text_length,
        max_prompt_length=args.max_prompt_length,
        pad_to_multiple_of=8,
        add_eos=True,
    )

    training_args = TrainingArguments(
        output_dir=str(outdir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=args.eval_steps if eval_ds is not None else None,
        save_total_limit=args.save_total_limit,
        fp16=args.fp16,
        bf16=args.bf16,
        seed=args.seed,
        max_grad_norm=args.max_grad_norm,
        remove_unused_columns=False,
        report_to=[],
        dataloader_pin_memory=True,
        save_safetensors=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    trainer.train()

    # ---- final save ----
    final_dir = outdir / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), final_dir / "pytorch_model.bin")
    cfg.save(final_dir / "tsrlm_config.json")
    if lora_cfg is not None:
        import json
        with (final_dir / "lora_config.json").open("w", encoding="utf-8") as f:
            json.dump(lora_cfg, f, ensure_ascii=False, indent=2)

    model.tokenizer.save_pretrained(final_dir)

    print(f"Saved final model to: {final_dir}")


if __name__ == "__main__":
    main()
