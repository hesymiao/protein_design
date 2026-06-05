import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, set_seed

from train_esm2_lora_tail_regression import (
    LabelTransform,
    TailAwareRegressionTrainer,
    TAIL_FRACTIONS,
    add_stage_weight_args,
    build_sample_weights,
    build_training_args,
    compute_regression_metrics,
    compute_tail_metrics,
    dataframe_to_dataset,
    extract_primary_label_array,
    limit_rows,
    save_predictions,
    select_key_metrics,
    to_serializable,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--model-name",
        default="/bigdat2/user/hesy/protein_design/last_year/ESM_finetune/facebook_esm2_t33_650M_UR50D",
    )
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--num-train-epochs", type=float, default=6.0)
    parser.add_argument("--tail-refine-epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--tail-refine-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--tail-refine-warmup-ratio", type=float, default=0.05)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--lora-r", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=1)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--label-transform", choices=["none", "zscore"], default="zscore")
    parser.add_argument("--regression-loss", choices=["huber", "mse"], default="huber")
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--stage1-rank-weight", type=float, default=0.0)
    parser.add_argument("--stage2-rank-weight", type=float, default=0.5)
    parser.add_argument("--rank-tail-quantile", type=float, default=0.90)
    parser.add_argument("--rank-min-gap", type=float, default=0.02)
    parser.add_argument("--stage1-metric-for-best-model", type=str, default="pearson")
    parser.add_argument("--stage2-metric-for-best-model", type=str, default="top2_pearson")
    parser.add_argument("--greater-is-better", action="store_true", default=True)
    add_stage_weight_args(parser, "stage1", 1.5, 2.0, 3.0, 4.0)
    add_stage_weight_args(parser, "stage2", 2.0, 4.0, 8.0, 12.0)
    return parser.parse_args()


def _clean_sequence(sequence: str) -> str:
    return sequence.split("#", 1)[0].strip()


def _clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["sequence"] = frame["sequence"].astype(str).map(_clean_sequence)
    if "sequence_length" in frame.columns:
        frame["sequence_length"] = frame["sequence"].str.len().astype(int)
    return frame


def build_fixed_folds(frame: pd.DataFrame, num_folds: int, seed: int) -> pd.DataFrame:
    frame = frame.reset_index(drop=False).rename(columns={"index": "source_index"})
    frame["original_split"] = frame["split"].astype(str) if "split" in frame.columns else "all"

    if len(frame) < num_folds:
        raise ValueError(f"rows={len(frame)} is smaller than num_folds={num_folds}")

    unique_scores = int(frame["score"].nunique())
    if unique_scores <= 1:
        score_bins = np.zeros(len(frame), dtype=np.int64)
    else:
        num_bins = min(max(num_folds, 10), unique_scores)
        score_bins = pd.qcut(frame["score"], q=num_bins, labels=False, duplicates="drop").to_numpy()
        score_bins = np.nan_to_num(score_bins, nan=0).astype(np.int64)

    rng = np.random.default_rng(seed)
    fold_ids = np.empty(len(frame), dtype=np.int64)
    for bin_id in np.unique(score_bins):
        indices = np.flatnonzero(score_bins == bin_id)
        shuffled = rng.permutation(indices)
        fold_ids[shuffled] = np.arange(len(shuffled), dtype=np.int64) % num_folds

    frame["fold"] = fold_ids
    return frame


def train_single_fold(
    args: argparse.Namespace,
    tokenizer: AutoTokenizer,
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    fold_index: int,
) -> Dict[str, object]:
    fold_dir = args.output_dir / f"fold_{fold_index}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    stage1_dir = fold_dir / "stage1"
    stage2_dir = fold_dir / "stage2_tail_refine"

    fold_seed = args.seed + fold_index
    set_seed(fold_seed)

    train_frame = _clean_frame(train_frame)
    val_frame = _clean_frame(val_frame)

    label_transform = LabelTransform(args.label_transform)
    train_scores = train_frame["score"].to_numpy(dtype=np.float32)
    label_transform.fit(train_scores)

    quantiles = {
        "top10": float(train_frame["score"].quantile(0.90)),
        "top5": float(train_frame["score"].quantile(0.95)),
        "top2": float(train_frame["score"].quantile(0.98)),
        "top1": float(train_frame["score"].quantile(0.99)),
        "rank_tail": float(train_frame["score"].quantile(args.rank_tail_quantile)),
    }

    for frame in (train_frame, val_frame):
        frame["labels"] = label_transform.transform(frame["score"].to_numpy(dtype=np.float32))
        build_sample_weights(frame, quantiles, "stage1", args)
        build_sample_weights(frame, quantiles, "stage2", args)

    def tokenize(batch: Dict[str, list[str]]) -> Dict[str, list[int]]:
        return tokenizer(
            batch["sequence"],
            truncation=True,
            max_length=args.max_length,
        )

    train_dataset = dataframe_to_dataset(train_frame)
    val_dataset = dataframe_to_dataset(val_frame)

    train_dataset = train_dataset.map(tokenize, batched=True, desc=f"Tokenizing train fold {fold_index}")
    val_dataset = val_dataset.map(tokenize, batched=True, desc=f"Tokenizing val fold {fold_index}")

    use_bf16 = args.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = args.fp16 and torch.cuda.is_available() and not use_bf16
    dtype = torch.bfloat16 if use_bf16 else torch.float16 if use_fp16 else torch.float32

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=1,
        ignore_mismatched_sizes=True,
        torch_dtype=dtype,
    )
    model.config.problem_type = "regression"

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="all",
        target_modules=[
            "attention.self.query",
            "attention.self.key",
            "attention.self.value",
            "attention.output.dense",
            "intermediate.dense",
            "output.dense",
        ],
        modules_to_save=["classifier"],
    )
    model = get_peft_model(model, lora_config)
    model.classifier.requires_grad_(True)
    model.print_trainable_parameters()

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred) -> Dict[str, float]:
        logits, labels = eval_pred
        predictions = logits.reshape(-1)
        labels = extract_primary_label_array(labels).reshape(-1)
        predictions = label_transform.inverse(predictions)
        labels = label_transform.inverse(labels)
        metrics = compute_regression_metrics(predictions, labels)
        metrics.update(compute_tail_metrics(predictions, labels, TAIL_FRACTIONS))
        return metrics

    trainer_kwargs = dict(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    stage1_trainer = TailAwareRegressionTrainer(
        args=build_training_args(
            output_dir=stage1_dir,
            learning_rate=args.learning_rate,
            warmup_ratio=args.warmup_ratio,
            epochs=args.num_train_epochs,
            args=args,
            metric_for_best_model=args.stage1_metric_for_best_model,
        ),
        sample_weight_key="sample_weight_stage1",
        rank_weight=args.stage1_rank_weight,
        rank_tail_threshold_raw=quantiles["rank_tail"],
        rank_min_gap_raw=args.rank_min_gap,
        regression_loss=args.regression_loss,
        huber_delta=args.huber_delta,
        **trainer_kwargs,
    )

    stage1_result = stage1_trainer.train()
    stage1_metrics = to_serializable(stage1_result.metrics)

    final_trainer = stage1_trainer
    stage2_metrics = None

    if args.tail_refine_epochs > 0:
        stage2_trainer = TailAwareRegressionTrainer(
            model=stage1_trainer.model,
            args=build_training_args(
                output_dir=stage2_dir,
                learning_rate=args.tail_refine_learning_rate,
                warmup_ratio=args.tail_refine_warmup_ratio,
                epochs=args.tail_refine_epochs,
                args=args,
                metric_for_best_model=args.stage2_metric_for_best_model,
            ),
            sample_weight_key="sample_weight_stage2",
            rank_weight=args.stage2_rank_weight,
            rank_tail_threshold_raw=quantiles["rank_tail"],
            rank_min_gap_raw=args.rank_min_gap,
            regression_loss=args.regression_loss,
            huber_delta=args.huber_delta,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        stage2_result = stage2_trainer.train()
        stage2_metrics = to_serializable(stage2_result.metrics)
        final_trainer = stage2_trainer

    final_trainer.save_model(fold_dir)

    with (fold_dir / "label_transform.json").open("w") as handle:
        json.dump(label_transform.state_dict(), handle, indent=2)
    with (fold_dir / "train_args.json").open("w") as handle:
        json.dump(vars(args), handle, indent=2, default=str)
    with (fold_dir / "stage1_train_metrics.json").open("w") as handle:
        json.dump(stage1_metrics, handle, indent=2)
    if stage2_metrics is not None:
        with (fold_dir / "stage2_train_metrics.json").open("w") as handle:
            json.dump(stage2_metrics, handle, indent=2)

    val_metrics = save_predictions(
        trainer=final_trainer,
        tokenized_dataset=val_dataset,
        raw_frame=val_frame,
        split_name="val",
        label_transform=label_transform,
        output_dir=fold_dir,
    )

    summary = {
        "fold": fold_index,
        "seed": fold_seed,
        "train_rows": len(train_frame),
        "val_rows": len(val_frame),
        "use_bf16": use_bf16,
        "use_fp16": use_fp16,
        "label_transform": label_transform.state_dict(),
        "tail_quantiles": quantiles,
        "stage1_metrics": stage1_metrics,
        "stage2_metrics": stage2_metrics,
        "val_metrics": val_metrics,
    }
    with (fold_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    final_metrics = {
        "fold": fold_index,
        "stage1_metric_for_best_model": args.stage1_metric_for_best_model,
        "stage2_metric_for_best_model": args.stage2_metric_for_best_model,
        "val": select_key_metrics(val_metrics),
    }
    with (fold_dir / "final_metrics.json").open("w") as handle:
        json.dump(final_metrics, handle, indent=2)

    val_prediction_frame = pd.read_csv(fold_dir / "val_predictions.csv")
    val_prediction_frame["fold"] = fold_index

    del final_trainer
    del stage1_trainer
    del model
    torch.cuda.empty_cache()

    print(json.dumps(to_serializable(summary), indent=2))
    print("\nFold metrics:")
    print(json.dumps(to_serializable(final_metrics), indent=2))

    return {
        "summary": summary,
        "final_metrics": final_metrics,
        "val_prediction_frame": val_prediction_frame,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    required_columns = {"sequence", "score"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = limit_rows(df.copy(), args.max_train_samples, args.seed)
    fold_frame = build_fixed_folds(df, args.num_folds, args.seed)
    fold_frame.to_csv(args.output_dir / "fold_assignments.csv", index=False)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    all_fold_metrics = []
    oof_frames = []

    for fold_index in range(args.num_folds):
        train_frame = fold_frame[fold_frame["fold"] != fold_index].copy().reset_index(drop=True)
        val_frame = fold_frame[fold_frame["fold"] == fold_index].copy().reset_index(drop=True)
        result = train_single_fold(
            args=args,
            tokenizer=tokenizer,
            train_frame=train_frame,
            val_frame=val_frame,
            fold_index=fold_index,
        )
        all_fold_metrics.append(result["final_metrics"]["val"])
        oof_frames.append(result["val_prediction_frame"])

    oof_frame = pd.concat(oof_frames, axis=0, ignore_index=True)
    oof_frame = oof_frame.sort_values(["source_index", "fold"]).reset_index(drop=True)
    oof_frame.to_csv(args.output_dir / "oof_predictions.csv", index=False)

    oof_predictions = oof_frame["prediction"].to_numpy(dtype=np.float64)
    oof_targets = oof_frame["target"].to_numpy(dtype=np.float64)
    oof_metrics = compute_regression_metrics(oof_predictions, oof_targets)
    oof_metrics.update(compute_tail_metrics(oof_predictions, oof_targets, TAIL_FRACTIONS))
    with (args.output_dir / "oof_metrics.json").open("w") as handle:
        json.dump(oof_metrics, handle, indent=2)

    fold_metrics_frame = pd.DataFrame(all_fold_metrics)
    fold_metrics_frame.insert(0, "fold", np.arange(args.num_folds, dtype=np.int64))
    fold_metrics_frame.to_csv(args.output_dir / "fold_metrics.csv", index=False)

    summary = {
        "input_csv": str(args.input_csv),
        "model_name": args.model_name,
        "num_folds": args.num_folds,
        "total_rows": int(len(fold_frame)),
        "oof_metrics": oof_metrics,
    }
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    final_metrics = {
        "num_folds": args.num_folds,
        "oof": select_key_metrics(oof_metrics),
    }
    with (args.output_dir / "final_metrics.json").open("w") as handle:
        json.dump(final_metrics, handle, indent=2)

    print("\nOOF metrics:")
    print(json.dumps(to_serializable(final_metrics), indent=2))


if __name__ == "__main__":
    main()
