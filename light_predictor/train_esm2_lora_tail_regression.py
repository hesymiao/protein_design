import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


TAIL_FRACTIONS = (0.10, 0.05, 0.02, 0.01)


class LabelTransform:
    def __init__(self, method: str) -> None:
        self.method = method
        self.mean = 0.0
        self.std = 1.0

    def fit(self, values: np.ndarray) -> None:
        if self.method == "zscore":
            self.mean = float(values.mean())
            self.std = float(values.std())
            if self.std == 0.0:
                self.std = 1.0

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.method == "zscore":
            return (values - self.mean) / self.std
        return values

    def inverse(self, values: np.ndarray) -> np.ndarray:
        if self.method == "zscore":
            return values * self.std + self.mean
        return values

    def scalar_to_transformed(self, value: float) -> float:
        if self.method == "zscore":
            return (value - self.mean) / self.std
        return value

    def state_dict(self) -> Dict[str, float | str]:
        return {"method": self.method, "mean": self.mean, "std": self.std}


class TailAwareRegressionTrainer(Trainer):
    def __init__(
        self,
        *args,
        sample_weight_key: str,
        rank_weight: float,
        rank_tail_threshold_raw: float,
        rank_min_gap_raw: float,
        regression_loss: str,
        huber_delta: float,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.sample_weight_key = sample_weight_key
        self.rank_weight = rank_weight
        self.rank_tail_threshold_raw = rank_tail_threshold_raw
        self.rank_min_gap_raw = rank_min_gap_raw
        self.regression_loss = regression_loss
        self.huber_delta = huber_delta

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        raw_scores = inputs.pop("raw_score")
        all_sample_weights = {
            "sample_weight_stage1": inputs.pop("sample_weight_stage1"),
            "sample_weight_stage2": inputs.pop("sample_weight_stage2"),
        }
        sample_weights = all_sample_weights[self.sample_weight_key]
        outputs = model(**inputs)
        predictions = outputs.logits.reshape(-1)

        labels = labels.reshape(-1).to(predictions.dtype)
        raw_scores = raw_scores.reshape(-1).to(predictions.dtype)
        sample_weights = sample_weights.reshape(-1).to(predictions.dtype)

        if self.regression_loss == "huber":
            regression_losses = F.huber_loss(
                predictions,
                labels,
                reduction="none",
                delta=self.huber_delta,
            )
        elif self.regression_loss == "mse":
            regression_losses = F.mse_loss(predictions, labels, reduction="none")
        else:
            raise ValueError(f"Unknown regression loss: {self.regression_loss}")

        regression_loss = (regression_losses * sample_weights).sum() / sample_weights.sum().clamp_min(1e-8)
        loss = regression_loss
        rank_loss = predictions.new_tensor(0.0)

        if self.rank_weight > 0.0 and predictions.numel() >= 2:
            rank_loss = self._pairwise_rank_loss(
                predictions=predictions,
                raw_scores=raw_scores,
                sample_weights=sample_weights,
            )
            loss = loss + self.rank_weight * rank_loss

        if return_outputs:
            outputs.regression_loss = regression_loss.detach()
            outputs.rank_loss = rank_loss.detach()
            return loss, outputs
        return loss

    def _pairwise_rank_loss(
        self,
        predictions: torch.Tensor,
        raw_scores: torch.Tensor,
        sample_weights: torch.Tensor,
    ) -> torch.Tensor:
        score_diff = raw_scores.unsqueeze(1) - raw_scores.unsqueeze(0)
        pred_diff = predictions.unsqueeze(1) - predictions.unsqueeze(0)
        sign = torch.sign(score_diff)

        valid = torch.triu(torch.ones_like(score_diff, dtype=torch.bool), diagonal=1)
        valid &= sign != 0
        valid &= score_diff.abs() >= self.rank_min_gap_raw

        tail_mask = raw_scores >= self.rank_tail_threshold_raw
        valid &= tail_mask.unsqueeze(1) | tail_mask.unsqueeze(0)

        if not torch.any(valid):
            return predictions.new_tensor(0.0)

        pair_weights = torch.maximum(sample_weights.unsqueeze(1), sample_weights.unsqueeze(0))
        pair_losses = F.softplus(-(sign * pred_diff))
        return (pair_losses[valid] * pair_weights[valid]).sum() / pair_weights[valid].sum().clamp_min(1e-8)


def to_serializable(value):
    if isinstance(value, dict):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def select_key_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    keys = (
        "pearson",
        "spearman",
        "rmse",
        "mae",
        "r2",
        "top10_pearson",
        "top5_pearson",
        "top2_pearson",
        "top1_pearson",
        "top10_mae",
        "top5_mae",
        "top2_mae",
        "top1_mae",
    )
    return {key: float(metrics[key]) for key in keys if key in metrics}


def add_stage_weight_args(
    parser: argparse.ArgumentParser,
    prefix: str,
    top10_default: float,
    top5_default: float,
    top2_default: float,
    top1_default: float,
) -> None:
    parser.add_argument(f"--{prefix}-top10-weight", type=float, default=top10_default)
    parser.add_argument(f"--{prefix}-top5-weight", type=float, default=top5_default)
    parser.add_argument(f"--{prefix}-top2-weight", type=float, default=top2_default)
    parser.add_argument(f"--{prefix}-top1-weight", type=float, default=top1_default)


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
    parser.add_argument("--max-eval-samples", type=int, default=None)
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
    parser.add_argument("--stage1-resume-from-checkpoint", type=Path, default=None)
    parser.add_argument("--stage2-resume-from-checkpoint", type=Path, default=None)
    parser.add_argument("--greater-is-better", action="store_true", default=True)
    add_stage_weight_args(parser, "stage1", 1.5, 2.0, 3.0, 4.0)
    add_stage_weight_args(parser, "stage2", 2.0, 4.0, 8.0, 12.0)
    return parser.parse_args()


def compute_regression_metrics(predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    predictions = predictions.astype(np.float64)
    labels = labels.astype(np.float64)
    spearman = spearmanr(labels, predictions).statistic
    pearson = pearsonr(labels, predictions).statistic if len(labels) > 1 else math.nan
    rmse = math.sqrt(mean_squared_error(labels, predictions))
    mse = mean_squared_error(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    r2 = r2_score(labels, predictions)
    return {
        "spearman": float(0.0 if np.isnan(spearman) else spearman),
        "pearson": float(0.0 if np.isnan(pearson) else pearson),
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
    }


def compute_tail_metrics(predictions: np.ndarray, labels: np.ndarray, fractions: Iterable[float]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for fraction in fractions:
        pct = int(round(fraction * 100))
        threshold = float(np.quantile(labels, 1.0 - fraction))
        mask = labels >= threshold
        sub_predictions = predictions[mask]
        sub_labels = labels[mask]
        metrics[f"top{pct}_n"] = float(mask.sum())
        metrics[f"top{pct}_threshold"] = threshold
        if mask.sum() < 2:
            metrics[f"top{pct}_pearson"] = 0.0
            metrics[f"top{pct}_spearman"] = 0.0
            metrics[f"top{pct}_mae"] = 0.0
            continue
        tail_stats = compute_regression_metrics(sub_predictions, sub_labels)
        metrics[f"top{pct}_pearson"] = tail_stats["pearson"]
        metrics[f"top{pct}_spearman"] = tail_stats["spearman"]
        metrics[f"top{pct}_mae"] = tail_stats["mae"]
    return metrics


def dataframe_to_dataset(frame: pd.DataFrame) -> Dataset:
    payload = {
        "sequence": frame["sequence"].tolist(),
        "labels": frame["labels"].astype(np.float32).tolist(),
        "raw_score": frame["score"].astype(np.float32).tolist(),
        "sample_weight_stage1": frame["sample_weight_stage1"].astype(np.float32).tolist(),
        "sample_weight_stage2": frame["sample_weight_stage2"].astype(np.float32).tolist(),
    }
    return Dataset.from_dict(payload)


def extract_primary_label_array(label_ids):
    if isinstance(label_ids, tuple):
        return label_ids[0]
    return label_ids


def limit_rows(frame: pd.DataFrame, limit: int | None, seed: int) -> pd.DataFrame:
    if limit is None or len(frame) <= limit:
        return frame.reset_index(drop=True)
    return frame.sample(n=limit, random_state=seed).reset_index(drop=True)


def build_sample_weights(frame: pd.DataFrame, thresholds: Dict[str, float], prefix: str, args: argparse.Namespace) -> None:
    weights = np.ones(len(frame), dtype=np.float32)
    score_array = frame["score"].to_numpy(dtype=np.float32)
    weights = np.where(score_array >= thresholds["top10"], getattr(args, f"{prefix}_top10_weight"), weights)
    weights = np.where(score_array >= thresholds["top5"], getattr(args, f"{prefix}_top5_weight"), weights)
    weights = np.where(score_array >= thresholds["top2"], getattr(args, f"{prefix}_top2_weight"), weights)
    weights = np.where(score_array >= thresholds["top1"], getattr(args, f"{prefix}_top1_weight"), weights)
    frame[f"sample_weight_{prefix}"] = weights


def save_predictions(
    trainer: Trainer,
    tokenized_dataset: Dataset,
    raw_frame: pd.DataFrame,
    split_name: str,
    label_transform: LabelTransform,
    output_dir: Path,
) -> Dict[str, float]:
    prediction_output = trainer.predict(tokenized_dataset, metric_key_prefix=split_name)
    pred_values = prediction_output.predictions.reshape(-1)
    pred_values = label_transform.inverse(pred_values)
    true_values = label_transform.inverse(extract_primary_label_array(prediction_output.label_ids).reshape(-1))

    metrics = compute_regression_metrics(pred_values, true_values)
    metrics.update(compute_tail_metrics(pred_values, true_values, TAIL_FRACTIONS))

    prediction_frame = raw_frame.copy().reset_index(drop=True)
    prediction_frame["prediction"] = pred_values
    prediction_frame["target"] = true_values
    prediction_frame.to_csv(output_dir / f"{split_name}_predictions.csv", index=False)

    with (output_dir / f"{split_name}_metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2)

    return metrics


def build_training_args(
    output_dir: Path,
    learning_rate: float,
    warmup_ratio: float,
    epochs: float,
    args: argparse.Namespace,
    metric_for_best_model: str,
) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        learning_rate=learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=warmup_ratio,
        num_train_epochs=epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        bf16=args.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=args.fp16 and torch.cuda.is_available() and not (args.bf16 and torch.cuda.is_bf16_supported()),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=args.greater_is_better,
        logging_strategy="steps",
        logging_steps=20,
        save_total_limit=2,
        report_to=[],
        remove_unused_columns=True,
        label_names=["labels", "raw_score", "sample_weight_stage1", "sample_weight_stage2"],
        seed=args.seed,
        data_seed=args.seed,
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage1_dir = args.output_dir / "stage1"
    stage2_dir = args.output_dir / "stage2_tail_refine"
    set_seed(args.seed)

    df = pd.read_csv(args.input_csv)
    required_columns = {"sequence", "score", "split"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    train_frame = limit_rows(df[df["split"] == "train"].copy(), args.max_train_samples, args.seed)
    val_frame = limit_rows(df[df["split"] == "val"].copy(), args.max_eval_samples, args.seed)
    test_frame = limit_rows(df[df["split"] == "test"].copy(), args.max_eval_samples, args.seed)

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

    for frame in (train_frame, val_frame, test_frame):
        frame["labels"] = label_transform.transform(frame["score"].to_numpy(dtype=np.float32))
        build_sample_weights(frame, quantiles, "stage1", args)
        build_sample_weights(frame, quantiles, "stage2", args)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize(batch: Dict[str, list[str]]) -> Dict[str, list[int]]:
        return tokenizer(
            batch["sequence"],
            truncation=True,
            max_length=args.max_length,
        )

    train_dataset = dataframe_to_dataset(train_frame)
    val_dataset = dataframe_to_dataset(val_frame)
    test_dataset = dataframe_to_dataset(test_frame)

    train_dataset = train_dataset.map(tokenize, batched=True, desc="Tokenizing train")
    val_dataset = val_dataset.map(tokenize, batched=True, desc="Tokenizing val")
    test_dataset = test_dataset.map(tokenize, batched=True, desc="Tokenizing test")

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

    stage1_result = stage1_trainer.train(
        resume_from_checkpoint=str(args.stage1_resume_from_checkpoint)
        if args.stage1_resume_from_checkpoint is not None
        else None
    )
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
        stage2_result = stage2_trainer.train(
            resume_from_checkpoint=str(args.stage2_resume_from_checkpoint)
            if args.stage2_resume_from_checkpoint is not None
            else None
        )
        stage2_metrics = to_serializable(stage2_result.metrics)
        final_trainer = stage2_trainer

    final_trainer.save_model(args.output_dir)

    with (args.output_dir / "label_transform.json").open("w") as handle:
        json.dump(label_transform.state_dict(), handle, indent=2)
    with (args.output_dir / "train_args.json").open("w") as handle:
        json.dump(vars(args), handle, indent=2, default=str)
    with (args.output_dir / "stage1_train_metrics.json").open("w") as handle:
        json.dump(stage1_metrics, handle, indent=2)
    if stage2_metrics is not None:
        with (args.output_dir / "stage2_train_metrics.json").open("w") as handle:
            json.dump(stage2_metrics, handle, indent=2)

    val_metrics = save_predictions(
        trainer=final_trainer,
        tokenized_dataset=val_dataset,
        raw_frame=val_frame,
        split_name="val",
        label_transform=label_transform,
        output_dir=args.output_dir,
    )
    test_metrics = save_predictions(
        trainer=final_trainer,
        tokenized_dataset=test_dataset,
        raw_frame=test_frame,
        split_name="test",
        label_transform=label_transform,
        output_dir=args.output_dir,
    )

    summary = {
        "model_name": args.model_name,
        "train_rows": len(train_frame),
        "val_rows": len(val_frame),
        "test_rows": len(test_frame),
        "use_bf16": use_bf16,
        "use_fp16": use_fp16,
        "label_transform": label_transform.state_dict(),
        "tail_quantiles": quantiles,
        "stage1_metrics": stage1_metrics,
        "stage2_metrics": stage2_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    final_metrics = {
        "stage1_metric_for_best_model": args.stage1_metric_for_best_model,
        "stage2_metric_for_best_model": args.stage2_metric_for_best_model,
        "val": select_key_metrics(val_metrics),
        "test": select_key_metrics(test_metrics),
    }
    with (args.output_dir / "final_metrics.json").open("w") as handle:
        json.dump(final_metrics, handle, indent=2)

    print(json.dumps(to_serializable(summary), indent=2))
    print("\nFinal metrics:")
    print(json.dumps(to_serializable(final_metrics), indent=2))


if __name__ == "__main__":
    main()
