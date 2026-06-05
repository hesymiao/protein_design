import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import generate as base_generate


def resolve_fold_adapter_dirs(adapter_root: Path) -> List[Path]:
    fold_dirs = sorted(
        path
        for path in adapter_root.glob("fold_*")
        if path.is_dir() and (path / "adapter_config.json").exists()
    )
    if not fold_dirs:
        raise ValueError(f"No fold adapter directories found under {adapter_root}")
    if len(fold_dirs) != 5:
        raise ValueError(f"Expected 5 completed fold adapters under {adapter_root}, found {len(fold_dirs)}")
    return fold_dirs


def load_label_transform(adapter_dir: Path) -> Tuple[str, float, float]:
    with (adapter_dir / "label_transform.json").open() as handle:
        state = json.load(handle)
    return state["method"], float(state["mean"]), float(state["std"])


def inverse_transform(values: torch.Tensor, method: str, mean: float, std: float) -> torch.Tensor:
    if method == "zscore":
        return values * std + mean
    return values


def score_sequences_with_details(sequences: List[str], args) -> List[Dict[str, object]]:
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    adapter_dirs = resolve_fold_adapter_dirs(args.lora_adapter)

    tokenizer = AutoTokenizer.from_pretrained(args.hf_base_model)
    ensemble_predictions: List[torch.Tensor] = []

    for adapter_dir in adapter_dirs:
        method, mean, std = load_label_transform(adapter_dir)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            args.hf_base_model,
            num_labels=1,
            torch_dtype=dtype,
        )
        model = PeftModel.from_pretrained(base_model, adapter_dir)
        model = model.to(device)
        model.eval()

        fold_scores: List[torch.Tensor] = []
        for start in range(0, len(sequences), args.score_batch_size):
            batch = sequences[start : start + args.score_batch_size]
            encoded = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                logits = model(**encoded).logits.reshape(-1).float().cpu()
            preds = inverse_transform(logits, method, mean, std)
            fold_scores.append(preds)

        ensemble_predictions.append(torch.cat(fold_scores, dim=0))

        del model
        del base_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    stacked_predictions = torch.stack(ensemble_predictions, dim=0)
    mean_predictions = stacked_predictions.mean(dim=0)
    std_predictions = stacked_predictions.std(dim=0, unbiased=False)

    records: List[Dict[str, object]] = []
    for seq_idx, sequence in enumerate(sequences):
        fold_scores = [float(x) for x in stacked_predictions[:, seq_idx].tolist()]
        records.append(
            {
                "sequence": sequence,
                "predicted_brightness": float(mean_predictions[seq_idx].item()),
                "prediction_std": float(std_predictions[seq_idx].item()),
                "fold_scores": fold_scores,
            }
        )
    return records


def score_unique_sequences_with_details(
    sequences: List[str],
    args,
    cache: Dict[str, Dict[str, object]],
) -> None:
    missing = [seq for seq in sequences if seq not in cache]
    if not missing:
        return
    records = score_sequences_with_details(missing, args)
    for record in records:
        cache[str(record["sequence"])] = record


def main() -> None:
    args = base_generate.parse_args()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    if args.summary_json is None:
        args.summary_json = args.output_csv.with_suffix(".summary.json")

    base_generate.random.seed(args.seed)
    rng = base_generate.random.Random(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    frame = pd.read_csv(args.avgfp_csv).copy()
    frame["sequence"] = frame["sequence"].astype(str).map(base_generate.clean_sequence)
    parent = base_generate.load_parent_sequence(args.parent_seq_file, args.parent_header)

    steering_vector_path, steering_info = base_generate.ensure_steering_vector(args, frame)

    top_hotspot_frame, residue_counts = base_generate.build_hotspot_statistics(frame, parent, args.top_k_hotspots)
    top1_sequence = base_generate.clean_sequence(top_hotspot_frame.iloc[0]["sequence"])
    top1_actual_score = float(top_hotspot_frame.iloc[0]["score"])
    seed_sequence, preserved_mutations, top1_mutations = base_generate.build_seed_from_top1(
        parent=parent,
        top1_sequence=top1_sequence,
        preserve_top1_count=args.preserve_top1_count,
    )
    preserved_positions = {idx for idx, _src, _dst in preserved_mutations}
    unpreserved_top1_positions = {idx for idx, _src, _dst in top1_mutations} - preserved_positions

    hotspot_positions: List[int] = []
    hotspot_weights: List[float] = []
    for idx, counts in sorted(residue_counts.items()):
        if idx in preserved_positions:
            continue
        if sum(counts.values()) >= args.min_hotspot_count or idx in unpreserved_top1_positions:
            hotspot_positions.append(idx)
            hotspot_weights.append(float(sum(counts.values())))
    if not hotspot_positions:
        raise ValueError("No hotspot positions selected; lower --min-hotspot-count or preserve fewer top1 mutations.")

    dataset_score_map = frame.groupby("sequence")["score"].max().to_dict()
    score_cache: Dict[str, Dict[str, object]] = {}
    baseline_sequences = [parent, seed_sequence, top1_sequence]
    score_unique_sequences_with_details(baseline_sequences, args, score_cache)

    beam = [seed_sequence]
    all_records: List[Dict[str, object]] = []
    seen_sequences = set()

    def add_record(
        sequence: str,
        source: str,
        round_idx: int,
        parent_sequence: str | None,
        chosen_positions: List[int] | None,
        site_candidates: Dict[int, List[str]] | None,
    ) -> None:
        if sequence in seen_sequences:
            return
        seen_sequences.add(sequence)
        seq_mutations = base_generate.mutation_list(parent, sequence)
        seq_mutation_positions = {idx for idx, _src, _dst in seq_mutations}
        recovered = sorted(idx for idx in unpreserved_top1_positions if idx in seq_mutation_positions)
        pred_record = score_cache[sequence]
        fold_scores = [float(x) for x in pred_record["fold_scores"]]
        record = {
            "round": round_idx,
            "source": source,
            "sequence": sequence,
            "predicted_brightness": float(pred_record["predicted_brightness"]),
            "prediction_std": float(pred_record["prediction_std"]),
            "mutation_count_vs_parent": len(seq_mutations),
            "mutations_vs_parent": base_generate.format_mutations(parent, sequence),
            "chosen_positions": "" if not chosen_positions else ",".join(str(x) for x in chosen_positions),
            "site_topk": ""
            if not site_candidates
            else json.dumps({str(k): v for k, v in site_candidates.items()}, ensure_ascii=False),
            "preserved_ok": int(all(idx in seq_mutation_positions for idx in preserved_positions)),
            "recovered_unpreserved_top1_positions": ",".join(str(x) for x in recovered),
            "exact_match_top1": int(sequence == top1_sequence),
            "exact_match_dataset": int(sequence in dataset_score_map),
            "dataset_score_if_known": dataset_score_map.get(sequence),
            "parent_sequence": parent_sequence,
        }
        for fold_idx, fold_score in enumerate(fold_scores):
            record[f"predicted_brightness_fold{fold_idx}"] = fold_score
        all_records.append(record)

    add_record(parent, "baseline_parent", 0, None, None, None)
    add_record(seed_sequence, "baseline_seed_locked_top1_partial", 0, None, sorted(preserved_positions), None)
    add_record(top1_sequence, "baseline_real_top1", 0, None, sorted(idx for idx, _src, _dst in top1_mutations), None)

    for round_idx in range(1, args.rounds + 1):
        model, alphabet, steering_vectors = base_generate.load_generator(
            ckpt_path=args.ckpt_path,
            steering_vector_path=steering_vector_path,
            alpha=args.alpha,
            device=args.device,
        )
        proposals: List[Tuple[str, str, List[int], Dict[int, List[str]]]] = []
        for parent_sequence in beam:
            proposals.append((parent_sequence, parent_sequence, [], {}))
            for _branch in range(args.branch_factor):
                budget = rng.randint(args.mutation_budget_min, args.mutation_budget_max)
                positions_1based = base_generate.weighted_sample_without_replacement(
                    hotspot_positions,
                    hotspot_weights,
                    budget,
                    rng,
                )
                candidate_sequences, site_candidates = base_generate.build_candidates_from_topk(
                    sequence=parent_sequence,
                    positions_1based=positions_1based,
                    model=model,
                    alphabet=alphabet,
                    steering_vectors=steering_vectors,
                    residue_top_k=args.residue_top_k,
                    device=args.device,
                )
                for mutated in candidate_sequences:
                    proposals.append((parent_sequence, mutated, positions_1based, site_candidates))

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        unique_sequences = sorted({sequence for _src, sequence, _pos, _cand in proposals})
        score_unique_sequences_with_details(unique_sequences, args, score_cache)

        for parent_sequence, sequence, chosen_positions, site_candidates in proposals:
            add_record(
                sequence=sequence,
                source=f"round{round_idx}",
                round_idx=round_idx,
                parent_sequence=parent_sequence,
                chosen_positions=chosen_positions,
                site_candidates=site_candidates,
            )

        round_candidates = sorted(
            {sequence for _src, sequence, _pos, _cand in proposals},
            key=lambda seq: float(score_cache[seq]["predicted_brightness"]),
            reverse=True,
        )
        beam = round_candidates[: args.beam_width]

    result = pd.DataFrame(all_records).sort_values(
        ["predicted_brightness", "exact_match_top1", "exact_match_dataset"],
        ascending=[False, False, False],
    )
    result.to_csv(args.output_csv, index=False)

    summary = {
        "parent_header": args.parent_header,
        "parent_sequence": parent,
        "seed_sequence": seed_sequence,
        "preserved_mutations": [f"{src}{idx}{dst}" for idx, src, dst in preserved_mutations],
        "top1_mutations": [f"{src}{idx}{dst}" for idx, src, dst in top1_mutations],
        "top1_actual_score": top1_actual_score,
        "top1_predicted_brightness": score_cache[top1_sequence]["predicted_brightness"],
        "top1_prediction_std": score_cache[top1_sequence]["prediction_std"],
        "parent_predicted_brightness": score_cache[parent]["predicted_brightness"],
        "parent_prediction_std": score_cache[parent]["prediction_std"],
        "seed_predicted_brightness": score_cache[seed_sequence]["predicted_brightness"],
        "seed_prediction_std": score_cache[seed_sequence]["prediction_std"],
        "hotspot_positions": hotspot_positions,
        "beam_width": args.beam_width,
        "branch_factor": args.branch_factor,
        "residue_top_k": args.residue_top_k,
        "rounds": args.rounds,
        "steering": steering_info,
        "top_candidates": result.head(20).to_dict(orient="records"),
    }
    with args.summary_json.open("w") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
