import argparse
import json
import random
import types
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from module.steerable_esm2 import steering_forward
from utils.esm2_utils import (
    decode,
    extract_esm2_features,
    get_esm2_layer_and_feature_dim,
    load_esm2_model,
)


AMINO_ACID_TOKENS = list(range(4, 24))
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--avgfp-csv",
        type=Path,
        default=Path(
            "/data/user/hesy/projects/protein_design/steer-PLM/light_predictor/avgfp_esm2_lora/pplu_data/pplu_split.csv"
        ),
    )
    parser.add_argument(
        "--parent-seq-file",
        type=Path,
        default=Path("/bigdat2/user/hesy/protein_design/data/2026Protein Design/AAseqs of 5 GFP proteins.txt"),
    )
    parser.add_argument("--parent-header", type=str, default="ppluGFP")
    parser.add_argument(
        "--ckpt-path",
        type=Path,
        default=Path("/bigdat2/user/hesy/protein_design/last_year/ESM_finetune/esm2_t33_650M_UR50D.pt"),
    )
    parser.add_argument(
        "--hf-base-model",
        type=Path,
        default=Path("/bigdat2/user/hesy/protein_design/last_year/ESM_finetune/facebook_esm2_t33_650M_UR50D"),
    )
    parser.add_argument(
        "--lora-adapter",
        type=Path,
        default=Path(
            "/data/user/hesy/projects/protein_design/steer-PLM/light_predictor/avgfp_esm2_lora/runs/pplu_04_lora_tail_regression_full_new"
        ),
    )
    parser.add_argument(
        "--steering-vector",
        type=Path,
        default=None,
        help="Optional steering-vector path. If omitted, generate/reuse one automatically.",
    )
    parser.add_argument(
        "--disable-steering",
        action="store_true",
        help="Disable steering entirely and use plain ESM2 masked-token proposals.",
    )
    parser.add_argument(
        "--steering-cache-dir",
        type=Path,
        default=Path("/data/user/hesy/projects/protein_design/steer-PLM/Steering-PLMs/saved_steering_vectors"),
    )
    parser.add_argument(
        "--steering-source-split",
        type=str,
        default="all",
        choices=["all", "train", "val", "test"],
    )
    parser.add_argument("--steering-pos-quantile", type=float, default=0.90)
    parser.add_argument("--steering-neg-quantile", type=float, default=0.10)
    parser.add_argument(
        "--steering-num-data",
        type=int,
        default=512,
        help="Max positive and negative sequences used for steering-vector extraction.",
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--beam-width", type=int, default=16)
    parser.add_argument(
        "--branch-factor",
        type=int,
        default=12,
        help="How many position subsets to explore per parent sequence at each round.",
    )
    parser.add_argument("--top-k-hotspots", type=int, default=50)
    parser.add_argument("--min-hotspot-count", type=int, default=3)
    parser.add_argument("--preserve-top1-count", type=int, default=2)
    parser.add_argument("--mutation-budget-min", type=int, default=1)
    parser.add_argument("--mutation-budget-max", type=int, default=2)
    parser.add_argument(
        "--residue-top-k",
        type=int,
        default=4,
        help="For each masked site, keep the top-k steered ESM amino-acid proposals.",
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--score-batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def clean_sequence(sequence: str) -> str:
    return str(sequence).split("#", 1)[0].strip().upper()


def load_parent_sequence(seq_file: Path, header: str) -> str:
    sequence: List[str] = []
    keep = False
    header = header.lower()
    for line in seq_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(">"):
            keep = line[1:].strip().lower() == header
            continue
        if keep:
            sequence.append(line.upper())
    parent = "".join(sequence)
    if not parent:
        raise ValueError(f"Could not find header {header!r} in {seq_file}")
    invalid = sorted({char for char in parent if char not in VALID_AA})
    if invalid:
        raise ValueError(f"Parent sequence contains invalid amino acids: {invalid}")
    return parent


def mutation_list(parent: str, sequence: str) -> List[Tuple[int, str, str]]:
    return [(idx, src, dst) for idx, (src, dst) in enumerate(zip(parent, sequence), start=1) if src != dst]


def format_mutations(parent: str, sequence: str) -> str:
    return ";".join(f"{src}{idx}{dst}" for idx, src, dst in mutation_list(parent, sequence))


def build_hotspot_statistics(frame: pd.DataFrame, parent: str, top_k: int) -> Tuple[pd.DataFrame, Dict[int, Counter]]:
    top = frame.sort_values("score", ascending=False).head(top_k).reset_index(drop=True)
    residue_counts: Dict[int, Counter] = defaultdict(Counter)
    for sequence in top["sequence"]:
        for idx, _src, dst in mutation_list(parent, sequence):
            residue_counts[idx][dst] += 1
    return top, residue_counts


def build_seed_from_top1(
    parent: str,
    top1_sequence: str,
    preserve_top1_count: int,
) -> Tuple[str, List[Tuple[int, str, str]], List[Tuple[int, str, str]]]:
    top1_mutations = mutation_list(parent, top1_sequence)
    preserve = top1_mutations[:preserve_top1_count]
    seed = list(parent)
    for idx, _src, dst in preserve:
        seed[idx - 1] = dst
    return "".join(seed), preserve, top1_mutations


def weighted_sample_without_replacement(
    positions: Sequence[int],
    weights: Sequence[float],
    budget: int,
    rng: random.Random,
) -> List[int]:
    selected: List[int] = []
    remaining_positions = list(positions)
    remaining_weights = list(weights)
    budget = min(budget, len(remaining_positions))
    for _ in range(budget):
        total = sum(remaining_weights)
        if total <= 0:
            choice_idx = rng.randrange(len(remaining_positions))
        else:
            ticket = rng.random() * total
            running = 0.0
            choice_idx = 0
            for idx, weight in enumerate(remaining_weights):
                running += weight
                if running >= ticket:
                    choice_idx = idx
                    break
        selected.append(remaining_positions.pop(choice_idx))
        remaining_weights.pop(choice_idx)
    return sorted(selected)


def load_label_transform(adapter_dir: Path) -> Tuple[str, float, float]:
    with (adapter_dir / "label_transform.json").open() as handle:
        state = json.load(handle)
    return state["method"], float(state["mean"]), float(state["std"])


def inverse_transform(values: torch.Tensor, method: str, mean: float, std: float) -> torch.Tensor:
    if method == "zscore":
        return values * std + mean
    return values


def select_steering_frame(frame: pd.DataFrame, split_name: str) -> pd.DataFrame:
    if split_name == "all" or "split" not in frame.columns:
        return frame
    return frame[frame["split"] == split_name].copy()


def ensure_steering_vector(
    args: argparse.Namespace,
    frame: pd.DataFrame,
) -> Tuple[Path | None, Dict[str, object]]:
    if args.disable_steering:
        return None, {"mode": "disabled"}

    if args.steering_vector is not None and args.steering_vector.exists():
        return args.steering_vector, {"mode": "reuse_explicit", "path": str(args.steering_vector)}

    source = select_steering_frame(frame, args.steering_source_split)
    if source.empty:
        raise ValueError(f"No rows available for steering-source split {args.steering_source_split!r}")

    pos_threshold = float(source["score"].quantile(args.steering_pos_quantile))
    neg_threshold = float(source["score"].quantile(args.steering_neg_quantile))
    if pos_threshold <= neg_threshold:
        raise ValueError("Positive steering threshold must be greater than negative threshold.")

    pos_rows = source[source["score"] >= pos_threshold].copy()
    neg_rows = source[source["score"] <= neg_threshold].copy()
    pos_seqs = pos_rows["sequence"].astype(str).map(clean_sequence).tolist()
    neg_seqs = neg_rows["sequence"].astype(str).map(clean_sequence).tolist()

    if args.steering_num_data is not None:
        pos_seqs = pos_seqs[: args.steering_num_data]
        neg_seqs = neg_seqs[: args.steering_num_data]

    if not pos_seqs or not neg_seqs:
        raise ValueError("Steering-vector extraction requires non-empty positive and negative sequence sets.")

    args.steering_cache_dir.mkdir(parents=True, exist_ok=True)
    source_name = args.avgfp_csv.stem
    split_tag = args.steering_source_split
    pos_tag = int(round(args.steering_pos_quantile * 100))
    neg_tag = int(round(args.steering_neg_quantile * 100))
    auto_path = args.steering_cache_dir / f"650M_brightness_{source_name}_{split_tag}_q{pos_tag}_{neg_tag}.pt"

    if auto_path.exists():
        return auto_path, {
            "mode": "reuse_auto",
            "path": str(auto_path),
            "split": split_tag,
            "pos_threshold": pos_threshold,
            "neg_threshold": neg_threshold,
            "num_pos": len(pos_seqs),
            "num_neg": len(neg_seqs),
        }

    model, alphabet = load_esm2_model("650M", device=args.device, ckpt_path=str(args.ckpt_path))
    n_layers, _ = get_esm2_layer_and_feature_dim("650M")
    pos_repr = extract_esm2_features(pos_seqs, model, alphabet, n_layers, batch_size=1, device=args.device)
    neg_repr = extract_esm2_features(neg_seqs, model, alphabet, n_layers, batch_size=1, device=args.device)

    pos_vectors = torch.stack([pos_repr[i].mean(dim=0) for i in range(n_layers)]).detach().cpu()
    neg_vectors = torch.stack([neg_repr[i].mean(dim=0) for i in range(n_layers)]).detach().cpu()
    torch.save((pos_vectors, neg_vectors), auto_path)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return auto_path, {
        "mode": "generated_auto",
        "path": str(auto_path),
        "split": split_tag,
        "pos_threshold": pos_threshold,
        "neg_threshold": neg_threshold,
        "num_pos": len(pos_seqs),
        "num_neg": len(neg_seqs),
    }


def load_generator(
    ckpt_path: Path,
    steering_vector_path: Path | None,
    alpha: float,
    device: str,
):
    model, alphabet = load_esm2_model("650M", device=device, ckpt_path=str(ckpt_path))
    model.steering_forward = types.MethodType(steering_forward, model)
    steering_vectors = None
    if steering_vector_path is not None and steering_vector_path.exists():
        pos_vectors, neg_vectors = torch.load(steering_vector_path, map_location="cpu")
        steering_vectors = (pos_vectors - neg_vectors).to(device) * alpha
    return model, alphabet, steering_vectors


def get_site_top_residues(
    sequence: str,
    site_1based: int,
    model,
    alphabet,
    steering_vectors: torch.Tensor | None,
    top_k: int,
    device: str,
) -> List[str]:
    batch_converter = alphabet.get_batch_converter()
    _, _, batch_tokens = batch_converter([("protein", sequence)])
    batch_tokens = batch_tokens.to(device)
    original_token_id = int(batch_tokens[0, site_1based].item())
    batch_tokens[0, site_1based] = alphabet.mask_idx

    with torch.no_grad():
        if steering_vectors is not None:
            outputs = model.steering_forward(tokens=batch_tokens, steering_vectors=steering_vectors)
        else:
            outputs = model(tokens=batch_tokens)

    logits = outputs["logits"][0, site_1based, AMINO_ACID_TOKENS].float()
    logits[original_token_id - 4] = -1e8
    top_ids = torch.topk(logits, k=min(top_k, logits.shape[0]), dim=-1).indices.tolist()

    residues: List[str] = []
    for token_offset in top_ids:
        residue = alphabet.get_tok(token_offset + 4)
        if residue not in residues and residue in VALID_AA:
            residues.append(residue)
    return residues


def build_candidates_from_topk(
    sequence: str,
    positions_1based: Sequence[int],
    model,
    alphabet,
    steering_vectors: torch.Tensor | None,
    residue_top_k: int,
    device: str,
) -> Tuple[List[str], Dict[int, List[str]]]:
    per_site: Dict[int, List[str]] = {}
    for position in positions_1based:
        per_site[position] = get_site_top_residues(
            sequence=sequence,
            site_1based=position,
            model=model,
            alphabet=alphabet,
            steering_vectors=steering_vectors,
            top_k=residue_top_k,
            device=device,
        )

    if any(len(candidates) == 0 for candidates in per_site.values()):
        return [], per_site

    ordered_positions = list(positions_1based)
    candidate_sequences: List[str] = []
    for residues in product(*(per_site[position] for position in ordered_positions)):
        seq_chars = list(sequence)
        changed = False
        for position, residue in zip(ordered_positions, residues):
            if seq_chars[position - 1] != residue:
                seq_chars[position - 1] = residue
                changed = True
        if changed:
            candidate_sequences.append("".join(seq_chars))

    deduped = list(dict.fromkeys(candidate_sequences))
    return deduped, per_site


def score_sequences(sequences: List[str], args: argparse.Namespace) -> List[float]:
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    method, mean, std = load_label_transform(args.lora_adapter)

    tokenizer = AutoTokenizer.from_pretrained(args.hf_base_model)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.hf_base_model,
        num_labels=1,
        torch_dtype=dtype,
    )
    model = PeftModel.from_pretrained(base_model, args.lora_adapter)
    model = model.to(device)
    model.eval()

    scores: List[float] = []
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
        scores.extend(float(x) for x in preds.tolist())

    del model
    del base_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return scores


def score_unique_sequences(sequences: List[str], args: argparse.Namespace, cache: Dict[str, float]) -> None:
    missing = [seq for seq in sequences if seq not in cache]
    if not missing:
        return
    scores = score_sequences(missing, args)
    cache.update(dict(zip(missing, scores)))


def main() -> None:
    args = parse_args()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    if args.summary_json is None:
        args.summary_json = args.output_csv.with_suffix(".summary.json")

    random.seed(args.seed)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    frame = pd.read_csv(args.avgfp_csv).copy()
    frame["sequence"] = frame["sequence"].astype(str).map(clean_sequence)
    parent = load_parent_sequence(args.parent_seq_file, args.parent_header)

    steering_vector_path, steering_info = ensure_steering_vector(args, frame)

    top_hotspot_frame, residue_counts = build_hotspot_statistics(frame, parent, args.top_k_hotspots)
    top1_sequence = clean_sequence(top_hotspot_frame.iloc[0]["sequence"])
    top1_actual_score = float(top_hotspot_frame.iloc[0]["score"])
    seed_sequence, preserved_mutations, top1_mutations = build_seed_from_top1(
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
    score_cache: Dict[str, float] = {}
    baseline_sequences = [parent, seed_sequence, top1_sequence]
    score_unique_sequences(baseline_sequences, args, score_cache)

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
        seq_mutations = mutation_list(parent, sequence)
        seq_mutation_positions = {idx for idx, _src, _dst in seq_mutations}
        recovered = sorted(idx for idx in unpreserved_top1_positions if idx in seq_mutation_positions)
        all_records.append(
            {
                "round": round_idx,
                "source": source,
                "sequence": sequence,
                "predicted_brightness": score_cache[sequence],
                "mutation_count_vs_parent": len(seq_mutations),
                "mutations_vs_parent": format_mutations(parent, sequence),
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
        )

    add_record(parent, "baseline_parent", 0, None, None, None)
    add_record(seed_sequence, "baseline_seed_locked_top1_partial", 0, None, sorted(preserved_positions), None)
    add_record(top1_sequence, "baseline_real_top1", 0, None, sorted(idx for idx, _src, _dst in top1_mutations), None)

    for round_idx in range(1, args.rounds + 1):
        model, alphabet, steering_vectors = load_generator(
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
                positions_1based = weighted_sample_without_replacement(hotspot_positions, hotspot_weights, budget, rng)
                candidate_sequences, site_candidates = build_candidates_from_topk(
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
        score_unique_sequences(unique_sequences, args, score_cache)

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
            key=lambda seq: score_cache[seq],
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
        "top1_predicted_brightness": score_cache[top1_sequence],
        "parent_predicted_brightness": score_cache[parent],
        "seed_predicted_brightness": score_cache[seed_sequence],
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
