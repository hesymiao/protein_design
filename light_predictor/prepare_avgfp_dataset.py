import argparse
import json
import re
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

MUTATION_RE = re.compile(r"^([A-Z])(\d+)([A-Z*])$")


def load_parent_sequences(path: Path) -> Dict[str, str]:
    parents = {}
    name = None
    seq_parts = []
    with path.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    parents[name] = "".join(seq_parts)
                name = line[1:].strip()
                seq_parts = []
            else:
                seq_parts.append(line)
    if name is not None:
        parents[name] = "".join(seq_parts)
    return parents


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def resolve_parent_name(name: str, parents: Dict[str, str]) -> str:
    norm_name = normalize_name(name)
    norm_parents = {normalize_name(key): key for key in parents}
    if norm_name in norm_parents:
        return norm_parents[norm_name]
    if norm_name == "pplugfp" and "pplugfp2" in norm_parents:
        return norm_parents["pplugfp2"]
    for candidate_norm, candidate_name in norm_parents.items():
        if candidate_norm.startswith(norm_name) or norm_name.startswith(candidate_norm):
            return candidate_name
    raise KeyError(f"Could not resolve parent type: {name}")


def apply_mutations(parent_sequence: str, mutation_spec: str) -> Tuple[str, int, bool]:
    if mutation_spec == "WT":
        return parent_sequence, 0, False

    sequence = list(parent_sequence)
    count = 0
    has_stop = False
    for token in mutation_spec.split(":"):
        match = MUTATION_RE.match(token)
        if not match:
            raise ValueError(f"Unsupported mutation format: {token}")
        from_aa, pos_str, to_aa = match.groups()
        # The mutation table numbers residues after the initiator methionine,
        # so position N in the spreadsheet maps to index N in the full FASTA.
        pos = int(pos_str)
        if pos < 0 or pos >= len(sequence):
            raise IndexError(f"Mutation position out of range: {token}")
        if sequence[pos] != from_aa:
            raise ValueError(
                f"Mutation parent mismatch for {token}: expected {from_aa}, found {sequence[pos]}"
            )
        count += 1
        if to_aa == "*":
            sequence = sequence[:pos]
            has_stop = True
            break
        sequence[pos] = to_aa
    return "".join(sequence), count, has_stop


def make_stratify_bins(scores: pd.Series, max_bins: int) -> pd.Series | None:
    unique_scores = scores.nunique()
    bins = min(max_bins, unique_scores)
    if bins < 2:
        return None
    ranked = scores.rank(method="first")
    return pd.qcut(ranked, q=bins, labels=False, duplicates="drop")


def split_dataset(
    df: pd.DataFrame,
    seed: int,
    val_fraction: float,
    test_fraction: float,
    stratify_bins: int,
) -> pd.DataFrame:
    if val_fraction <= 0 or test_fraction <= 0 or val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction and test_fraction must be > 0 and sum to < 1")

    test_rows = max(1, int(round(len(df) * test_fraction)))
    val_rows = max(1, int(round(len(df) * val_fraction)))
    effective_bins = min(stratify_bins, test_rows, val_rows)

    stratify_all = make_stratify_bins(df["score"], effective_bins)
    train_val, test = train_test_split(
        df,
        test_size=test_fraction,
        random_state=seed,
        stratify=stratify_all,
    )

    val_size_within_train_val = val_fraction / (1.0 - test_fraction)
    stratify_train_val = make_stratify_bins(train_val["score"], effective_bins)
    train, val = train_test_split(
        train_val,
        test_size=val_size_within_train_val,
        random_state=seed,
        stratify=stratify_train_val,
    )

    train = train.copy()
    val = val.copy()
    test = test.copy()
    train["split"] = "train"
    val["split"] = "val"
    test["split"] = "test"

    return pd.concat([train, val, test], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xlsx-path", required=True, type=Path)
    parser.add_argument("--parent-fasta", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--target-parent", default="avGFP")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--test-fraction", type=float, default=0.05)
    parser.add_argument("--stratify-bins", type=int, default=10)
    parser.add_argument("--full-length-only", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    parents = load_parent_sequences(args.parent_fasta)
    data = pd.read_excel(args.xlsx_path)
    data["resolved_parent"] = data["GFP type"].map(lambda value: resolve_parent_name(value, parents))

    target_norm = normalize_name(args.target_parent)
    data = data[data["resolved_parent"].map(normalize_name) == target_norm].copy()
    data = data.dropna(subset=["aaMutations", "Brightness"])
    data["Brightness"] = data["Brightness"].astype(float)

    records = []
    errors = []
    parent_key = resolve_parent_name(args.target_parent, parents)
    parent_sequence = parents[parent_key]
    parent_length = len(parent_sequence)

    for row in data.to_dict(orient="records"):
        mutation_spec = str(row["aaMutations"])
        try:
            sequence, num_mutations, has_stop = apply_mutations(parent_sequence, mutation_spec)
        except Exception as exc:  # pragma: no cover - diagnostic path
            errors.append({"mutation": mutation_spec, "error": str(exc)})
            continue

        records.append(
            {
                "sequence": sequence,
                "sequence_length": len(sequence),
                "score": float(row["Brightness"]),
                "parent_type": row["GFP type"],
                "resolved_parent": row["resolved_parent"],
                "mutations": mutation_spec,
                "num_mutations": num_mutations,
                "has_stop": has_stop,
            }
        )

    raw_df = pd.DataFrame.from_records(records)
    raw_rows_before_filter = int(len(raw_df))
    deduplicated_rows_before_filter = int(raw_df["sequence"].nunique())
    stop_rows_before_filter = int(raw_df["has_stop"].sum())

    if args.full_length_only:
        raw_df = raw_df[(~raw_df["has_stop"]) & (raw_df["sequence_length"] == parent_length)].copy()

    raw_path = args.output_dir / "avgfp_raw.csv"
    raw_df.to_csv(raw_path, index=False)

    dedup_df = (
        raw_df.groupby("sequence", as_index=False)
        .agg(
            sequence_length=("sequence_length", "first"),
            score=("score", "mean"),
            score_std=("score", "std"),
            n_measurements=("score", "size"),
            parent_type=("parent_type", "first"),
            resolved_parent=("resolved_parent", "first"),
            mutations=("mutations", "first"),
            num_mutations=("num_mutations", "first"),
            has_stop=("has_stop", "first"),
        )
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )
    dedup_df["score_std"] = dedup_df["score_std"].fillna(0.0)

    dedup_path = args.output_dir / "avgfp_dedup.csv"
    dedup_df.to_csv(dedup_path, index=False)

    split_df = split_dataset(
        dedup_df,
        seed=args.seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        stratify_bins=args.stratify_bins,
    )
    split_path = args.output_dir / "avgfp_split.csv"
    split_df.to_csv(split_path, index=False)

    for split_name in ("train", "val", "test"):
        split_df[split_df["split"] == split_name].to_csv(
            args.output_dir / f"{split_name}.csv",
            index=False,
        )

    summary = {
        "target_parent": args.target_parent,
        "parent_length": int(parent_length),
        "full_length_only": bool(args.full_length_only),
        "seed": args.seed,
        "raw_rows_before_filter": raw_rows_before_filter,
        "raw_rows": int(len(raw_df)),
        "deduplicated_rows_before_filter": int(deduplicated_rows_before_filter),
        "deduplicated_rows": int(len(dedup_df)),
        "split_counts": split_df["split"].value_counts().to_dict(),
        "mean_score": float(dedup_df["score"].mean()),
        "std_score": float(dedup_df["score"].std()),
        "stop_mutation_rows_before_filter": stop_rows_before_filter,
        "stop_mutation_rows": int(raw_df["has_stop"].sum()),
        "failed_rows": int(len(errors)),
    }

    if errors:
        pd.DataFrame(errors).to_csv(args.output_dir / "reconstruction_errors.csv", index=False)

    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved raw dataset to: {raw_path}")
    print(f"Saved deduplicated dataset to: {dedup_path}")
    print(f"Saved split dataset to: {split_path}")


if __name__ == "__main__":
    main()
