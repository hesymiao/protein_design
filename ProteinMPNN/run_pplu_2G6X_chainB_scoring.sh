#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/user/hesy/projects/protein_design/ProteinMPNN"
PDB_PATH="${1:-$ROOT/data/2G6X.pdb}"
INPUT_CSV="${2:-$ROOT/data/pplu_5fold.csv}"
OUTPUT_CSV="${3:-$ROOT/data/$(basename "${INPUT_CSV%.csv}")_chainB_unconditional_scores.csv}"
OUT_FOLDER="${4:-$ROOT/data/2G6X_chainB_unconditional}"
MAPPING_TSV="${5:-$ROOT/data/pplu_5fold_chainB_mapping.tsv}"

source /data/user/hesy/miniconda3/etc/profile.d/conda.sh
conda activate light_predictor_esm2

cd "$ROOT"

python protein_mpnn_run.py \
  --pdb_path "$PDB_PATH" \
  --pdb_path_chains B \
  --out_folder "$OUT_FOLDER" \
  --unconditional_probs_only 1 \
  --batch_size 1 \
  --num_seq_per_target 1 \
  --seed 13

python - <<'PY' "$PDB_PATH" "$INPUT_CSV" "$OUTPUT_CSV" "$MAPPING_TSV" "$OUT_FOLDER/unconditional_probs_only/2G6X.npz"
import csv
import sys
from pathlib import Path

import numpy as np

pdb_path = Path(sys.argv[1])
input_csv = Path(sys.argv[2])
output_csv = Path(sys.argv[3])
mapping_tsv = Path(sys.argv[4])
npz_path = Path(sys.argv[5])

map3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "CR2": "Y",
}


def needleman_wunsch(a: str, b: str, match: int = 2, mismatch: int = -1, gap: int = -2):
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    bt = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + gap
        bt[i][0] = "U"
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + gap
        bt[0][j] = "L"
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            candidates = [
                (dp[i - 1][j - 1] + (match if ai == b[j - 1] else mismatch), "D"),
                (dp[i - 1][j] + gap, "U"),
                (dp[i][j - 1] + gap, "L"),
            ]
            dp[i][j], bt[i][j] = max(candidates, key=lambda x: x[0])
    i, j = n, m
    aa, bb = [], []
    while i > 0 or j > 0:
        move = bt[i][j]
        if move == "D":
            aa.append(a[i - 1])
            bb.append(b[j - 1])
            i -= 1
            j -= 1
        elif move == "U":
            aa.append(a[i - 1])
            bb.append("-")
            i -= 1
        else:
            aa.append("-")
            bb.append(b[j - 1])
            j -= 1
    return "".join(reversed(aa)), "".join(reversed(bb))


seen = set()
chain = []
with pdb_path.open() as handle:
    for line in handle:
        if line.startswith("ATOM") and line[21] == "B":
            key = (line[22:26], line[26])
            if key in seen:
                continue
            seen.add(key)
            chain.append((int(line[22:26]), map3.get(line[17:20].strip(), "X")))
chain_seq = "".join(aa for _, aa in chain)

with input_csv.open() as handle:
    rows = list(csv.DictReader(handle))
parent = rows[0]["parent_sequence"]

aligned_parent, aligned_chain = needleman_wunsch(parent, chain_seq)
parent_indices = []
mapping_rows = []
parent_pos = 0
chain_pos = 0
for a, b in zip(aligned_parent, aligned_chain):
    if a != "-":
        parent_pos += 1
    if b != "-":
        chain_pos += 1
    if a != "-" and b != "-":
        parent_indices.append(parent_pos - 1)
        mapping_rows.append((chain_pos, chain[chain_pos - 1][0], parent_pos, a, b))

with mapping_tsv.open("w") as handle:
    handle.write("chain_position\tpdb_resseq\tparent_position\tparent_aa\tchain_aa\n")
    for row in mapping_rows:
        handle.write("\t".join(map(str, row)) + "\n")

arr = np.load(npz_path)
alphabet = "ACDEFGHIKLMNPQRSTVWYX"
aa_to_idx = {aa: i for i, aa in enumerate(alphabet)}
log_p = arr["log_p"][0]
design_idx = np.where(arr["design_mask"].astype(int) == 1)[0]
log_p = log_p[design_idx]

parent_trimmed = "".join(parent[i] for i in parent_indices)
parent_score = float(-np.mean([log_p[i, aa_to_idx[aa]] for i, aa in enumerate(parent_trimmed)]))

fieldnames = list(rows[0].keys()) + [
    "proteinmpnn_chain",
    "proteinmpnn_trimmed_length",
    "proteinmpnn_backbone_nll_mean",
    "proteinmpnn_backbone_delta_vs_parent",
    "proteinmpnn_mapped_mutation_count",
    "proteinmpnn_unmapped_mutation_count",
    "proteinmpnn_mut_delta_nll_sum",
    "proteinmpnn_mut_delta_nll_mean",
    "proteinmpnn_mapped_mutations",
]

parent_to_chain = {parent_pos: chain_pos for chain_pos, _, parent_pos, _, _ in mapping_rows}

with output_csv.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        seq = row["sequence"]
        trimmed = "".join(seq[i] for i in parent_indices)
        nlls = [-float(log_p[i, aa_to_idx[aa]]) for i, aa in enumerate(trimmed)]
        score = float(np.mean(nlls))
        mapped = []
        unmapped = []
        deltas = []
        muts = (row.get("mutations_vs_parent") or "").strip()
        if muts:
            for mut in muts.split(";"):
                if not mut:
                    continue
                wt = mut[0]
                pos = int(mut[1:-1])
                mt = mut[-1]
                chain_pos = parent_to_chain.get(pos)
                if chain_pos is None:
                    unmapped.append(mut)
                    continue
                idx = chain_pos - 1
                delta = (-float(log_p[idx, aa_to_idx[mt]])) - (-float(log_p[idx, aa_to_idx[wt]]))
                deltas.append(delta)
                mapped.append(f"{mut}|chainB:{chain_pos}|delta_nll={delta:.4f}")
        row.update(
            {
                "proteinmpnn_chain": "B",
                "proteinmpnn_trimmed_length": len(parent_indices),
                "proteinmpnn_backbone_nll_mean": f"{score:.6f}",
                "proteinmpnn_backbone_delta_vs_parent": f"{score - parent_score:.6f}",
                "proteinmpnn_mapped_mutation_count": len(mapped),
                "proteinmpnn_unmapped_mutation_count": len(unmapped),
                "proteinmpnn_mut_delta_nll_sum": f"{sum(deltas):.6f}" if deltas else "",
                "proteinmpnn_mut_delta_nll_mean": f"{(sum(deltas) / len(deltas)):.6f}" if deltas else "",
                "proteinmpnn_mapped_mutations": ";".join(mapped),
            }
        )
        writer.writerow(row)

print(f"Saved mapping to {mapping_tsv}")
print(f"Saved scores to {output_csv}")
PY
