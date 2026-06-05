#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/user/hesy/projects/protein_design/Venus-MAXWELL"
PDB_PATH="${1:-/data/user/hesy/projects/protein_design/ProteinMPNN/data/2G6X.pdb}"
INPUT_CSV="${2:-/data/user/hesy/projects/protein_design/ProteinMPNN/data/pplu_5fold_chainB_unconditional_scores.csv}"
OUTPUT_CSV="${3:-/data/user/hesy/projects/protein_design/ProteinMPNN/data/$(basename "${INPUT_CSV%.csv}")_venus_maxwell.csv}"
LANDSCAPE_CSV="${4:-/data/user/hesy/projects/protein_design/ProteinMPNN/data/2G6X_chainB_venus_maxwell_landscape.csv}"
MAPPING_TSV="${5:-/data/user/hesy/projects/protein_design/ProteinMPNN/data/pplu_5fold_chainB_mapping.tsv}"
CKPT_PATH="${6:-$ROOT/weights/esmif-maxwell.ckpt}"

TRUE_PARENT="${TRUE_PARENT:-MPAMKIECRITGTLNGVEFELVGGGEGTPEQGRMTNKMKSTKGALTFSPYLLSHVMGYGFYHFGTYPSGYENPFLHAINNGGYTNTRIEKYEDGGVLHVSFSYRYEAGRVIGDFKVVGTGFPEDSVIFTDKIIRSNASVEHLHPMGDNVLVGSFARTFSLRDGGYYSFVVDSHMHFKSAIHPSILQNGGPMFAFRRVEELHSNTELEIVEYQHAFKTPIAFA}"

source /data/user/hesy/miniconda3/etc/profile.d/conda.sh
conda activate light_predictor_esm2

cd "$ROOT"

python - <<'PY' "$PDB_PATH" "$INPUT_CSV" "$OUTPUT_CSV" "$LANDSCAPE_CSV" "$MAPPING_TSV" "$CKPT_PATH" "$TRUE_PARENT"
import csv
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import esm

pdb_path = Path(sys.argv[1])
input_csv = Path(sys.argv[2])
output_csv = Path(sys.argv[3])
landscape_csv = Path(sys.argv[4])
mapping_tsv = Path(sys.argv[5])
ckpt_path = Path(sys.argv[6])
true_parent = sys.argv[7].strip()

aa = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']


class MutantNetIF(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.model, self.alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
        self.model.to(device)
        self.model.eval()
        vocab_size = len(self.alphabet)
        self.extra_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.SELU(),
            nn.Linear(512, vocab_size),
        )
        self.aa = aa
        self.aa_index = torch.tensor([self.alphabet.get_idx(a) for a in self.aa], device=device)
        self.device = device

    @torch.no_grad()
    def landscape(self, pdb_file: Path, chain: str = "B"):
        coords, native_seq = esm.inverse_folding.util.load_coords(str(pdb_file), chain)
        batch = [(coords, None, native_seq)]
        batch_converter = esm.inverse_folding.util.CoordBatchConverter(self.alphabet)
        coords, confidence, strs, tokens, padding_mask = batch_converter(batch, device=self.device)
        prev_output_tokens = tokens[:, :-1].to(self.device)
        logits, _ = self.model.forward(coords, padding_mask, confidence, prev_output_tokens)
        logits = logits.transpose(1, 2)
        logits = torch.log_softmax(logits, dim=-1)
        one_hot = F.one_hot(tokens[:, 1:], num_classes=len(self.alphabet))
        logits = logits - (logits * one_hot).sum(dim=-1, keepdim=True)
        logits = logits.squeeze(0)
        landscape = torch.zeros((logits.shape[0], len(self.aa)), device=self.device)
        for i in range(logits.shape[0]):
            landscape[i, :] = logits[i, self.aa_index]
        return native_seq, (-landscape).detach().cpu().numpy()


device = "cuda" if torch.cuda.is_available() else "cpu"
model = MutantNetIF(device=device)
state = torch.load(ckpt_path, map_location="cpu")["state_dict"]
model.load_state_dict(state, strict=False)
model.eval()
chain_seq, landscape = model.landscape(pdb_path, chain="B")

with landscape_csv.open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["chain", "chain_position", "wt_chain_aa", "mutant", "relative_ddg"])
    for i, wt in enumerate(chain_seq, start=1):
        for j, mt in enumerate(aa):
            writer.writerow(["B", i, wt, f"{wt}{i}{mt}", float(landscape[i - 1, j])])

parent_positions = []
chain_wt = []
with mapping_tsv.open() as handle:
    next(handle)
    for line in handle:
        chain_position, pdb_resseq, parent_position, parent_aa, chain_aa = line.rstrip("\n").split("\t")
        parent_positions.append(int(parent_position))
        chain_wt.append(chain_aa)

lookup = {}
for i, wt in enumerate(chain_seq, start=1):
    for j, mt in enumerate(aa):
        lookup[(i, wt, mt)] = float(landscape[i - 1, j])

parent_trimmed = "".join(true_parent[pos - 1] for pos in parent_positions)
parent_contribs = []
for pos, (obs, wt) in enumerate(zip(parent_trimmed, chain_wt), start=1):
    if obs != wt:
        parent_contribs.append(lookup[(pos, wt, obs)])
parent_sum = sum(parent_contribs)
parent_mean = parent_sum / len(parent_contribs) if parent_contribs else 0.0

rows = list(csv.DictReader(input_csv.open()))
fieldnames = list(rows[0].keys())
extra_cols = [
    "parent_sequence_ref",
    "venus_maxwell_chain",
    "venus_maxwell_struct_positions",
    "venus_maxwell_mutated_vs_chain_count",
    "venus_maxwell_relative_ddg_sum",
    "venus_maxwell_relative_ddg_mean",
    "venus_maxwell_delta_vs_parent_sum",
    "venus_maxwell_delta_vs_parent_mean",
    "venus_maxwell_changed_positions",
]
for col in extra_cols:
    if col not in fieldnames:
        fieldnames.append(col)

for row in rows:
    seq = row["sequence"].strip()
    trimmed = "".join(seq[pos - 1] for pos in parent_positions)
    contribs = []
    changed = []
    for pos, (obs, wt) in enumerate(zip(trimmed, chain_wt), start=1):
        if obs != wt:
            score = lookup[(pos, wt, obs)]
            contribs.append(score)
            changed.append(f"{wt}{pos}{obs}|relative_ddg={score:.4f}")
    total = sum(contribs)
    mean = total / len(contribs) if contribs else 0.0
    row.update(
        {
            "parent_sequence_ref": true_parent,
            "venus_maxwell_chain": "B",
            "venus_maxwell_struct_positions": len(parent_positions),
            "venus_maxwell_mutated_vs_chain_count": len(contribs),
            "venus_maxwell_relative_ddg_sum": f"{total:.6f}",
            "venus_maxwell_relative_ddg_mean": f"{mean:.6f}",
            "venus_maxwell_delta_vs_parent_sum": f"{total - parent_sum:.6f}",
            "venus_maxwell_delta_vs_parent_mean": f"{mean - parent_mean:.6f}",
            "venus_maxwell_changed_positions": ";".join(changed),
        }
    )

with output_csv.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved landscape to {landscape_csv}")
print(f"Saved Venus-MAXWELL scores to {output_csv}")
PY
