#!/usr/bin/env bash

/data/user/hesy/miniconda3/envs/light_predictor_esm2/bin/python \
  /data/user/hesy/projects/protein_design/steer-PLM/light_predictor/avgfp_esm2_lora/prepare_avgfp_dataset.py \
  --xlsx-path "/bigdat2/user/hesy/protein_design/data/2026Protein Design/GFP_data.xlsx" \
  --parent-fasta "/bigdat2/user/hesy/protein_design/data/2026Protein Design/AAseqs of 5 GFP proteins.txt" \
  --output-dir "/data/user/hesy/projects/protein_design/steer-PLM/light_predictor/avgfp_esm2_lora/pplu_data" \
  --target-parent "ppluGFP"

mv /data/user/hesy/projects/protein_design/steer-PLM/light_predictor/avgfp_esm2_lora/pplu_data/avgfp_raw.csv \
  /data/user/hesy/projects/protein_design/steer-PLM/light_predictor/avgfp_esm2_lora/pplu_data/pplu_raw.csv

mv /data/user/hesy/projects/protein_design/steer-PLM/light_predictor/avgfp_esm2_lora/pplu_data/avgfp_dedup.csv \
  /data/user/hesy/projects/protein_design/steer-PLM/light_predictor/avgfp_esm2_lora/pplu_data/pplu_dedup.csv

mv /data/user/hesy/projects/protein_design/steer-PLM/light_predictor/avgfp_esm2_lora/pplu_data/avgfp_split.csv \
  /data/user/hesy/projects/protein_design/steer-PLM/light_predictor/avgfp_esm2_lora/pplu_data/pplu_split.csv
