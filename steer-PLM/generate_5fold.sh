#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 /data/user/hesy/miniconda3/envs/light_predictor_esm2/bin/python \
  /data/user/hesy/projects/protein_design/steer-PLM/Steering-PLMs/generate_5fold.py \
  --avgfp-csv /data/user/hesy/projects/protein_design/steer-PLM/light_predictor/avgfp_esm2_lora/pplu_data/pplu_split.csv \
  --parent-seq-file "/bigdat2/user/hesy/protein_design/data/2026Protein Design/AAseqs of 5 GFP proteins.txt" \
  --parent-header ppluGFP \
  --ckpt-path /bigdat2/user/hesy/protein_design/last_year/ESM_finetune/esm2_t33_650M_UR50D.pt \
  --hf-base-model /bigdat2/user/hesy/protein_design/last_year/ESM_finetune/facebook_esm2_t33_650M_UR50D \
  --lora-adapter /data/user/hesy/projects/protein_design/steer-PLM/light_predictor/avgfp_esm2_lora/runs/pplu_05_lora_tail_regression_5fold_ensemble \
  --steering-source-split all \
  --steering-pos-quantile 0.90 \
  --steering-neg-quantile 0.10 \
  --steering-num-data 512 \
  --output-csv /data/user/hesy/projects/protein_design/steer-PLM/Steering-PLMs/results/pplu_5fold_6change.csv \
  --summary-json /data/user/hesy/projects/protein_design/steer-PLM/Steering-PLMs/results/pplu_5fold_6change.summary.json \
  --device cuda \
  --rounds 6 \
  --beam-width 64 \
  --branch-factor 100 \
  --top-k-hotspots 80 \
  --min-hotspot-count 3 \
  --preserve-top1-count 0 \
  --mutation-budget-min 0 \
  --mutation-budget-max 1 \
  --residue-top-k 4 \
  --temperature 0.8 \
  --top-p 0.9 \
  --alpha 1.0 \
  --score-batch-size 16 \
  --max-length 256 \
  --seed 42
