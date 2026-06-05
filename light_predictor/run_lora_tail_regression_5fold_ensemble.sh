#!/usr/bin/env bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

CUDA_VISIBLE_DEVICES=4 /data/user/hesy/miniconda3/envs/light_predictor_esm2/bin/python \
  /data/user/hesy/projects/protein_design/steer-PLM/light_predictor/avgfp_esm2_lora/train_esm2_lora_tail_regression_5fold_ensemble.py \
  --input-csv /data/user/hesy/projects/protein_design/steer-PLM/light_predictor/avgfp_esm2_lora/pplu_data/pplu_split.csv \
  --output-dir /data/user/hesy/projects/protein_design/steer-PLM/light_predictor/avgfp_esm2_lora/runs/pplu_05_lora_tail_regression_5fold_ensemble \
  --model-name /bigdat2/user/hesy/protein_design/last_year/ESM_finetune/facebook_esm2_t33_650M_UR50D \
  --max-length 256 \
  --seed 42 \
  --num-folds 5 \
  --num-train-epochs 6 \
  --tail-refine-epochs 4 \
  --learning-rate 3e-4 \
  --tail-refine-learning-rate 1e-4 \
  --weight-decay 0.01 \
  --warmup-ratio 0.1 \
  --tail-refine-warmup-ratio 0.05 \
  --per-device-train-batch-size 20 \
  --per-device-eval-batch-size 20 \
  --gradient-accumulation-steps 1 \
  --dataloader-num-workers 0 \
  --lora-r 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  --bf16 \
  --label-transform zscore \
  --regression-loss huber \
  --huber-delta 1.0 \
  --stage1-rank-weight 0.0 \
  --stage2-rank-weight 0.5 \
  --rank-tail-quantile 0.90 \
  --rank-min-gap 0.02 \
  --stage1-top10-weight 1.5 \
  --stage1-top5-weight 2.0 \
  --stage1-top2-weight 3.0 \
  --stage1-top1-weight 3.0 \
  --stage2-top10-weight 2.0 \
  --stage2-top5-weight 4.0 \
  --stage2-top2-weight 8.0 \
  --stage2-top1-weight 8.0 \
  --stage1-metric-for-best-model pearson \
  --stage2-metric-for-best-model top5_pearson
