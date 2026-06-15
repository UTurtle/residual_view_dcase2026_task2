#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-.}"

COMMON_ARGS=(
  --dataset_name dcase2026
  --train_split eval_train
  --eval_split eval
  --input_type near
  --different_view fixed_residual_view
  --fixed_residual_alpha 0.5
  --top_k 1
  --ld_k 16
  --ld_ref_mode combined
  --temporal_pooling
  --n_mix_support 990
  --alpha 0.9
  --save_official
  --no_wandb
)

run_model() {
  local model_name="$1"
  local pretrained_model_dir="$2"

  python run_residual_view.py \
    --model_name "${model_name}" \
    --pretrained_model_dir "${pretrained_model_dir}" \
    "${COMMON_ARGS[@]}"
}

run_model sslam ./transformer-ssl-asd/sslam
run_model beats_iter3 ./transformer-ssl-asd/beats
run_model dasheng_base .
