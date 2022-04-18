#!/usr/bin/env bash

source activate ance
dataset=$1
ckpt=$2
mode="fullrank"
num_feedbacks=$3
python get_marco_eval_output.py  \
  --ance_checkpoint_path "/bos/tmp10/hongqiay/ance/${dataset}_output/"   \
  --processed_data_dir "/bos/tmp10/hongqiay/ance/${dataset}_preprocessed"  \
  --devI_path "/bos/tmp10/hongqiay/prf-query-encoder/models/k_${num_feedbacks}/checkpoint-${ckpt}/${dataset}_devI_${mode}.npy"
