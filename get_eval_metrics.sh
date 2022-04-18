#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=60000 # Memory - Use up to 40G
#SBATCH --time=0 # No time limit
#SBATCH --mail-user=hongqiay@andrew.cmu.edu
#SBATCH --mail-type=END
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist=boston-2-35

source activate ance

lr=1e-5
save_steps=10000
prev_ckpt=0
end_eval_ckpt=450000
num_feedbacks=3
dataset="marco"
eval_mode="rerank"
python get_eval_metrics.py   \
    --eval  \
    --first_stage_inn_path "/bos/usr0/hongqiay/ANCE/results/ance_fullrank_top1K_${dataset}" \
    --output_dir "/bos/tmp10/hongqiay/prf-query-encoder/models/k_${num_feedbacks}"  \
    --save_steps ${save_steps}  \
    --eval_mode ${eval_mode}  \
    --prev_evaluated_ckpt  ${prev_ckpt}   \
    --ance_checkpoint_path  "/bos/tmp10/hongqiay/ance/${dataset}_output/"   \
    --preprocessed_dir   "/bos/tmp10/hongqiay/ance/${dataset}_preprocessed"  \
    --dev_data_dir "/bos/tmp10/hongqiay/prf-query-encoder/ance_format_data/prf_encoder_dev_${dataset}"  \
    --end_eval_ckpt ${end_eval_ckpt} \
    --dataset ${dataset}  \
    --per_gpu_eval_batch_size 150