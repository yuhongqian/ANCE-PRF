lr=1e-5
save_steps=2000
prev_ckpt=-100
end_eval_ckpt=1
num_feedbacks=3
repo_dir=$(pwd)
output_dir=${repo_dir}/outputs 
data_dir=${repo_dir}/data
dataset="marco"
eval_mode="rerank"
python main.py   \
    --eval  \
    --first_stage_inn_path ${data_dir}/ance_fullrank_1k_${dataset}_dev  \
    --output_dir ${output_dir}/k_${num_feedbacks}  \
    --save_steps ${save_steps}  \
    --eval_mode ${eval_mode}  \
    --prev_evaluated_ckpt  ${prev_ckpt}   \
    --ance_checkpoint_path  ${data_dir}/${dataset}_output/   \
    --preprocessed_dir   ${data_dir}/${dataset}_preprocessed  \
    --dev_data_dir ${data_dir}/${dataset}_dev_prf \
    --dataset ${dataset}  \
    --per_gpu_eval_batch_size 50
