# This 
lr=1e-5
save_steps=2000
gpu_id=${1:-2}
end_eval_ckpt=1
num_feedbacks=3
repo_dir=$(pwd)
output_dir=${repo_dir}/outputs 
data_dir=${repo_dir}/data
dataset="marco"
eval_mode="rerank"  # switch to "full" if you would like to see the full retrieval results

CUDA_VISIBLE_DEVICES=${gpu_id} python main.py   \
    --eval  \
    --first_stage_inn_path ${data_dir}/${dataset}_output/${dataset}_dev_I.npy  \
    --output_dir ${output_dir}/k_${num_feedbacks}  \
    --save_steps ${save_steps}  \
    --eval_mode ${eval_mode}  \
    --ance_checkpoint_path  ${data_dir}/${dataset}_output/   \
    --preprocessed_dir   ${data_dir}/${dataset}_preprocessed  \
    --dev_data_dir ${data_dir}/${dataset}_output \
    --dataset ${dataset}  \
    --per_gpu_eval_batch_size 50



