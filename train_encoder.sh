gpu_no=2
lr=1e-5
num_feedbacks=3
repo_dir=$(pwd)
output_dir=${repo_dir}/outputs 
data_dir=${repo_dir}/data
mkdir -p ${output_dir}

python -m torch.distributed.launch --nproc_per_node=${gpu_no} main.py   \
    --train   \
    --logging_steps 100 \
    --save_steps 2000  \
    --gradient_accumulation_steps 8   \
    --warmup_steps=5000   \
    --output_dir ${output_dir}/k_${num_feedbacks}  \
    --learning_rate ${lr}  \
    --num_feedbacks ${num_feedbacks}  \
    --per_gpu_train_batch_size 4 \
    --load_optimizer_scheduler  \
    --ance_checkpoint_path ${data}/marco_output \
    --preprocessed_dir ${data}/marco_preprocessed  \
    --train_data_dir ${data_dir}/${dataset}_train_prf