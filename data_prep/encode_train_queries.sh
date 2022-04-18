repo_dir=$(builtin cd ..; pwd)
data_dir=${repo_dir}/data
model_dir=${repo_dir}/model

# total number of GPUs on machine
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
CUDA_ID=-1

# assuming all GPUs have the same memory 
total_mem=$(nvidia-smi --query-gpu=memory.total --format=csv -i 0 | grep -Eo [0-9]+)

# There are 398792 training queries in total. 
# Each of the 4 GPUs is responsible for computing ~100K query embeddings. 
pids=""
for i in {0..3}; do 
    free_mem=0
    while [ $free_mem -lt $((total_mem-10)) ]; do
        CUDA_ID=$(((CUDA_ID+1)%NUM_GPUS))
        free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv -i $CUDA_ID | grep -Eo [0-9]+)
        sleep 60s
    done
    echo "Generating train query embeddings for split ${i} on GPU ${CUDA_ID}."
    CUDA_VISIBLE_DEVICES=$CUDA_ID python -u get_train_query_embeds.py  \
    --raw_data_dir ${data_dir}/marco_raw_data \
    --output_dir ${data_dir}/marco_output \
    --model_name_or_path ${model_dir}/ance_firstp  \
    --split ${i} \
    --chunk_size 100000 & 
    pids="$pids $!"
done

for pid in $pids; do
    wait $pid
done
echo "Done generating train query embeddings for all splits."


