# Run initial retrieval on all the datasets. 

repo_dir=$(builtin cd ..; pwd)
data_dir=${repo_dir}/data
model_dir=${repo_dir}/model

pids=""

dataset="marco"
echo "Getting initial ANCE ranking on ${dataset} train set..."
python -u get_ance_ranking.py  \
--processed_data_dir ${data_dir}/${dataset}_preprocessed \
--ance_checkpoint_path ${data_dir}/${dataset}_output \
--dataset ${dataset} \
--mode "train" & 
pids="$pids $!"

dataset_array=(
    marco
    trec19psg
    trec20psg
    dlhard
)

for dataset in "${dataset_array[@]}"; do
    echo "Getting initial ANCE ranking on ${dataset} dev set..."
    python -u get_ance_ranking.py  \
    --processed_data_dir ${data_dir}/${dataset}_preprocessed \
    --ance_checkpoint_path ${data_dir}/${dataset}_output \
    --dataset ${dataset} \
    --mode "dev" &
    pids="$pids $!"
done

for pid in $pids; do
    wait $pid
done
echo "Initial ranking on all datasets done." 


