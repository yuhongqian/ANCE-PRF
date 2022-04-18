repo_dir=$(builtin cd ..; pwd)
data_dir=${repo_dir}/data
model_dir=${repo_dir}/model

pids=""
for i in {0..3}; do 
    echo "Getting initial ANCE ranking on split ${i}..."
    python -u get_ance_ranking.py  \
    --processed_data_dir ${data_dir}/marco_preprocessed \
    --output_dir ${data_dir}/marco_output \
    --split ${i}  & 
    pids="$pids $!"
done

for pid in $pids; do
    wait $pid
done
echo "Initial ranking on all splits done." 


