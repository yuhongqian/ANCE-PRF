# Generates PRF data for all datasets. 

data_dir=$(builtin cd ../data; pwd)

pids=""

echo "Generating PRF training data from ANCE top ranking on MARCO training set..."
dataset="marco"
mode="train"
python -u get_prf_data.py  \
    --processed_data_dir ${data_dir}/${dataset}_preprocessed  \
    --output_dir ${data_dir}/${dataset}_output \
    --inn_path ${data_dir}/${dataset}_output/${dataset}_${mode}_I.npy \
    --ance_checkpoint_path ${data_dir}/${dataset}_output \
    --dataset ${dataset} \
    --mode train & 
pids="$pids $!"


dataset_array=(
    marco
    trec19psg
    trec20psg
    dlhard
)

mode="dev"
for dataset in "${dataset_array[@]}"; do
  echo "Generating PRF dev data from ANCE top ranking on ${dataset} dev set..."
  python -u get_prf_data.py  \
    --processed_data_dir ${data_dir}/${dataset}_preprocessed  \
    --output_dir ${data_dir}/${dataset}_output \
    --inn_path ${data_dir}/${dataset}_output/${dataset}_${mode}_I.npy \
    --ance_checkpoint_path ${data_dir}/${dataset}_output \
    --dataset ${dataset} \
    --mode dev &
  pids="$pids $!"
done

for pid in $pids; do
    wait $pid
done
echo "Generated PRF data for all datasets." 