data_dir=$(builtin cd ../data; pwd)
echo "Generating ANCE-PRF training data from ANCE top ranking on MARCO training set..."
dataset="marco"
num_chunks=100
for ((i=0;i<num_chunks;i++)); do
  echo "Generating training data chunk ${i}..."
  python -u get_data_ann_format.py  \
      --generate_train \
      --processed_data_dir ${data_dir}/${dataset}_preprocessed  \
      --output_dir ${data_dir}/${dataset}_train_prf \
      --ann_chunk_factor=${num_chunks}  \
      --ance_checkpoint_path ${data_dir}/${dataset}_output \
      --split ${i}
done

dataset_array=(
    marco
    trec19psg
    trec20psg
    dlhard
)

for dataset in "${dataset_array[@]}"; do
  echo "Generating ANCE-PRF dev data from ANCE top ranking on ${dataset} dev/test set..."
  python -u get_data_ann_format.py  \
    --generate_dev  \
    --processed_data_dir ${data_dir}/${dataset}_preprocessed  \
    --output_dir ${data_dir}/${dataset}_dev_prf \
    --dev_inn_path ${data_dir}/ance_fullrank_1k_${dataset}_dev  \
    --ance_checkpoint_path ${data_dir}/${dataset}_output \
    --split 0
done