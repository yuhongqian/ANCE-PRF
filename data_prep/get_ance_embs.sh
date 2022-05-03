# This script uses ANCE FirstP model to generate 
# marco passage embeddings, marco training query embeddings, and dev query embeddings for all the datasets. 

gpu_no=${1:-4}

repo_dir=$(builtin cd ..; pwd)
model_dir=${repo_dir}/model
data_dir=${repo_dir}/data
mkdir -p ${model_dir}

echo "Downloading ANCE model..."
cd ${model_dir}
wget https://webdatamltrainingdiag842.blob.core.windows.net/semistructstore/OpenSource/Passage_ANCE_FirstP_Checkpoint.zip
unzip Passage_ANCE_FirstP_Checkpoint.zip
mv "Passage ANCE(FirstP) Checkpoint" ance_firstp
rm Passage_ANCE_FirstP_Checkpoint.zip

echo "Encoding passages using ANCE model..."
cd ${repo_dir}/data_prep

dataset_array=(
    marco
    trec19psg
    trec20psg
    dlhard
)

for dataset in "${dataset_array[@]}"; do
  echo "Generating ANCE embeddings for dataset ${dataset}..."
  python -m torch.distributed.launch --nproc_per_node=$gpu_no run_ann_data_gen.py \
   --dataset ${dataset} \
   --init_model_dir ${model_dir}/ance_firstp  \
   --model_type rdot_nll \
   --output_dir ${data_dir}/${dataset}_output \
   --cache_dir ${data_dir}/${dataset}_cache \
   --data_dir ${data_dir}/${dataset}_preprocessed \
   --max_seq_length 512 \
   --per_gpu_eval_batch_size 64 \
   --topk_training 200 \
   --negative_sample 20 \
   --end_output_num 0  \
   --inference
  if [ "$dataset" != "marco" ]; then
    ln -s ${data_dir}/marco_output/passage* ${data_dir}/${dataset}_output/
  fi
done