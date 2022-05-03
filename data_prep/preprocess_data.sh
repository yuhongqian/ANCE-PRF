# This script does two things: 
# 1. Multi-process tokenization of the passages & queries, and save the tokenized data in binary format: 
#    (1) Binary passage tokens are saved in `passages`
#    (2) Binary query tokens are saved in `train-query` and `dev-query`
#    Note that files suffixed with `_split*` are simply unmerged segments of (1) & (2). 
# 2. Save pid2offset & qid2offset since multiprocessing messed up the passage & query order. 
#    pid & qid are the ids from the original dataset. offsets are the orders of the passages/queries in the multiprocessed binary.  


dataset_array=(
    marco
    trec19psg
    trec20psg
    dlhard
)
data_dir=$(builtin cd ../data; pwd)

for dataset in "${dataset_array[@]}"; do
  echo "Preprocessing ${dataset} data..."
  python preprocess_data.py \
    --dataset ${dataset}  \
    --data_dir ${data_dir}/${dataset}_raw_data \
    --out_data_dir ${data_dir}/${dataset}_preprocessed/ \
    --model_type rdot_nll \
    --model_name_or_path roberta-base \
    --max_seq_length 512 \
    --data_type 1
  if [ "$dataset" != "marco" ]; then
    # create soft-link to use preprocessed passage data from marco
    ln -s ${data_dir}/marco_preprocessed/passage* ${data_dir}/${dataset}_preprocessed/
    ln -s ${data_dir}/marco_preprocessed/pid2offset.pickle ${data_dir}/${dataset}_preprocessed/pid2offset.pickle
  fi
done

echo "Finished preprocessing data."