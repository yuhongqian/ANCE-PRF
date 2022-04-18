repo_dir=$(builtin cd ..; pwd)
data_dir=${repo_dir}/data

echo "Start generating train query ids..."
python -u get_train_qids.py  \
    --raw_data_dir ${data_dir}/marco_raw_data 
echo "Done generating train query ids." 