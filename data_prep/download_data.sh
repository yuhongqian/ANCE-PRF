# This script downloads all data used in the paper.
# It also renames the data files using the same format,
# so that the same preprocessing pipeline can be applied more easily.

repo_dir=$(pwd)/..
data_dir=${repo_dir}/data
mkdir -p ${data_dir}

echo "Downloading MS MARCO passage data..."
dataset="marco"
raw_data_dir=${data_dir}/${dataset}_raw_data
mkdir -p ${raw_data_dir}
cd ${raw_data_dir}

wget https://msmarco.blob.core.windows.net/msmarcoranking/collectionandqueries.tar.gz
tar -zxvf collectionandqueries.tar.gz
rm collectionandqueries.tar.gz

wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-passagetest2019-top1000.tsv.gz
gunzip msmarco-passagetest2019-top1000.tsv.gz

wget https://msmarco.blob.core.windows.net/msmarcoranking/top1000.dev.tar.gz
tar -zxvf top1000.dev.tar.gz
mv top1000.dev top1000.dev.tsv
rm top1000.dev.tar.gz

wget https://msmarco.blob.core.windows.net/msmarcoranking/triples.train.small.tar.gz
tar -zxvf triples.train.small.tar.gz
rm triples.train.small.tar.gz

echo "Downloading TREC DL 2019 passage data..."
dataset="trec19psg"
raw_data_dir=${data_dir}/${dataset}_raw_data
mkdir -p ${raw_data_dir}
cd ${raw_data_dir}

# trec dl 19 shares marco corpus
ln -s ${data_dir}/marco_raw_data/collection.tsv .

wget --no-check-certificate https://trec.nist.gov/data/deep/2019qrels-pass.txt -O qrels.dev.small.tsv

wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz
gunzip msmarco-test2019-queries.tsv.gz
mv msmarco-test2019-queries.tsv queries.dev.small.tsv

wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-passagetest2019-top1000.tsv.gz
gunzip msmarco-passagetest2019-top1000.tsv.gz
mv msmarco-passagetest2019-top1000.tsv top1000.dev.tsv


echo "Downloading TREC DL 2020 passage data..."
dataset="trec20psg"
raw_data_dir=${data_dir}/${dataset}_raw_data
mkdir -p ${raw_data_dir}
cd ${raw_data_dir}

# trec dl 20 shares marco corpus
ln -s ${data_dir}/marco_raw_data/collection.tsv .

wget --no-check-certificate https://trec.nist.gov/data/deep/2020qrels-pass.txt -O qrels.dev.small.tsv

wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-test2020-queries.tsv.gz
gunzip msmarco-test2020-queries.tsv.gz
mv msmarco-test2020-queries.tsv queries.dev.small.tsv

wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-passagetest2020-top1000.tsv.gz
gunzip msmarco-passagetest2020-top1000.tsv.gz
mv msmarco-passagetest2020-top1000.tsv top1000.dev.tsv


echo "Downloading DL-HARD passage data..."
# Using the files from this commit: https://github.com/grill-lab/DL-Hard/commit/c58ce8d9e8932a7b560c0f2cb3435b6c2db578fe

dataset="dlhard"
raw_data_dir=${data_dir}/${dataset}_raw_data
mkdir -p ${raw_data_dir}
cd ${raw_data_dir}

# dl-hard shares marco corpus
ln -s ${data_dir}/marco_raw_data/collection.tsv .
wget https://raw.githubusercontent.com/grill-lab/DL-Hard/main/dataset/dl_hard-passage.qrels -O qrels.dev.small.tsv
wget https://raw.githubusercontent.com/grill-lab/DL-Hard/main/dataset/topics.tsv -O queries.dev.small.tsv
wget https://raw.githubusercontent.com/grill-lab/DL-Hard/main/dataset/baselines/passage/bm25.run -O top1000.dev.tsv

echo "Finished downloading data."
