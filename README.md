# Improving Query Representations for Dense Retrieval with Pseudo Relevance Feedback

HongChien Yu, Chenyan Xiong, Jamie Callan

This repository holds code that reproduces results reported in 
[Improving Query Representations for Dense Retrieval with Pseudo Relevance Feedback](https://arxiv.org/abs/2108.13454). 


Dense retrieval systems conduct first-stage retrieval using embedded representations and simple similarity metrics to 
match a query to documents. Its effectiveness depends on encoded embeddings to capture the semantics of queries and 
documents, a challenging task due to the shortness and ambiguity of search queries. This paper proposes ANCE-PRF, 
a new query encoder that uses pseudo relevance feedback (PRF) to improve query representations for dense retrieval. 
ANCE-PRF uses a BERT encoder that consumes the query and the top retrieved documents from a dense retrieval model, 
ANCE, and it learns to produce better query embeddings directly from relevance labels. 
It also keeps the document index unchanged to reduce overhead. ANCE-PRF significantly outperforms ANCE and other recent
dense retrieval systems on several datasets. Analysis shows that the PRF encoder effectively captures the relevant and 
complementary information from PRF documents, while ignoring the noise with its learned attention mechanism.

## Requirements
```
pip install -r requirements.txt
```
## Data Preparation 

### Data Preprocessing
Run the following script to preprocess data: 
```angular2html
cd data_prep
bash download_data.sh [tested, all files seemed there.]
bash preprocess_data.sh [tested with improved .sh comments so that it shows actual intention of the code]
bash get_train_qids.sh [this doesn't look useful. haven't run for now]
```


### Get ANCE Passage Embeddings
```angular2html
cd data_prep
bash get_ance_embs.sh [tested, runnable]
```


### Get ANCE Ranking
# on CPU: 
```
bash get_ance_ranking.sh [tested, runnable. tein files also matched] 
```

### Prepare PRF data 
Run the following command to create PRF data from ANCE top-retrieved documents: 
```angular2html
cd data_prep 
bash get_prf_data.sh
```

## Training 
```angular2html
bash train_encoder.sh
```
While training is running, concurrently run
```
bash eval.sh
```
which keeps looking for the newest checkpoints and evaluate it on marco. 
This is sadly not a very effective use of GPU in terms of utilization percentage, but it makes the training faster by avoiding periodic switching from training to evaluation. 

In our work, we picked the model that performs best on marco dev as reported by `eval.sh` tensorboard. 


## Trained Models and Ranking Files 
Trained models for k=3 can be downloaded [here](https://drive.google.com/file/d/1xbMgP0Z5tuoqymbWUhfuvRvUx6TvNuVw/view?usp=sharing).

Ranking files for k=3 can be downloaded [here](https://drive.google.com/drive/folders/1FybKqWbE1Ap1xDd8MR01ZOqXn9W0Xy8b?usp=sharing). 

