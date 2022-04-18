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

### Data preprocessing
Run the following script to preprocess data (on CPU): 
```angular2html
cd data_prep
bash download_data.sh data
bash preprocess_data.sh
bash get_train_qids.sh
```


### Get ANCE Passage Embeddings
```angular2html
cd data_prep
bash get_ance_psg_embs.sh
```


### Get ANCE Ranking
Run the following script to obtain ANCE passage embeddings: 
```angular2html
cd data_prep
# on GPU: 
bash encode_train_queries.sh
# on CPU: 
bash get_ance_ranking.sh
```


### Prepare PRF data 
Download all the files from [this folder](https://drive.google.com/drive/folders/1umsN7rnlWAcLZBZPuXs5ay5sdIQuIOct?usp=sharing) and put them in `./data` (e.g., `./data/ance_fullrank_1k_marco_dev`). These files contain the top 1k passages ranked by ANCE. 

Then, run the following command to create PRF data from ANCE top-retrieved documents (on CPU): 
```angular2html
cd data_prep 
bash get_prf_data.sh
```

## Training 
```angular2html
bash train_encoder.sh
```

## Inference 
```
bash eval.sh
```

## Trained Models and Ranking Files 
Trained models and ranking files (for k=3) can be downloaded [here](https://drive.google.com/drive/folders/1FybKqWbE1Ap1xDd8MR01ZOqXn9W0Xy8b?usp=sharing). 


