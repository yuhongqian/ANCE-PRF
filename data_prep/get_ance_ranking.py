import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
import argparse
import numpy as np
import csv
import faiss
import pickle
from utils.util import * 
# from data_utils import * 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_data_dir", default=None, help="Path to <dataset>_preprocessed folder.")
    parser.add_argument("--ance_checkpoint_path", default=None, help="Path to <dataset>_output.")
    parser.add_argument("--dataset", default=None, help="Name of the dataset.")
    parser.add_argument("--mode", choices=["train", "dev"], required=True)
    return parser.parse_args()

def ance_full_rank(args, query_embedding, passage_embedding, topN=1000):
    dim = passage_embedding.shape[1]
    faiss.omp_set_num_threads(16)
    cpu_index = faiss.IndexFlatIP(dim)
    cpu_index.add(passage_embedding)
    print(f"Starting CPU search on {args.dataset} {args.mode} data...")
    _, I = cpu_index.search(query_embedding, topN)
    print(f"Finished CPU search on {args.dataset} {args.mode} data.")
    with open(os.path.join(args.ance_checkpoint_path, f"{args.dataset}_{args.mode}_I.npy"), "wb") as f:
        np.save(f, I)
    tein_path = os.path.join(args.ance_checkpoint_path, f"{args.dataset}_{args.mode}.tein")
    return I


if __name__ == '__main__':
    args = parse_args()
    query_embedding, query_embedding2id, passage_embedding, passage_embedding2id = load_embeddings(args, args.mode)
    ance_full_rank(args, query_embedding, passage_embedding)
    query_embedding2qid = get_embedding2qid(args)
    if args.mode == "dev": 
        query_embedding2qid = get_embedding2qid(args) 
        devI_path = os.path.join(args.ance_checkpoint_path, f"{args.dataset}_{args.mode}_I.npy")
        devI_to_tein(args, query_embedding2qid, devI_path)

    print(f"Initial ranking on {args.dataset} {args.mode} data done.")