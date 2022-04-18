import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
import torch
import argparse
import numpy as np
import csv
import faiss
import json
import pickle
from model import RobertaDot_NLL_LN
from transformers import RobertaTokenizer, RobertaConfig
from tqdm import tqdm
from data_utils import * 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_dir", default=None, help="Path to MARCO raw data.")
    parser.add_argument("--output_dir", default=None, help="Path to save the embeddings.")
    parser.add_argument("--model_name_or_path", default=None, help="Path to the ANCE first-p model.")
    parser.add_argument("--split", type=int)
    parser.add_argument("--chunk_size", type=int, default=10000)

    return parser.parse_args()


def get_train_query_embeds(args):
    qid2query_path=os.path.join(f"{args.raw_data_dir}", "train.query2qid.json")
    split, chunk_size = args.split, args.chunk_size
    config = RobertaConfig.from_pretrained(args.model_name_or_path, num_labels=2, finetuning_task="MSMarco")
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=True)
    qry_encoder = RobertaDot_NLL_LN.from_pretrained(args.model_name_or_path,
                                                         from_tf=bool(".ckpt" in args.model_name_or_path),
                                                         config=config).eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    qry_encoder.to(device)
    qry_encoder = qry_encoder.query_emb

    idx = 0
    with open(qid2query_path, "r") as f:
        queries = json.load(f)
        query_embeds = dict()
        for query, qid in tqdm(queries.items(), desc="Encode Query"):
            if idx < split * chunk_size or idx >= (split + 1) * chunk_size:
                idx += 1
                continue
            tokenized = tokenizer.encode_plus(query, return_tensors="pt", max_length=512)
            query_input_ids = tokenized["input_ids"].to(device)
            attention_mask = tokenized["attention_mask"].to(device)
            query_embed = qry_encoder(input_ids=query_input_ids, attention_mask=attention_mask).detach().cpu().numpy()
            query_embeds[qid] = query_embed
            idx += 1
        with open(os.path.join(args.output_dir, f"query_embeds_split{split}.pkl"), "wb") as fout:
            pickle.dump(query_embeds, fout)


def get_np_query_embeds(args):
    embedid2qid = []
    split = args.split
    with open(os.path.join(args.output_dir, f"query_embeds_split{split}.pkl"), "rb") as f:
        data = pickle.load(f)
        embeds = []
        for k, v in data.items():
            embeds.append(v)
            embedid2qid.append(k)
        with open(os.path.join(args.output_dir, f"query_embeds_split{split}.npy"), "wb") as fnp:
            embeds = np.concatenate(embeds)
            np.save(fnp, embeds)
        with open(os.path.join(args.output_dir, f"embedid2qid{split}.pkl"), "wb") as fout:
            pickle.dump(embedid2qid, fout)

if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    get_train_query_embeds(args)
    get_np_query_embeds(args)
    print(f"Done generating train query embeddings for split {args.split}.")