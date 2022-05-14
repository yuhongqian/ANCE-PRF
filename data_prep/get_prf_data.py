import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import csv
import random
import argparse
import faiss
import logging
import pickle
import numpy as np
from utils.util import load_embeddings


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_data_dir", required=True)
    parser.add_argument("--negative_sample", type=int, default=20, help="Number of negative documents to sample.")
    parser.add_argument("--num_feedbacks", type=int, default=20, help="Number of feedback documents to save.")
    parser.add_argument("--output_dir")
    parser.add_argument("--inn_path", help="Path to the num", required=True)
    parser.add_argument("--ance_checkpoint_path", help="The path to the <dataset>_output directory", required=True)
    parser.add_argument("--mode", choices=["train", "dev"], required=True)
    parser.add_argument("--dataset", default=None, help="Name of the dataset.", required=True)
    parser.add_argument("--ann_chunk_factor", type=int, default=100, help="number of chunks to split the training data into.")

    return parser.parse_args()


def add_psgs(res_psgs, passage_embedding2id, query_id, selected_ann_idx, num_examples):
    cnt = 0
    rank = 0
    for idx in selected_ann_idx:
        pid = passage_embedding2id[idx]
        rank += 1

        if pid in res_psgs[query_id]:
            continue

        if cnt >= num_examples:
            break

        res_psgs[query_id].append(pid)
        cnt += 1


def generate_pids(args, query_embedding2id, passage_embedding2id, I_nearest_neighbor, effective_q_id):

    query_negative_passage = {}
    ance_top_passage = {}
    num_queries = 0

    for query_idx in range(I_nearest_neighbor.shape[0]):

        query_id = query_embedding2id[query_idx]

        if query_id not in effective_q_id:
            continue

        num_queries += 1

        top_ann_pid = I_nearest_neighbor[query_idx, :].copy()

        ance_top_passage[query_id] = []
        add_psgs(ance_top_passage, passage_embedding2id, query_id, top_ann_pid, args.num_feedbacks)

        # Randomly sample negative
        negative_sample_I_idx = list(range(I_nearest_neighbor.shape[1]))
        random.shuffle(negative_sample_I_idx)
        selected_ann_idx = top_ann_pid[negative_sample_I_idx]
        query_negative_passage[query_id] = []
        add_psgs(query_negative_passage, passage_embedding2id, query_id, selected_ann_idx, args.negative_sample)

    return query_negative_passage, ance_top_passage

# TODO: this function is problematic! dev data sometimes have more than one rel docs
def load_positive_ids(args):
    mode = args.mode
    logger.info("Loading query_2_pos_docid")
    query_positive_id = {}
    query_positive_id_path = os.path.join(args.processed_data_dir, f"{mode}-qrel.tsv")
    with open(query_positive_id_path, 'r', encoding='utf8') as f:
        tsvreader = csv.reader(f, delimiter="\t")
        for [topicid, docid, rel] in tsvreader:
            if int(rel) != 0:
            # assert rel == "1"
                topicid = int(topicid)
                docid = int(docid)
                query_positive_id[topicid] = docid

    return query_positive_id


def generate_data(args, query_embedding2id, passage_embedding2id):
    query_positive_id = load_positive_ids(args)
    with open(args.inn_path, "rb") as f:
        I = np.load(f)
    effective_q_id = set(query_embedding2id.flatten())
    query_negative_passage, ance_top_passage = generate_pids(args, query_embedding2id, passage_embedding2id, I, effective_q_id)
    with open(os.path.join(args.output_dir, f"prf_{args.mode}.tsv"), "w") as f:
        query_range = list(range(I.shape[0]))
        for query_idx in query_range:
            query_id = query_embedding2id[query_idx]
            if query_id not in effective_q_id or query_id not in query_positive_id:
                print(f"invalid qid {query_id}")
            pos_pid = query_positive_id[query_id]
            f.write(
                "{}\t{}\t{}\t{}\n".format(
                    query_id, pos_pid,
                    ','.join(str(feedback_pid) for feedback_pid in ance_top_passage[query_id]),
                    ','.join(str(neg_pid) for neg_pid in query_negative_passage[query_id])))


def get_psg_embeds(args, passage_embedding, passage_embedding2id):
    id2embedding = dict()
    for embed_id, id in enumerate(passage_embedding2id):
        id2embedding[id] = embed_id
    embeddings = []
    tsv_path = os.path.join(args.output_dir, f"prf_{args.mode}.tsv")

    with open(tsv_path, "r") as f:
        for l in f:
            line_arr = l.split("\t")
            pos_pid = id2embedding[int(line_arr[1])]
            neg_pids = line_arr[3].split(",")
            neg_pids = [id2embedding[int(neg_pid)] for neg_pid in neg_pids]
            all_pids = [pos_pid] + neg_pids
            embeddings.append(passage_embedding[all_pids])
    embeddings = np.array(embeddings)
    output_path = os.path.join(args.output_dir, f"psg_embeds_{args.mode}")
    with open(output_path, "wb") as fout:
        pickle.dump(embeddings, fout, protocol=4)
    return embeddings


def split_train_data(args, psg_embeds, num_queries):
    tsv_path = os.path.join(args.output_dir, f"prf_train.tsv")
    queries_per_chunk = num_queries // args.ann_chunk_factor
    with open(tsv_path, "r") as f: 
        for i in range(args.ann_chunk_factor): 
            output_tsv_path = os.path.join(args.output_dir, f"prf_train_{i}.tsv")
            output_embed_path = os.path.join(args.output_dir, f"psg_embeds_{args.mode}_{i}")
            q_start_idx = queries_per_chunk * i
            q_end_idx = num_queries if (
                i == (
                    args.ann_chunk_factor -
                    1)) else (
                q_start_idx +
                queries_per_chunk)
            with open(output_tsv_path, "w") as fout: 
                for _ in q_start_idx, q_end_idx: 
                    l = f.readline() 
                    fout.write(l.strip() + "\n")
            with open(output_embed_path, "wb") as fout: 
                pickle.dump(psg_embeds[q_start_idx:q_end_idx], fout, protocol=4)


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    query_embedding, query_embedding2id, passage_embedding, passage_embedding2id = load_embeddings(args, args.mode)
    generate_data(args, query_embedding2id, passage_embedding2id)
    psg_embeds = get_psg_embeds(args, passage_embedding, passage_embedding2id)
    if args.mode == "train": 
        num_queries = psg_embeds.shape[0]
        print(f"Splitting training data into {args.ann_chunk_factor} chunks. There are {num_queries} queries in total." )
        split_train_data(args, psg_embeds, num_queries)
    print(f"Generated PRF {args.mode} data for {args.dataset}")
