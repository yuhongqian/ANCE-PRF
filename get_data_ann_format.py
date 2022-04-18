import os
import csv
import random
import argparse
import faiss
import logging
import pickle
import numpy as np
from utils.util import load_embeddings

import pdb

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_data_dir", default=None)
    parser.add_argument("--ann_chunk_factor", type=int, default=100)
    parser.add_argument("--negative_sample", type=int, default=20, help="Number of negative documents to sample.")
    parser.add_argument("--topk_training", type=int, default=200, help="Sampling depth for negative passages.")
    parser.add_argument("--num_feedbacks", type=int, default=20, help="Number of feedback documents to save.")
    parser.add_argument("--output_dir")
    parser.add_argument("--split", type=int)
    parser.add_argument("--dev_inn_path", default=None, help="Path to ANCE top rankings.")
    parser.add_argument("--generate_train", action="store_true", help="Generate PRF training data.") 
    parser.add_argument("--generate_dev", action="store_true", help="Generate PRF dev data.") 

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
def load_positive_ids(args, mode="train"):

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


def generate_train_data(args):
    output_num = args.split
    training_query_positive_id = load_positive_ids(args)

    query_embedding, query_embedding2id, passage_embedding, passage_embedding2id = load_embeddings(args)
    dim = passage_embedding.shape[1]
    print('passage embedding shape: ' + str(passage_embedding.shape))
    top_k = args.topk_training
    faiss.omp_set_num_threads(16)
    cpu_index = faiss.IndexFlatIP(dim)
    cpu_index.add(passage_embedding)
    print("added index")

    # Construct new training set 
    chunk_factor = args.ann_chunk_factor
    effective_idx = output_num % chunk_factor

    if chunk_factor <= 0:
        chunk_factor = 1
    num_queries = len(query_embedding)
    queries_per_chunk = num_queries // chunk_factor
    q_start_idx = queries_per_chunk * effective_idx
    q_end_idx = num_queries if (
        effective_idx == (
            chunk_factor -
            1)) else (
        q_start_idx +
        queries_per_chunk)
    query_embedding = query_embedding[q_start_idx:q_end_idx]
    query_embedding2id = query_embedding2id[q_start_idx:q_end_idx]

    print(
        "Chunked {} query from {}".format(
            len(query_embedding),
            num_queries))
    # I: [number of queries, topk]
    _, I = cpu_index.search(query_embedding, top_k)
    print("Saving inn...")
    inn_path = os.path.join(args.output_dir, f"inn_{output_num}.npy")
    with open(inn_path, "wb") as f:
        np.save(f, I)

    effective_q_id = set(query_embedding2id.flatten())
    query_negative_passage, ance_top_passage = generate_pids(
        args,
        query_embedding2id,
        passage_embedding2id,
        I,
        effective_q_id)

    print("***** Construct Query Encoder Training Data *****")
    train_data_output_path = os.path.join(
        args.output_dir, "qry_encoder_training_data_" + str(output_num))

    with open(train_data_output_path, 'w') as f:
        query_range = list(range(I.shape[0]))
        random.shuffle(query_range)
        for query_idx in query_range:
            query_id = query_embedding2id[query_idx]
            if query_id not in effective_q_id or query_id not in training_query_positive_id:
                continue
            pos_pid = training_query_positive_id[query_id]
            f.write(
                "{}\t{}\t{}\t{}\n".format(
                    query_id, pos_pid,
                    ','.join(str(feedback_pid) for feedback_pid in ance_top_passage[query_id]),
                    ','.join(str(neg_pid) for neg_pid in query_negative_passage[query_id])))


def generate_dev_data(args):
    dev_query_positive_id = load_positive_ids(args, mode="dev")
    query_embedding, query_embedding2id, passage_embedding, passage_embedding2id = load_embeddings(args, mode="dev")
    with open(args.dev_inn_path, "rb") as f:
        I = np.load(f)
    effective_q_id = set(query_embedding2id.flatten())
    query_negative_passage, ance_top_passage = generate_pids(args, query_embedding2id, passage_embedding2id, I, effective_q_id)
    with open(os.path.join(args.output_dir, "qry_encoder_dev_data_full"), "w") as f:
        query_range = list(range(I.shape[0]))
        for query_idx in query_range:
            query_id = query_embedding2id[query_idx]
            if query_id not in effective_q_id or query_id not in dev_query_positive_id:
                print(f"invalid qid {query_id}")
            pos_pid = dev_query_positive_id[query_id]
            f.write(
                "{}\t{}\t{}\t{}\n".format(
                    query_id, pos_pid,
                    ','.join(str(feedback_pid) for feedback_pid in ance_top_passage[query_id]),
                    ','.join(str(neg_pid) for neg_pid in query_negative_passage[query_id])))


def get_psg_embeds(args, mode="train"):
    _, _, passage_embedding, passage_embedding2id = load_embeddings(args, mode=mode)
    id2embedding = dict()
    for embed_id, id in enumerate(passage_embedding2id):
        id2embedding[id] = embed_id
    embeddings = []
    tsv_path = os.path.join(args.output_dir, f"qry_encoder_training_data_{args.split}") if mode == "train" \
        else os.path.join(args.output_dir, "qry_encoder_dev_data_full")

    with open(tsv_path, "r") as f:
        for l in f:
            line_arr = l.split("\t")
            pos_pid = id2embedding[int(line_arr[1])]
            neg_pids = line_arr[3].split(",")
            neg_pids = [id2embedding[int(neg_pid)] for neg_pid in neg_pids]
            all_pids = [pos_pid] + neg_pids
            embeddings.append(passage_embedding[all_pids])
    embeddings = np.array(embeddings)
    output_path = os.path.join(args.output_dir, f"psg_embeds_{args.split}") if mode == "train" \
        else os.path.join(args.output_dir, f"psg_embeds_dev")
    with open(output_path, "wb") as fout:
        pickle.dump(embeddings, fout)


def get_binary_qrel(qrel_tsv_path, bin_qrel_path):
    with open(qrel_tsv_path, "r") as fin, open(bin_qrel_path, "w") as fout:
        for l in fin:
            qid, pid, rel = l.split()
            rel = 0 if int(rel) < 2 else 1
            fout.write(f"{qid}\t{pid}\t{rel}\n")


if __name__ == '__main__':
    args = parse_args()
    if args.generate_train:
        generate_train_data(args)
    elif args.generaet_dev: 
        generate_dev_data(args)
