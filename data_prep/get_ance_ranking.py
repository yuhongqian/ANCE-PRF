import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
import argparse
import numpy as np
import csv
import faiss
import pickle
from data_utils import * 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_data_dir", default=None, help="Path to marco_preprocessed folder.")
    parser.add_argument("--output_dir", default=None, help="Path to save the embeddings.")
    parser.add_argument("--split", type=int)
    return parser.parse_args()


# qid, pid = marco qid, pid
# offset = anceID (e.g. 0, 3, 6 ...)
# embed_id = id into the passage_embedding / query_embedding matrices (as the order it appear when it's given to faiss)
# In ANCE implementation, all evaluations are done on offset
def ance_ranking_to_tein(I, passage_embedding2id, query_embedid2qid, output_path):
    """
    :param I: each row consists of relevant passage embed_id for query embed_id
    :param passage_embedding2id: passage embed_id -> passage offset
    :param query_embedding2id: query embed_id -> query offset
    :param pid2offset: pid -> passage offset
    :return:
    """
    with open(os.path.join(args.processed_data_dir, "pid2offset.pickle"), "rb") as f:
        pid2offset = pickle.load(f)
    offset2pid = offset_to_orig_id(pid2offset)

    with open(output_path, "w") as f:
        writer = csv.writer(f, delimiter="\t")
        for qry_embed_id, rel_psg_embed_ids in enumerate(I):
            qid = query_embedid2qid[qry_embed_id]
            pids = [offset2pid[passage_embedding2id[p_embed_id]] for p_embed_id in rel_psg_embed_ids]
            rows = [[qid, "Q0", pid, rank+1, -rank, "full_ance"] for rank, pid in enumerate(pids)]
            writer.writerows(rows)


def ance_full_rank(args, topN=1000):
    query_embedding, query_embedid2qid, passage_embedding, passage_embedding2id = load_embeddings(args)
    dim = passage_embedding.shape[1]
    faiss.omp_set_num_threads(16)
    cpu_index = faiss.IndexFlatIP(dim)
    cpu_index.add(passage_embedding)
    print(f"Start CPU search on split {args.split}...")
    _, train_I = cpu_index.search(query_embedding, topN)
    print(f"Finished CPU search on split {args.split}.")
    with open(os.path.join(args.output_dir, f"train_I{args.split}.npy"), "wb") as f:
        np.save(f, train_I)
    tein_path = os.path.join(args.output_dir, f"ranking_split{args.split}.tein")
    ance_ranking_to_tein(train_I, passage_embedding2id, query_embedid2qid, tein_path)
    return train_I


if __name__ == '__main__':
    args = parse_args()
    ance_full_rank(args)
    I_to_tein(args)
    print(f"Initial ranking on split {args.split} done.")
