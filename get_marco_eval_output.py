import os
import csv
import pickle
import argparse
import numpy as np
from util import load_embedding_prefix, load_embeddings


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ance_checkpoint_path", default="/bos/tmp10/hongqiay/ance/marco_output/",
                        help="location for dumpped query and passage/document embeddings which is output_dir")
    parser.add_argument("--processed_data_dir", default="/bos/tmp10/hongqiay/ance/marco_preprocessed")
    parser.add_argument("--devI_path", required=True)

    return parser.parse_args()


def devI_to_tein(args, embedid2qid):
    _, passage_embedding2id = load_embedding_prefix(args.ance_checkpoint_path + "passage_0")

    with open(args.devI_path, "rb") as f:
        dev_I = np.load(f)
    tein_path = args.devI_path + ".tein"
    ance_ranking_to_tein(args, dev_I, passage_embedding2id, embedid2qid, tein_path)


def offset_to_orig_id(orig2offset):
    offset2orig = dict()
    for k, v in orig2offset.items():
        offset2orig[v] = k
    return offset2orig


# qid, pid = marco qid, pid
# offset = anceID (e.g. 0, 3, 6 ...)
# embed_id = id into the passage_embedding / query_embedding matrices (as the order it appear when it's given to faiss)
# In ANCE implementation, all evaluations are done on offset
def ance_ranking_to_tein(args, dev_I, passage_embedding2id, query_embedid2qid, output_path, top=10):
    """
    :param dev_I: each row consists of relevant passage embed_id for query embed_id
    :param passage_embedding2id: passage embed_id -> passage offset
    :param query_embedding2id: query embed_id -> query offset
    :param pid2offset: pid -> passage offset
    :return:
    """
    with open(os.path.join(args.processed_data_dir, "pid2offset.pickle"), "rb") as f:
        pid2offset = pickle.load(f)
    offset2pid = offset_to_orig_id(pid2offset)

    with open(output_path, "w") as f1, open(output_path+".marco", "w") as f2, \
            open(f"{output_path}.marco.{top}", "w") as f3:
        writer1 = csv.writer(f1, delimiter="\t")
        writer2 = csv.writer(f2, delimiter="\t")
        writer3 = csv.writer(f3, delimiter="\t")
        for qry_embed_id, rel_psg_embed_ids in enumerate(dev_I):
            qid = query_embedid2qid[qry_embed_id]
            pids = [offset2pid[passage_embedding2id[p_embed_id]] for p_embed_id in rel_psg_embed_ids]
            rows1 = [[qid, "Q0", pid, rank+1, -rank, "full_ance"] for rank, pid in enumerate(pids)]
            rows2 = [[qid, pid, rank+1] for rank, pid in enumerate(pids)]
            rows3 = [[qid, pid, rank+1] for rank, pid in enumerate(pids[:top])]
            writer1.writerows(rows1)
            writer2.writerows(rows2)
            writer3.writerows(rows3)


def get_offset2qid(args, mode="qid"):
    path = os.path.join(args.processed_data_dir, f"offset2{mode}.pickle")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    with open(os.path.join(args.processed_data_dir, f"{mode}2offset.pickle"), "rb") as f:
        qid2offset = pickle.load(f)
    offset2qid = {}
    for qid, offset in qid2offset.items():
        offset2qid[offset] = qid
    with open(os.path.join(args.processed_data_dir, f"offset2{mode}.pickle"), "wb") as f:
        pickle.dump(offset2qid, f)
    return offset2qid


def get_embedid2qid(args, offset2qid, mode="qid"):
    path = os.path.join(args.processed_data_dir, f"embedid2{mode}.pickle")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    if mode == "qid":
        _, embedid2offset = load_embedding_prefix(args.ance_checkpoint_path + "dev_query_0")
    else:
        _, embedid2offset = load_embedding_prefix(args.ance_checkpoint_path + "passage_0")

    embedid2qid = []
    for offset in embedid2offset:
        embedid2qid.append(offset2qid[offset])
    with open(path, "wb") as f:
        pickle.dump(embedid2qid, f)
    return embedid2qid


def main():
    args = parse_args()
    offset2qid = get_offset2qid(args)
    embedid2qid = get_embedid2qid(args, offset2qid)
    devI_to_tein(args, embedid2qid)


if __name__ == "__main__":
    main()
