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




def main():
    args = parse_args()
    offset2qid = get_offset2qid(args)
    embedid2qid = get_embedid2qid(args, offset2qid)
    devI_to_tein(args, embedid2qid)


if __name__ == "__main__":
    main()
