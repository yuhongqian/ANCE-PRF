import os
import argparse
import csv
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_dir", default=None, help="Path to MARCO raw data.")
    return parser.parse_args()


def add_train_query_ids(args): 
    in_path = os.path.join(f"{args.raw_data_dir}", "triples.train.small.tsv")
    out_path = os.path.join(f"{args.raw_data_dir}", "id.triples.train.small.tsv")
    
    query2qid = dict()
    with open(in_path, "r") as fin:
        reader = csv.reader(fin, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        print("Reading training queries...")
        for row in reader:
            query = row[0]
            if query not in query2qid:
                query2qid[query] = len(query2qid)
        fin.seek(0)
        print("Writing qids to CSV format...")
        with open(out_path, "w") as fout:
            writer = csv.writer(fout, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
            for row in reader:
                query = row[0]
                writer.writerow([query2qid[query]] + row)
    print("Dumping to json...")
    with open(os.path.join(f"{args.raw_data_dir}", "train.query2qid.json"), "w") as f:
        json.dump(query2qid, f)

if __name__ == '__main__':
    args = parse_args()
    add_train_query_ids(args)