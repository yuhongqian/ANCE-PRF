import os
import pickle
import numpy as np

# def load_embeddings(args, checkpoint=0):
#     split = args.split
#     passage_embedding = []
#     passage_embedding2id = []

#     with open(os.path.join(args.output_dir, f"query_embeds_split{split}.npy"), "rb") as f:
#         query_embedding = np.load(f, allow_pickle=True)
#     with open(os.path.join(args.output_dir, f"embedid2qid{split}.pkl"), "rb") as f:
#         query_embedid2qid = pickle.load(f)

#     for i in range(8):
#         try:
#             with open(os.path.join(args.output_dir, "passage_" + str(checkpoint) + "__emb_p__data_obj_" + str(i) + ".pb"), 'rb') as handle:
#                 passage_embedding.append(np.load(handle))
#             with open(os.path.join(args.output_dir, "passage_" + str(checkpoint) + "__embid_p__data_obj_" + str(i) + ".pb"), 'rb') as handle:
#                 passage_embedding2id.append(np.load(handle))
#         except:
#             print(f"Loaded {i} passage embedding splits.")
#             break

#     passage_embedding = np.concatenate(passage_embedding, axis=0)
#     passage_embedding2id = np.concatenate(passage_embedding2id, axis=0)
#     return query_embedding, query_embedid2qid, passage_embedding, passage_embedding2id


def offset_to_orig_id(orig2offset):
    offset2orig = dict()
    for k, v in orig2offset.items():
        offset2orig[v] = k
    return offset2orig