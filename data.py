import os
import json
import pickle
import numpy as np
import csv
import torch
from torch.utils.data import IterableDataset, TensorDataset
import torch.distributed as dist
import logging
import pdb
import random


class EmbeddingCache:
    def __init__(self, base_path, seed=-1):
        self.base_path = base_path
        with open(base_path + '_meta', 'r') as f:
            meta = json.load(f)
            self.dtype = np.dtype(meta['type'])
            self.total_number = meta['total_number']
            self.record_size = int(
                meta['embedding_size']) * self.dtype.itemsize + 4
        if seed >= 0:
            self.ix_array = np.random.RandomState(
                seed).permutation(self.total_number)
        else:
            self.ix_array = np.arange(self.total_number)
        self.f = None

    def open(self):
        self.f = open(self.base_path, 'rb')

    def close(self):
        self.f.close()

    def read_single_record(self):
        record_bytes = self.f.read(self.record_size)
        passage_len = int.from_bytes(record_bytes[:4], 'big')
        passage = np.frombuffer(record_bytes[4:], dtype=self.dtype)
        return passage_len, passage

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __getitem__(self, key):
        if key < 0 or key > self.total_number:
            raise IndexError(
                "Index {} is out of bound for cached embeddings of size {}".format(
                    key, self.total_number))
        self.f.seek(key * self.record_size)
        return self.read_single_record()

    def __iter__(self):
        self.f.seek(0)
        for i in range(self.total_number):
            new_ix = self.ix_array[i]
            yield self.__getitem__(new_ix)

    def __len__(self):
        return self.total_number


class StreamingDataset(IterableDataset):
    def __init__(self, args, elements, fn, distributed=True):
        super().__init__()
        self.elements = elements
        self.fn = fn
        self.num_replicas = -1
        self.distributed = distributed
        self.psg_embeds = None
        if args.train:
            with open(os.path.join(args.train_data_dir, "psg_embeds_0"), "rb") as f:
                self.psg_embeds = pickle.load(f)
        elif args.eval:
            with open(os.path.join(args.dev_data_dir, "psg_embeds_dev"), "rb") as f:
                self.psg_embeds = pickle.load(f)
        self.curr_split_idx = 0
        self.queries_per_chunk = args.num_queries // args.ann_chunk_factor
        self.args = args

    def load_psg_embeds(self, i):
        split_idx = (i // self.queries_per_chunk) % self.args.ann_chunk_factor
        if self.curr_split_idx != split_idx:
            logging.info(f"Training on split {split_idx}...")
            with open(os.path.join(self.args.train_data_dir, f"psg_embeds_{split_idx}"), "rb") as f:
                self.psg_embeds = pickle.load(f)
            self.curr_split_idx = split_idx

    def __iter__(self):
        if dist.is_initialized():
            self.num_replicas = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            print("Not running in distributed mode")
        for i, element in enumerate(self.elements):
            if self.args.train:
                self.load_psg_embeds(i)
            if self.distributed and self.num_replicas != -1 and i % self.num_replicas != self.rank:
                continue
            records = self.fn(self.psg_embeds, element, i)
            # Each file line corresponds to several examples (1 + # neg samples)
            for rec in records:
                yield rec


def GetProcessingFn(args):
    """
    Modified from ANCE's GetProcessingFn
    :param args:
    :return:
    """
    def fn(psg_emb, qry_len, qry, feedback_cache_items):
        # Get model input data
        sep_id = 2
        all_input_ids = [qry[:qry_len]]
        for feedback_len, feedback in feedback_cache_items:
            all_input_ids.append(feedback[1:feedback_len])
        all_input_ids = np.concatenate(all_input_ids)
        if len(all_input_ids) > args.max_seq_length:
            all_input_ids = np.append(all_input_ids[:args.max_seq_length - 1], all_input_ids[-1])    # -1 is CLS
            content_len = args.max_seq_length
        else:
            all_sep_idxs = [idx for idx in range(len(all_input_ids)) if all_input_ids[idx] == sep_id]
            content_len = all_sep_idxs[-1] + 1
        pad_len = args.max_seq_length - content_len
        attention_mask = [1] * content_len + [0] * pad_len
        all_input_ids = torch.tensor([list(all_input_ids) + [0] * pad_len], dtype=torch.int)
        attention_mask = torch.tensor([attention_mask], dtype=torch.bool)
        if psg_emb is not None:
            psg_emb = torch.tensor([psg_emb], dtype=torch.float32)

        dataset = TensorDataset(
            all_input_ids,
            attention_mask,
            psg_emb
        )
        return [ts for ts in dataset]

    return fn


def GetTripletTrainingDataProcessingFn(args, query_cache, passage_cache, num_feedbacks=3):
    def fn(ordered_psg_embeds, line, i):
        queries_per_chunk = args.num_queries // args.ann_chunk_factor
        effective_idx = i % queries_per_chunk
        psg_embeds = ordered_psg_embeds[effective_idx]

        line_arr = line.split('\t')
        qid = int(line_arr[0])
        pos_pid = int(line_arr[1])
        feedback_pids = line_arr[2].split(",")[:num_feedbacks]
        feedback_pids = [int(feedback_pid) for feedback_pid in feedback_pids]
        neg_pids = line_arr[3].split(',')
        neg_pids = [int(neg_pid) for neg_pid in neg_pids]


        qry_len, qry = query_cache[qid]
        feedback_cache_items = [passage_cache[feedback_pid] for feedback_pid in feedback_pids]

        pos_data = GetProcessingFn(args)(psg_embeds[0], qry_len, qry, feedback_cache_items)[0]

        if args.eval:
            neg_pids = neg_pids[:1]
        for i in range(len(neg_pids)):
            neg_psg_emb = psg_embeds[1+i]
            neg_psg_emb = torch.tensor(neg_psg_emb, dtype=torch.float32)
            res = [data for data in pos_data]
            res.append(neg_psg_emb)
            yield res

    return fn


# TODO: fix getprocessingfn
def attentionVisProcessingFn(args):
    sep_id = 2
    def fn(qry_len, qry, feedback_cache_items):
        all_input_ids = [qry[:qry_len]]
        for feedback_len, feedback in feedback_cache_items:
            all_input_ids.append(feedback[1:feedback_len])
        all_input_ids = np.concatenate(all_input_ids)
        if len(all_input_ids) > args.max_seq_length:
            all_input_ids = np.append(all_input_ids[:args.max_seq_length - 1], all_input_ids[-1])    # -1 is CLS
            all_sep_idxs = [idx for idx in range(len(all_input_ids)) if all_input_ids[idx] == sep_id]
            content_len = args.max_seq_length
        else:
            all_sep_idxs = [idx for idx in range(len(all_input_ids)) if all_input_ids[idx] == sep_id]
            content_len = all_sep_idxs[-1] + 1
        if len(all_sep_idxs) < (args.num_feedbacks + 1):
            all_sep_idxs += [all_sep_idxs[-1]] * (args.num_feedbacks + 1 - len(all_sep_idxs))
        all_sep_idxs = torch.tensor([all_sep_idxs], dtype=torch.int)
        pad_len = args.max_seq_length - content_len
        attention_mask = [1] * content_len + [0] * pad_len
        all_input_ids = torch.tensor([list(all_input_ids) + [0] * pad_len], dtype=torch.int)
        attention_mask = torch.tensor([attention_mask], dtype=torch.bool)

        dataset = TensorDataset(
            all_input_ids,
            attention_mask,
            all_sep_idxs
        )
        return [ts for ts in dataset]

    return fn


def GetAttentionVisProcessingFn(args, query_cache, passage_cache, num_feedbacks=3):
    def fn(ordered_psg_embeds, line, i):
        line_arr = line.split('\t')
        qid = int(line_arr[0])
        pos_pid = int(line_arr[1])
        feedback_pids = line_arr[2].split(",")[:num_feedbacks]
        feedback_pids = [int(feedback_pid) for feedback_pid in feedback_pids]

        qry_len, qry = query_cache[qid]
        feedback_cache_items = [passage_cache[feedback_pid] for feedback_pid in feedback_pids]

        pos_data = attentionVisProcessingFn(args)(qry_len, qry, feedback_cache_items)[0]

        if pos_pid in feedback_pids:
            pos_idx = feedback_pids.index(pos_pid)
        else:
            pos_idx = -1

        yield [data for data in pos_data] + [pos_idx] + [qid]

    return fn


def GetDotProductProcessingFn(args, query_cache, passage_cache, ance_query_embedding, ance_qid2embedid,
                              ance_passage_embedding, ance_pid2embedid,
                              num_feedbacks=3):
    def fn(ordered_psg_embeds, line, i):
        queries_per_chunk = args.num_queries // args.ann_chunk_factor
        effective_idx = i % queries_per_chunk
        psg_embeds = ordered_psg_embeds[effective_idx]
        line_arr = line.split('\t')
        qid = int(line_arr[0])
        pos_pid = int(line_arr[1])
        feedback_pids = line_arr[2].split(",")[:num_feedbacks]
        feedback_pids = [int(feedback_pid) for feedback_pid in feedback_pids]
        included = pos_pid in feedback_pids
        if included:
            random.shuffle(feedback_pids)
            pos_idx = feedback_pids.index(pos_pid)
        neg_pids = line_arr[3].split(',')
        neg_pids = [int(neg_pid) for neg_pid in neg_pids]

        ance_qry_embed = torch.tensor([ance_query_embedding[ance_qid2embedid[qid]]], dtype=torch.float32)
        pos_qry_embed = torch.tensor([psg_embeds[0]], dtype=torch.float32)
        neg_psg_embeds = [neg_psg_emb for neg_psg_emb in psg_embeds[1:]]
        neg_psg_embeds = torch.tensor([neg_psg_embeds], dtype=torch.float32)
        feedback_embeds = [ance_passage_embedding[ance_pid2embedid[feedback_pid]] for feedback_pid in feedback_pids]
        feedback_embeds = torch.tensor([feedback_embeds], dtype=torch.float32)

        qry_len, qry = query_cache[qid]
        if included:
            feedback_cache_items = [passage_cache[feedback_pid] for feedback_pid in feedback_pids]
        else:
            pos_idx = random.randrange(3)
            feedback_cache_items = [passage_cache[pos_pid] if i == pos_idx else passage_cache[feedback_pid]
                                    for i, feedback_pid in enumerate(feedback_pids)]
        pos_idx = torch.tensor([pos_idx], dtype=torch.int)

        pos_data = GetProcessingFn(args)(psg_embeds[0], qry_len, qry, feedback_cache_items)[0]
        res = [data for data in pos_data[:-1]]
        res += [ance_qry_embed, pos_qry_embed, neg_psg_embeds, feedback_embeds, pos_idx]

        yield res   # (all_input_ids, attention_mask, ance_qry_emb, pos_emb, neg_embs, feedback_embeds)

    return fn



def GetTsneDotProductrocessingFn(args, query_cache, passage_cache, ance_query_embedding, ance_qid2embedid,
                              ance_passage_embedding, ance_pid2embedid,
                              num_feedbacks=3):
    def fn(ordered_psg_embeds, line, i):
        queries_per_chunk = args.num_queries // args.ann_chunk_factor
        effective_idx = i % queries_per_chunk
        psg_embeds = ordered_psg_embeds[effective_idx]
        line_arr = line.split('\t')
        qid = int(line_arr[0])
        pos_pid = int(line_arr[1])
        feedback_pids = line_arr[2].split(",")[:num_feedbacks]
        feedback_pids = [int(feedback_pid) for feedback_pid in feedback_pids]
        neg_pids = line_arr[3].split(',')
        neg_pids = [int(neg_pid) for neg_pid in neg_pids]

        curr_qid = torch.tensor([qid], dtype=torch.int)
        ance_qry_embed = torch.tensor([ance_query_embedding[ance_qid2embedid[qid]]], dtype=torch.float32)
        pos_qry_embed = torch.tensor([psg_embeds[0]], dtype=torch.float32)
        neg_psg_embeds = [neg_psg_emb for neg_psg_emb in psg_embeds[1:]]
        neg_psg_embeds = torch.tensor([neg_psg_embeds], dtype=torch.float32)
        feedback_embeds = [ance_passage_embedding[ance_pid2embedid[feedback_pid]] for feedback_pid in feedback_pids]
        feedback_embeds = torch.tensor([feedback_embeds], dtype=torch.float32)

        qry_len, qry = query_cache[qid]
        feedback_cache_items = [passage_cache[feedback_pid] for feedback_pid in feedback_pids]

        pos_data = GetProcessingFn(args)(psg_embeds[0], qry_len, qry, feedback_cache_items)[0]
        res = [data for data in pos_data[:-1]]
        res += [ance_qry_embed, pos_qry_embed, neg_psg_embeds, feedback_embeds, curr_qid]

        yield res   # (all_input_ids, attention_mask, ance_qry_emb, pos_emb, neg_embs, feedback_embeds)

    return fn


def GetTripletDevDataProcessingFn(args, query_cache, passage_cache, num_feedbacks=3):
    def fn(ordered_psg_embeds, line, i):

        line_arr = line.split('\t')
        qid = int(line_arr[0])
        feedback_pids = line_arr[1].split(",")[:num_feedbacks]
        feedback_pids = [int(feedback_pid) for feedback_pid in feedback_pids]
        qry_len, qry = query_cache[qid]
        feedback_cache_items = [passage_cache[feedback_pid] for feedback_pid in feedback_pids]

        data = GetProcessingFn(args)(None, qry_len, qry, feedback_cache_items)[0]
        yield data

    return fn