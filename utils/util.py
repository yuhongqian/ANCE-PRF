import sys
import os
from os import listdir
from os.path import isfile, join
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from sklearn.metrics import roc_curve, auc
import gzip
import copy
import torch
from torch import nn
import torch.distributed as dist
from tqdm import tqdm, trange
import json
import logging
import random
import pytrec_eval
import pickle
import numpy as np
import torch

torch.multiprocessing.set_sharing_strategy('file_system')
from multiprocessing import Process
from torch.utils.data import DataLoader, Dataset, TensorDataset, IterableDataset
import re
from model import MSMarcoConfig, RobertaDot_NLL_LN

logger = logging.getLogger(__name__)


def load_embedding_prefix(prefix):
    embedding = []
    embedding2id = []
    for i in range(8):
        try:
            with open(prefix + "__emb_p__data_obj_" + str(i) + ".pb",
                      "rb") as handle:
                embedding.append(np.load(handle, allow_pickle=True))
            with open(prefix + "__embid_p__data_obj_" + str(i) + ".pb",
                      "rb") as handle:
                embedding2id.append(np.load(handle, allow_pickle=True))
        except Exception as e:
            print(e)
            break
    embedding = np.concatenate(embedding, axis=0)
    embedding2id = np.concatenate(embedding2id, axis=0)
    return embedding, embedding2id


def load_embeddings(args, mode="train", checkpoint=0):

    if mode == "train":
        query_prefix = "query_"
    elif mode == "dev":
        query_prefix = "dev_query_"

    query_embedding, query_embedding2id = load_embedding_prefix(os.path.join(args.ance_checkpoint_path, query_prefix + str(checkpoint)))
    passage_embedding, passage_embedding2id = load_embedding_prefix(os.path.join(args.ance_checkpoint_path, "passage_" + str(checkpoint)))

    return query_embedding, query_embedding2id, passage_embedding, passage_embedding2id
    

class InputFeaturesPair(object):
    """
    A single set of features of data.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    def __init__(
            self,
            input_ids_a,
            attention_mask_a=None,
            token_type_ids_a=None,
            input_ids_b=None,
            attention_mask_b=None,
            token_type_ids_b=None,
            label=None):
        self.input_ids_a = input_ids_a
        self.attention_mask_a = attention_mask_a
        self.token_type_ids_a = token_type_ids_a

        self.input_ids_b = input_ids_b
        self.attention_mask_b = attention_mask_b
        self.token_type_ids_b = token_type_ids_b

        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def getattr_recursive(obj, name):
    for layer in name.split("."):
        if hasattr(obj, layer):
            obj = getattr(obj, layer)
        else:
            return None
    return obj


def barrier_array_merge(
        args,
        data_array,
        prefix="",
        load_cache=False):
    # data array: [B, any dimension]
    # merge alone one axis

    if args.local_rank == -1:
        return data_array

    if not load_cache:
        rank = args.rank

        pickle_path = os.path.join(
            args.output_dir,
            "{1}_data_obj_{0}.pb".format(
                str(rank),
                prefix))

        with open(pickle_path, 'wb') as handle:
            np.save(handle, data_array)


def pad_input_ids(input_ids, max_length,
                  pad_on_left=False,
                  pad_token=0):
    padding_length = max_length - len(input_ids)
    padding_id = [pad_token] * padding_length

    if padding_length <= 0:
        input_ids = input_ids[:max_length]
    else:
        if pad_on_left:
            input_ids = padding_id + input_ids
        else:
            input_ids = input_ids + padding_id

    return input_ids

def pad_ids(input_ids, attention_mask, token_type_ids, max_length,
            pad_on_left=False,
            pad_token=0,
            pad_token_segment_id=0,
            mask_padding_with_zero=True):
    padding_length = max_length - len(input_ids)
    padding_id = [pad_token] * padding_length
    padding_type = [pad_token_segment_id] * padding_length
    padding_attention = [0 if mask_padding_with_zero else 1] * padding_length

    if padding_length <= 0:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        token_type_ids = token_type_ids[:max_length]
    else:
        if pad_on_left:
            input_ids = padding_id + input_ids
            attention_mask = padding_attention + attention_mask
            token_type_ids = padding_type + token_type_ids
        else:
            input_ids = input_ids + padding_id
            attention_mask = attention_mask + padding_attention
            token_type_ids = token_type_ids + padding_type

    return input_ids, attention_mask, token_type_ids


def triple_process_fn(line, i, tokenizer, args):
    features = []
    cells = line.split("\t")
    if len(cells) == 3:
        # this is for training and validation
        # query, positive_passage, negative_passage = line
        mask_padding_with_zero = True
        pad_token_segment_id = 0
        pad_on_left = False

        for text in cells:
            input_id_a = tokenizer.encode(
                text.strip(), add_special_tokens=True, max_length=args.max_seq_length, )
            token_type_ids_a = [0] * len(input_id_a)
            attention_mask_a = [
                                   1 if mask_padding_with_zero else 0] * len(input_id_a)
            input_id_a, attention_mask_a, token_type_ids_a = pad_ids(
                input_id_a, attention_mask_a, token_type_ids_a, args.max_seq_length, tokenizer.pad_token_id,
                mask_padding_with_zero, pad_token_segment_id, pad_on_left)
            features += [torch.tensor(input_id_a, dtype=torch.int),
                         torch.tensor(attention_mask_a, dtype=torch.bool)]
    else:
        raise Exception(
            "Line doesn't have correct length: {0}. Expected 3.".format(str(len(cells))))
    return [features]


# to reuse pytrec_eval, id must be string
def convert_to_string_id(result_dict):
    string_id_dict = {}

    # format [string, dict[string, val]]
    for k, v in result_dict.items():
        _temp_v = {}
        for inner_k, inner_v in v.items():
            _temp_v[str(inner_k)] = inner_v

        string_id_dict[str(k)] = _temp_v

    return string_id_dict


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def set_env(args):
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
    )

    # Set seed
    set_seed(args)


def load_model(args, output_attentions=False):
    # Prepare GLUE task
    args.output_mode = "classification"
    label_list = ["0", "1"]
    num_labels = len(label_list)

    # store args
    if args.local_rank != -1:
        args.world_size = torch.distributed.get_world_size()
        args.rank = dist.get_rank()

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will
        # download model & vocab
        torch.distributed.barrier()

    config = RobertaConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=args.cache_dir if args.cache_dir else None,
        output_attentions=output_attentions
    )
    tokenizer = RobertaTokenizer.from_pretrained(
        args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
        use_fast=True
    )
    model = RobertaDot_NLL_LN.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will
        # download model & vocab
        torch.distributed.barrier()

    model.to(args.device)

    return tokenizer, model

    
def is_first_worker():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def concat_key(all_list, key, axis=0):
    return np.concatenate([ele[key] for ele in all_list], axis=axis)


def get_checkpoint_no(checkpoint_path):
    nums = re.findall(r'\d+', checkpoint_path)
    return int(nums[-1]) if len(nums) > 0 else 0


def get_latest_ann_data(ann_data_path):
    ANN_PREFIX = "ann_ndcg_"
    if not os.path.exists(ann_data_path):
        return -1, None, None
    files = list(next(os.walk(ann_data_path))[2])
    num_start_pos = len(ANN_PREFIX)
    data_no_list = [int(s[num_start_pos:])
                    for s in files if s[:num_start_pos] == ANN_PREFIX]
    if len(data_no_list) > 0:
        data_no = max(data_no_list)
        with open(os.path.join(ann_data_path, ANN_PREFIX + str(data_no)), 'r') as f:
            ndcg_json = json.load(f)
        return data_no, os.path.join(
            ann_data_path, "ann_training_data_" + str(data_no)), ndcg_json
    return -1, None, None


def numbered_byte_file_generator(base_path, file_no, record_size):
    for i in range(file_no):
        with open('{}_split{}'.format(base_path, i), 'rb') as f:
            while True:
                b = f.read(record_size)
                if not b:
                    # eof
                    break
                yield b


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
    def __init__(self, elements, fn, distributed=True):
        super().__init__()
        self.elements = elements
        self.fn = fn
        self.num_replicas = -1
        self.distributed = distributed

    def __iter__(self):
        if dist.is_initialized():
            self.num_replicas = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            print("Not running in distributed mode")
        for i, element in enumerate(self.elements):
            if self.distributed and self.num_replicas != -1 and i % self.num_replicas != self.rank:
                continue
            records = self.fn(element, i)
            for rec in records:
                yield rec


def tokenize_to_file(args, i, num_process, in_path, out_path, line_fn):
    configObj = MSMarcoConfig(name="rdot_nll", model=RobertaDot_NLL_LN, process_fn=triple_process_fn, use_mean=False)
    tokenizer = configObj.tokenizer_class.from_pretrained(
        args.model_name_or_path,
        do_lower_case=True,
        cache_dir=None,
    )

    with open(in_path, 'r', encoding='utf-8') if in_path[-2:] != "gz" else gzip.open(in_path, 'rt',
                                                                                     encoding='utf8') as in_f, \
            open('{}_split{}'.format(out_path, i), 'wb') as out_f:
        for idx, line in enumerate(in_f):
            if idx % num_process != i:
                continue
            out_f.write(line_fn(args, line, tokenizer))


def multi_file_process(args, num_process, in_path, out_path, line_fn):
    processes = []
    for i in range(num_process):
        p = Process(
            target=tokenize_to_file,
            args=(
                args,
                i,
                num_process,
                in_path,
                out_path,
                line_fn,
            ))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return [data]

    world_size = dist.get_world_size()
    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.LongTensor([tensor.numel()]).to("cuda")
    size_list = [torch.LongTensor([0]).to("cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,)).to("cuda"))
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).to("cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list