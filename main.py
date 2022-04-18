import os
import random
import logging
import argparse
from data import *
from utils.util import * 
from torch.utils.data import DataLoader
from runner import Trainer, Evaluator


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--first_stage_inn_path", default=None)
    parser.add_argument("--eval_mode", choices=["rerank", "full"], default="rerank")
    parser.add_argument("--ann_chunk_factor", type=int, default=100)
    parser.add_argument("--ance_checkpoint_path", default=None,
                        help="location for dumpped query and passage/document embeddings which is output_dir")
    parser.add_argument("--train_data_dir",
                        default=None, help="Path to training data.")
    parser.add_argument("--dev_data_dir",
                        default=None, help="Path to dev data.")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--optimizer", choices=["lamb", "adamw"], default="lamb")
    parser.add_argument("--num_queries", type=int, default=502939)
    parser.add_argument("--num_feedbacks", type=int, default=3)
    parser.add_argument("--preprocessed_dir", default=None, help="Path to [dataset]_preprocessed folder.")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=50, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--max_steps", default=1000000, type=int,
                        help="If > 0: set total number of training steps to perform")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--model_name_or_path", default="roberta-base", type=str)
    parser.add_argument("--load_optimizer_scheduler", default=False, action="store_true",
                        help="load scheduler from checkpoint or not")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.")
    parser.add_argument("--logging_steps", type=int, default=100, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=2000, help="Save checkpoint every X updates steps.")
    parser.add_argument("--output_dir", default=None, help="Directory to save output.")
    parser.add_argument("--prev_evaluated_ckpt", type=int, default=0, help="Start evaluating ckpt after this step.")
    parser.add_argument("--dataset", choices=["marco", "marco_eval", "trec19psg", "trec20psg", "dlhard"], default="marco")
    parser.add_argument("--end_eval_ckpt", type=int, default=1000000, help="Start evaluating ckpt after this step.")

    return parser.parse_args()


def main():
    args = parse_args()
    if is_first_worker() and args.train:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "args"), "w") as f:
            f.write(str(args))
    set_env(args)
    tokenizer, model = load_model(args)

    passage_collection_path = os.path.join(args.preprocessed_dir, "passages")
    passage_cache = EmbeddingCache(passage_collection_path)

    if args.train and args.eval:
        raise ValueError("train and eval are supposed to be initiated as separate tasks. ")

    if args.train:
        query_collection_path = os.path.join(args.preprocessed_dir, "train-query")
        query_cache = EmbeddingCache(query_collection_path)
        all_lines = []
        for i in range(args.ann_chunk_factor):
            with open(os.path.join(args.train_data_dir, f"qry_encoder_training_data_{i}")) as f:
                all_lines.extend(f.readlines())
        with query_cache, passage_cache:
            train_dataset = StreamingDataset(args, all_lines,
                                             GetTripletTrainingDataProcessingFn(args, query_cache, passage_cache,
                                                                                num_feedbacks=args.num_feedbacks))
            args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
            train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size)
            trainer = Trainer(args, model, tokenizer, train_dataloader)
            trainer.train()

    if args.eval:
        rerank_depths = None if args.eval_mode == "full" else [20, 50, 100, 200, 500, 1000]
        query_collection_path = os.path.join(args.preprocessed_dir, "dev-query")
        query_cache = EmbeddingCache(query_collection_path)
        with open(os.path.join(args.dev_data_dir, f"qry_encoder_dev_data_full")) as f:
            all_lines = f.readlines()

        with query_cache, passage_cache:
            dev_dataset = StreamingDataset(args, all_lines,
                                           GetTripletTrainingDataProcessingFn(args, query_cache, passage_cache,
                                                                              num_feedbacks=args.num_feedbacks))
            args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
            dev_dataloader = DataLoader(dev_dataset, batch_size=args.eval_batch_size)
            with torch.no_grad():
                evaluator = Evaluator(args, dev_dataloader)
                evaluator.eval(rerank_depths=rerank_depths, mode=args.eval_mode)


if __name__ == '__main__':
    main()