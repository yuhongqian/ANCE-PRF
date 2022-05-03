import time
import logging
from utils.util import *
from data import *
from lamb import Lamb
import torch
from tqdm import tqdm
from tensorboardX import SummaryWriter
from transformers import AdamW, get_linear_schedule_with_warmup
logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, args, model, tokenizer, train_dataloader):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader

    def train(self):
        args, model, tokenizer, train_dataloader = self.args, self.model, self.tokenizer, self.train_dataloader
        """ Train the model """
        logger.info("Training/evaluation parameters %s", args)
        tb_writer = None
        if is_first_worker():
            tb_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tb_logs"))

        real_batch_size = args.train_batch_size * args.gradient_accumulation_steps * \
            (torch.distributed.get_world_size() if args.local_rank != -1 else 1)

        optimizer_grouped_parameters = []
        layer_optim_params = set()
        for layer_name in [
            "roberta.embeddings",
            "score_out",
            "downsample1",
            "downsample2",
            "downsample3"]:
            layer = getattr_recursive(model, layer_name)
            if layer is not None:
                optimizer_grouped_parameters.append({"params": layer.parameters()})
                for p in layer.parameters():
                    layer_optim_params.add(p)
        if getattr_recursive(model, "roberta.encoder.layer") is not None:
            for layer in model.roberta.encoder.layer:
                optimizer_grouped_parameters.append({"params": layer.parameters()})
                for p in layer.parameters():
                    layer_optim_params.add(p)

        optimizer_grouped_parameters.append(
            {"params": [p for p in model.parameters() if p not in layer_optim_params]})

        if args.optimizer.lower() == "lamb":
            optimizer = Lamb(
                optimizer_grouped_parameters,
                lr=args.learning_rate,
                eps=args.adam_epsilon)
        elif args.optimizer.lower() == "adamw":
            optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=args.learning_rate,
                eps=args.adam_epsilon)
        else:
            raise Exception(
                "optimizer {0} not recognized! Can only be lamb or adamW".format(
                    args.optimizer))
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=args.max_steps)

        # Check if saved optimizer or scheduler states exist
        # Load in optimizer and scheduler states
        if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and args.load_optimizer_scheduler:
            optimizer.load_state_dict(
                torch.load(
                    os.path.join(
                        args.model_name_or_path,
                        "optimizer.pt")))
        if os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt")) and args.load_optimizer_scheduler:
            scheduler.load_state_dict(
                torch.load(
                    os.path.join(
                        args.model_name_or_path,
                        "scheduler.pt")))


        # multi-gpu training (should be after apex fp16 initialization)
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[
                    args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=True,
            )

        # Train
        logger.info("***** Running training *****")
        logger.info("  Max steps = %d", args.max_steps)
        logger.info(
            "  Instantaneous batch size per GPU = %d",
            args.per_gpu_train_batch_size)
        logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            args.train_batch_size
            * args.gradient_accumulation_steps
            * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
        )
        logger.info(
            "  Gradient Accumulation steps = %d",
            args.gradient_accumulation_steps)

        global_step = 0
        # Check if continuing training from a checkpoint
        if os.path.exists(args.model_name_or_path):
            # set global_step to gobal_step of last saved checkpoint from model
            # path
            if "-" in args.model_name_or_path:
                global_step = int(
                    args.model_name_or_path.split("-")[-1].split("/")[0])
            else:
                global_step = 0
            logger.info(
                "  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from global step %d", global_step)

        tr_loss = 0.0
        model.zero_grad()
        model.train()
        set_seed(args)  # Added here for reproducibility

        step = 0

        train_dataloader = self.train_dataloader
        train_dataloader_iter = iter(train_dataloader)


        while global_step < args.max_steps:
            # pdb.set_trace()
            try:
                batch = next(train_dataloader_iter)
            except StopIteration:
                logger.info("Finished iterating current dataset, begin reiterate")
                train_dataloader_iter = iter(train_dataloader)
                batch = next(train_dataloader_iter)

            batch = tuple(t.to(args.device) for t in batch)
            step += 1
            inputs = {
                "input_ids": batch[0].long(),
                "attention_mask": batch[1].long(),
                "pos_emb": batch[2].float(),
                "neg_emb": batch[3].float(),
            }
            # pdb.set_trace()
            # sync gradients only at gradient accumulation step
            if step % args.gradient_accumulation_steps == 0 or args.local_rank == -1:
                outputs = model(**inputs)
            else:
                with model.no_sync():
                    outputs = model(**inputs)
            # model outputs are always tuple in transformers (see doc)
            loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if step % args.gradient_accumulation_steps == 0 or args.local_rank == -1:
                loss.backward()
            else:
                with model.no_sync():
                    loss.backward()

            tr_loss += loss.item()
            if step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    loss_scalar = tr_loss / args.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar
                    tr_loss = 0

                    if is_first_worker():
                        for key, value in logs.items():
                            tb_writer.add_scalar(key, value, global_step)
                        logger.info(json.dumps({**logs, **{"step": global_step}}))

                if is_first_worker() and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(
                        args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    # make sure logging_steps < save_steps
                    torch.save(
                        {
                            "loss": loss_scalar,
                            "learning_rate": learning_rate_scalar
                        },
                        os.path.join(output_dir, "stats.pt")
                    )
                    logger.info(
                        "Saving optimizer and scheduler states to %s",
                        output_dir)

        if is_first_worker():
            tb_writer.close()

        return global_step


class Evaluator:
    def __init__(self, args, dev_dataloader):
        self.args = args
        self.dev_dataloader = dev_dataloader
        self.prev_evaluated_ckpt = 0 if args.prev_evaluated_ckpt is None else args.prev_evaluated_ckpt

    def eval(self, rerank_depths=None, mode="rerank", results=None):
        dataset = self.args.dataset
        _, query_embedding2id, passage_embedding, passage_embedding2id = load_embeddings(self.args, mode="dev")
        # query_embs, query_embedding2id, passage_embedding, passage_embedding2id = load_embeddings(self.args, mode="dev")
        dev_query_positive_id = load_positve_query_id(self.args)
        binary_dev_query_positive_id = None if dataset in {"marco", "marco_eval"} else load_positve_query_id(self.args, binary=True)

        topN = 1000
        if is_first_worker():
            # TODO: this is for current compatibility. change all names to f"eval_logs_{mode}_{dataset}"  later
            name = f"eval_logs_{mode}" if dataset == "marco" else f"eval_logs_{mode}_{dataset}"
            tb_writer = SummaryWriter(log_dir=os.path.join(self.args.output_dir, name))
        while self.prev_evaluated_ckpt < self.args.max_steps:
            curr_ckpt = self.prev_evaluated_ckpt + self.args.save_steps
            if curr_ckpt >= self.args.end_eval_ckpt:
                break
            output_dir = os.path.join(self.args.output_dir, f"checkpoint-{curr_ckpt}")
            while not os.path.exists(output_dir):
                logging.info(f"Waiting for step {curr_ckpt}")
                time.sleep(100)
            logging.info(f"Evaluating step {curr_ckpt}...")
            self.args.model_name_or_path = output_dir

            model = None
            while model is None:
                try:
                    _, model = load_model(self.args)
                except:
                    time.sleep(100)
                    pass
            model.eval()

            # query embedding inference
            dev_query_embs_path = os.path.join(output_dir, f"dev_query_embs_{dataset}.npy")
            # TODO: distributed eval currently not supported (currently not necessary considering the size of marco dev)
            loss = 0
            cnt = 0
            if os.path.exists(dev_query_embs_path):
                query_embs = np.load(dev_query_embs_path)
            else:
                query_embs = []
                for batch in tqdm(self.dev_dataloader, desc=f"Eval ckpt{curr_ckpt}"):
                    cnt += 1
                    batch = tuple(t.to(self.args.device) for t in batch)
                    inputs = {
                        "input_ids": batch[0].long(),
                        "attention_mask": batch[1].long(),
                        "pos_emb": batch[2].float(),
                        "neg_emb": batch[3].float(),
                    }
                    query_emb = model.query_emb(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
                    loss += model(**inputs)[0].item()
                    query_embs.append(query_emb.detach().cpu().numpy())
                # 20 is the number of negative documents per positive document
                query_embs = np.concatenate(query_embs)
                np.save(os.path.join(output_dir, f"dev_query_embs_{dataset}.npy"), query_embs)
            logs = dict()
            if mode == "full":
                dev_I = full_rank(query_embs, passage_embedding, output_dir, dataset)
                result = EvalDevQuery(query_embedding2id, passage_embedding2id, dev_query_positive_id, dev_I, topN)
                final_ndcg, eval_query_cnt, final_Map, final_mrr, final_recall, hole_rate, ms_mrr, Ahole_rate, \
                metrics, prediction = result
                logs = {
                    "full_ndcg_10": final_ndcg,
                    "full_map_10": final_Map,
                    "full_pytrec_mrr": final_mrr,
                    "full_ms_mrr": ms_mrr["MRR @10"],
                    f"full_recall_{topN}": final_recall,
                    f"hole_rate": hole_rate,
                    f"Ahole_rate": Ahole_rate
                }
                if dataset not in {"marco", "marco_eval"}:
                    binary_result = EvalDevQuery(query_embedding2id, passage_embedding2id, binary_dev_query_positive_id,
                                                 dev_I, topN)
                    _, _, _, _, binary_recall, _, binary_ms_mrr, _, _, _ = binary_result
                    logs[f"full_binary_recall_{topN}"] = binary_recall
                    logs["full_binary_ms_mrr"] = binary_ms_mrr["MRR @10"]

            elif mode == "rerank":
                first_stage_inn = np.load(self.args.first_stage_inn_path, allow_pickle=True)
                dev_I = rerank(first_stage_inn, query_embs, query_embedding2id, passage_embedding, passage_embedding2id,
                               output_dir, dataset)
                reranked_w_scores = []
                for inn in dev_I:
                    reranked_w_scores.append({pid: rank for (rank, pid) in enumerate(inn)})  # rank, id
                if rerank_depths is not None:
                    for depth in rerank_depths:
                        reranked_I = get_inn_rerank_depth(reranked_w_scores, first_stage_inn, depth)
                        result = EvalDevQuery(query_embedding2id, passage_embedding2id, dev_query_positive_id,
                                              reranked_I, topN)
                        final_ndcg, eval_query_cnt, final_Map, final_mrr, final_recall, hole_rate, ms_mrr, Ahole_rate, \
                        metrics, prediction = result
                        if loss > 0 and cnt > 0:
                            logs["loss"] = loss / cnt
                        logs[f"reranked_depth{depth}_ndcg_10"] = final_ndcg
                        logs[f"reranked_depth{depth}_map_10"] = final_Map
                        logs[f"reranked_depth{depth}_pytrec_mrr"] = final_mrr
                        logs[f"reranked_depth{depth}_ms_mrr"] = ms_mrr["MRR @10"]
                        logs["hole_rate"] = hole_rate
                        logs["Ahole_rate"] = Ahole_rate
                        if dataset not in {"marco", "marco_eval"}:
                            binary_result = EvalDevQuery(query_embedding2id, passage_embedding2id,
                                                         binary_dev_query_positive_id,
                                                         reranked_I, topN)
                            _, _, _, _, binary_recall, _, binary_ms_mrr, _, _, _ = binary_result
                            logs[f"rerank_depth{depth}_binary_recall_{topN}"] = binary_recall
                            logs[f"rerank_depth{depth}_binary_ms_mrr"] = binary_ms_mrr["MRR @10"]
            if results is not None:
                results.append(logs)
            if is_first_worker():
                logger.info(json.dumps({**logs, **{"step": curr_ckpt}}))
                for key, value in logs.items():
                    tb_writer.add_scalar(key, value, curr_ckpt)
            self.prev_evaluated_ckpt = curr_ckpt


