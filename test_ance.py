import torch
from torch import nn
import torch.functional as F
from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig
import pdb


def pad_ids(input_ids, attention_mask, token_type_ids, max_length, pad_token, mask_padding_with_zero,
            pad_token_segment_id, pad_on_left=False):
    padding_length = max_length - len(input_ids)
    if pad_on_left:
        input_ids = ([pad_token] * padding_length) + input_ids
        attention_mask = ([0 if mask_padding_with_zero else 1]
                          * padding_length) + attention_mask
        token_type_ids = ([pad_token_segment_id] *
                          padding_length) + token_type_ids
    else:
        input_ids += [pad_token] * padding_length
        attention_mask += [0 if mask_padding_with_zero else 1] * padding_length
        token_type_ids += [pad_token_segment_id] * padding_length

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
                text.strip(), add_special_tokens=True, max_length=args.max_seq_length,)
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


class EmbeddingMixin:
    """
    Mixin for common functions in most embedding models. Each model should define its own bert-like backbone and forward.
    We inherit from RobertaModel to use from_pretrained
    """
    def __init__(self, model_argobj):
        if model_argobj is None:
            self.use_mean = False
        else:
            self.use_mean = model_argobj.use_mean
        print("Using mean:", self.use_mean)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def masked_mean(self, t, mask):
        s = torch.sum(t * mask.unsqueeze(-1).float(), axis=1)
        d = mask.sum(axis=1, keepdim=True).float()
        return s / d

    def masked_mean_or_first(self, emb_all, mask):
        # emb_all is a tuple from bert - sequence output, pooler
        assert isinstance(emb_all, tuple)
        if self.use_mean:
            return self.masked_mean(emb_all[0], mask)
        else:
            return emb_all[0][:, 0]

    def query_emb(self, input_ids, attention_mask):
        raise NotImplementedError("Please Implement this method")

    def body_emb(self, input_ids, attention_mask):
        raise NotImplementedError("Please Implement this method")


class NLL(EmbeddingMixin):
    def forward(
            self,
            query_ids,
            attention_mask_q,
            input_ids_a=None,
            attention_mask_a=None,
            input_ids_b=None,
            attention_mask_b=None,
            is_query=True):
        if input_ids_b is None and is_query:
            return self.query_emb(query_ids, attention_mask_q)
        elif input_ids_b is None:
            return self.body_emb(query_ids, attention_mask_q)
        # pdb.set_trace()
        q_embs = self.query_emb(query_ids, attention_mask_q)
        a_embs = self.body_emb(input_ids_a, attention_mask_a)
        b_embs = self.body_emb(input_ids_b, attention_mask_b)

        logit_matrix = torch.cat([(q_embs * a_embs).sum(-1).unsqueeze(1),
                                  (q_embs * b_embs).sum(-1).unsqueeze(1)], dim=1)  # [B, 2]
        lsm = F.log_softmax(logit_matrix, dim=1)
        loss = -1.0 * lsm[:, 0]
        return (loss.mean(),)


class RobertaDot_NLL_LN(NLL, RobertaForSequenceClassification):
    """None
    Compress embedding to 200d, then computes NLL loss.
    """

    def __init__(self, config, model_argobj=None):
        NLL.__init__(self, model_argobj)
        RobertaForSequenceClassification.__init__(self, config)
        self.embeddingHead = nn.Linear(config.hidden_size, 768)
        self.norm = nn.LayerNorm(768)
        self.apply(self._init_weights)

    def query_emb(self, input_ids, attention_mask):
        outputs1 = self.roberta(input_ids=input_ids,
                                attention_mask=attention_mask)
        full_emb = self.masked_mean_or_first(outputs1, attention_mask)
        query1 = self.norm(self.embeddingHead(full_emb))
        return query1

    def body_emb(self, input_ids, attention_mask):
        return self.query_emb(input_ids, attention_mask)


default_process_fn = triple_process_fn


class MSMarcoConfig:
    def __init__(self, name, model, process_fn=default_process_fn, use_mean=True, tokenizer_class=RobertaTokenizer,
                 config_class=RobertaConfig):
        self.name = name
        self.process_fn = process_fn
        self.model_class = model
        self.use_mean = use_mean
        self.tokenizer_class = tokenizer_class
        self.config_class = config_class


def test_loss():
    config_obj = MSMarcoConfig(name="rdot_nll", model=RobertaDot_NLL_LN, use_mean=False)
    config = config_obj.config_class.from_pretrained("roberta-base", num_labels=2)
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base", do_lower_case=False)
    model = RobertaDot_NLL_LN.from_pretrained("roberta-base", config=config)
    query = "Test query"
    doc1 = "Positive doc"
    doc2 = "Negative doc"
    query_tokens = tokenizer(query, return_tensors="pt", max_length=128, truncation=True)
    doc1_tokens = tokenizer(doc1, return_tensors="pt", max_length=128, truncation=True)
    doc2_tokens = tokenizer(doc2, return_tensors="pt", max_length=128, truncation=True)
    _ = model(query_ids=query_tokens["input_ids"], attention_mask_q=query_tokens["attention_mask"],
              input_ids_a=doc1_tokens["input_ids"], attention_mask_a=doc1_tokens["attention_mask"],
              input_ids_b=doc2_tokens["input_ids"], attention_mask_b=doc2_tokens["attention_mask"])


def main():
    test_loss()


if __name__ == '__main__':
    main()