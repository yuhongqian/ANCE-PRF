import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import torch
from torch import nn
import torch.nn.functional as F
from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig

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
        # emb_all is a tuple from bert - sequence output, pooler (in old version only0
        # assert isinstance(emb_all, tuple)
        if self.use_mean:
            return self.masked_mean(emb_all[0], mask)
        else:
            return emb_all[0][:, 0]

    def query_emb(self, input_ids, attention_mask):
        raise NotImplementedError("Please Implement this method")

    def body_emb(self, input_ids, attention_mask):
        raise NotImplementedError("Please Implement this method")


class NLL(EmbeddingMixin):
    def forward(self, input_ids, attention_mask, pos_emb, neg_emb):
        # pdb.set_trace()
        q_embs = self.query_emb(input_ids, attention_mask)

        logit_matrix = torch.cat([(q_embs * pos_emb).sum(-1).unsqueeze(1),
                                  (q_embs * neg_emb).sum(-1).unsqueeze(1)], dim=1)  # [B, 2]
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

class MSMarcoConfig:
    def __init__(self, name, model, process_fn, use_mean=True, tokenizer_class=RobertaTokenizer, config_class=RobertaConfig):
        self.name = name
        self.process_fn = process_fn
        self.model_class = model
        self.use_mean = use_mean
        self.tokenizer_class = tokenizer_class
        self.config_class = config_class

