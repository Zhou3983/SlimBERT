import torch.nn as nn
from model.SlimLayer import *
import torch.nn.functional as F
import numpy as np
import math


class LayerNorm(nn.Module):
    def __init__(self, config, variance_epsilon=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(config.dim))
        self.beta = nn.Parameter(torch.zeros(config.dim))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        # normalize and expand
        mean = x.mean(-1, keepdim=True)
        var = (x-mean).pow(2).mean(-1, keepdim=True)
        x = (x-mean)/torch.sqrt(var+self.variance_epsilon)
        return self.gamma * x + self.beta


class Embedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_emb = nn.Embedding(config.vocab_size, config.dim)
        self.pos_emb = nn.Embedding(512, config.dim)
        self.seg_emb = nn.Embedding(config.n_segments, config.dim)
        if config.freeze_embeddings:
            self.tok_emb.weight.requires_grad = False
            self.pos_emb.weight.requires_grad = False
            self.seg_emb.weight.requires_grad = False
        self.LN = LayerNorm(config)
        self.drop_out = nn.Dropout(p=config.drop_out)

    def forward(self, x, seg):
        seq_len = x.size(1)
        # get pos index of x
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand_as(x)
        embedding = self.tok_emb(x) + self.pos_emb(pos) + self.seg_emb(seg)
        return self.drop_out(self.LN(embedding))


class MultiheadSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.q = nn.Linear(config.dim, config.dim)
        self.k = nn.Linear(config.dim, config.dim)
        self.v = nn.Linear(config.dim, config.dim)
        self.drop_out = nn.Dropout(p=config.drop_out)
        self.n_heads = config.n_heads

    def forward(self, x, mask):
        q, k, v = self.q(x), self.k(x), self.v(x)
        reshaped_size = [self.n_heads, int(x.size(-1)/self.n_heads)]
        # transfer into batch size, head no., seq length, head dim
        q, k, v = (x.view(*x.size()[:-1], *reshaped_size).transpose(1, 2)
                   for x in [q, k, v])
        # attention_score batch size, head no., seq length, seq length
        attention_score = torch.matmul(q,k.transpose(2,3)) /np.sqrt(k.size(-1))
        if mask is not None:
            mask = mask[:, None, None, :].float()
            attention_score -= 10000.0 * (1.0 - mask)
        # attention_score batch size, head no., seq length, seq length
        attention_score = self.drop_out(F.softmax(attention_score, -1))
        # attention_score batch size, head no., seq length, head dim
        attention_score = torch.matmul(attention_score,v).transpose(1, 2).contiguous()
        # attention_score batch size, seq length, dim
        return attention_score.view(*attention_score.size()[:2], -1)


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.dim, config.n_hidden)
        self.fc2 = nn.Linear(config.n_hidden, config.dim)
        self.ib1 = SlimLayer(dim=config.dim, masking=config.masking)
        self.ib2 = SlimLayer(dim=config.n_hidden, masking=config.masking)

    @staticmethod
    def gelu(x):
        "Implementation of the gelu activation function by Hugging Face"
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def forward(self, x):
        return self.fc2(self.ib2(self.gelu(self.fc1(self.ib1(x)))))


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiheadSelfAttention(config)
        self.ib_out = SlimLayer(dim=config.dim, masking=config.masking)
        self.ib_attention = SlimLayer(dim=config.dim, masking=config.masking)
        self.LN1 = LayerNorm(config)
        self.LN2 = LayerNorm(config)
        self.FF = FeedForward(config)
        self.drop_out = nn.Dropout(p=config.drop_out)
        self.dense = nn.Linear(config.dim, config.dim)

    def forward(self, x, mask):
        attention_x = self.attention(x, mask)
        x = self.LN1(x + self.drop_out(self.dense(self.ib_attention(attention_x))))
        x = self.LN2(x + self.drop_out(self.FF(x)))
        x = self.ib_out(x)
        return x


class BottledBERT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = Embedding(config)
        self.bottledBERT = nn.ModuleList([Encoder(config) for _ in range(config.n_layers)])
        if config.prune_emb:
            self.ib_emb = SlimLayer(dim=config.dim, masking=config.masking)

    def forward(self, x, segment, mask, prune_emb):
        x = self.embedding(x, segment)
        if prune_emb:
            x = self.ib_emb(x)
        for encoder in self.bottledBERT:
            x = encoder(x, mask)
        return x

    def get_prune_threshold_by_remaining_percentage(self, percentage, prune_emb):
        if percentage == 1:
            return -10
        all_prune_factor = []
        for block in self.bottledBERT:
            all_prune_factor += block.ib_out.alpha.data.cpu().tolist()
            all_prune_factor += block.ib_attention.alpha.data.cpu().tolist()
            all_prune_factor += block.FF.ib1.alpha.data.cpu().tolist()
            all_prune_factor += block.FF.ib2.alpha.data.cpu().tolist()
        if prune_emb:
            all_prune_factor += self.ib_emb.alpha.data.cpu().tolist()

        abs_all_prune_factor = [abs(ele) for ele in all_prune_factor]
        sorted_abs_all_prune_factor = sorted(abs_all_prune_factor)
        prune_index = int(len(sorted_abs_all_prune_factor)*(1-percentage))

        return (sorted_abs_all_prune_factor[prune_index]+sorted_abs_all_prune_factor[prune_index+1])/2

    def compute_slim_loss(self, config):
        ib_kld = 0
        for block in self.bottledBERT:
            if config.prune_attn:
                ib_kld += block.ib_out.kld
                ib_kld += block.ib_attention.kld
            if config.prune_ff:
                ib_kld += block.FF.ib1.kld
                ib_kld += block.FF.ib2.kld
        if config.prune_emb:
            ib_kld += self.ib_emb.kld*config.emb_factor
        return ib_kld

    def get_attention_head_pruned_percentage(self, threshold=0):
        ib_attention_masks = [block.ib_attention.get_mask_hard(threshold) for block in self.bottledBERT]
        head_size = 768/12
        all_pruned_percentage = []
        for i in range(12):
            current_attention_layer = ib_attention_masks[i]
            cur_pruned_percentage_list = []
            splited_current_attention_layer = torch.tensor_split(current_attention_layer, 12)
            for item in splited_current_attention_layer:
                pruned_neurons_by_head_sum = np.sum(item.cpu().numpy() == 0)
                pruned_percentage = pruned_neurons_by_head_sum/head_size
                cur_pruned_percentage_list.append(pruned_percentage)
            all_pruned_percentage.append(cur_pruned_percentage_list)
        return all_pruned_percentage

    def get_pruned_neurons_by_layer(self, threshold=0):
        ib_out_masks = [block.ib_out.get_mask_hard(threshold) for block in self.bottledBERT]
        ib_attention_masks = [block.ib_attention.get_mask_hard(threshold) for block in self.bottledBERT]
        ff_ib1_masks = [block.FF.ib1.get_mask_hard(threshold) for block in self.bottledBERT]
        ff_ib2_masks = [block.FF.ib2.get_mask_hard(threshold) for block in self.bottledBERT]

        percentage_list = []
        for i in range(12):
            pruned_neurons_attention = np.sum(ib_attention_masks[i].cpu().numpy() == 0)
            pruned_neurons_ff1 = np.sum(ff_ib1_masks[i].cpu().numpy() == 0)
            pruned_neurons_ff2 = np.sum(ff_ib2_masks[i].cpu().numpy() == 0)
            pruned_neurons_out = np.sum(ib_out_masks[i].cpu().numpy() == 0)
            pruned_percentage = (pruned_neurons_attention+pruned_neurons_ff1+pruned_neurons_ff2+pruned_neurons_out)\
                                   /(768*7)
            percentage_list.append(pruned_percentage)
        return percentage_list

    def get_hard_masks(self, prune_emb, threshold=0.001):
        ib_out_masks = [block.ib_out.get_mask_hard(threshold) for block in self.bottledBERT]
        ib_attention_masks = [block.ib_attention.get_mask_hard(threshold) for block in self.bottledBERT]
        ff_ib1_masks = [block.FF.ib1.get_mask_hard(threshold) for block in self.bottledBERT]
        ff_ib2_masks = [block.FF.ib2.get_mask_hard(threshold) for block in self.bottledBERT]
        if prune_emb:
            emb_mask = self.ib_emb.get_mask_hard(threshold)
            # print(f"ib emb mask: {emb_mask}")
            remaining_neurons_emb = np.sum(emb_mask.cpu().numpy() == 1)
            remaining_neurons = [remaining_neurons_emb]
        else:
            remaining_neurons = [768]
        for i in range(12):
            remaining_neurons_attention = np.sum(ib_attention_masks[i].cpu().numpy() == 1)
            remaining_neurons.append(remaining_neurons_attention)
            remaining_neurons_ff1 = np.sum(ff_ib1_masks[i].cpu().numpy() == 1)
            remaining_neurons.append(remaining_neurons_ff1)
            remaining_neurons_ff2 = np.sum(ff_ib2_masks[i].cpu().numpy() == 1)
            remaining_neurons.append(remaining_neurons_ff2)
            remaining_neurons_out = np.sum(ib_out_masks[i].cpu().numpy() == 1)
            remaining_neurons.append(remaining_neurons_out)

        remaining_params = 0
        for i in range(12*4):
            if i % 4 == 0:
                remaining_params += remaining_neurons[i]*remaining_neurons[i+1]*3
            else:
                remaining_params += remaining_neurons[i]*remaining_neurons[i+1]

        ib_out_pruned_n = sum([np.sum(mask.cpu().numpy() == 0) for mask in ib_out_masks])
        ib_attention_pruned_n = sum([np.sum(mask.cpu().numpy() == 0) for mask in ib_attention_masks])
        ff_ib1_pruned_n = sum([np.sum(mask.cpu().numpy() == 0) for mask in ff_ib1_masks])
        ff_ib2_pruned_n = sum([np.sum(mask.cpu().numpy() == 0) for mask in ff_ib2_masks])
        if prune_emb:
            emb_pruned_n = np.sum(emb_mask.cpu().numpy() == 0)
            return [ib_out_pruned_n, ib_attention_pruned_n, ff_ib1_pruned_n, ff_ib2_pruned_n, emb_pruned_n], \
                   remaining_params, remaining_params+30522*remaining_neurons_emb
        else:
            return [ib_out_pruned_n, ib_attention_pruned_n, ff_ib1_pruned_n, ff_ib2_pruned_n], \
                   remaining_params

    def setup_prune_threshold(self, threshold, prune_emb):
        for block in self.bottledBERT:
            block.ib_out.mask_thresh = threshold
            block.ib_attention.mask_thresh = threshold
            block.FF.ib1.mask_thresh = threshold
            block.FF.ib2.mask_thresh = threshold
        if prune_emb:
            self.ib_emb.mask_thresh = threshold


class BERTClassifier(nn.Module):
    """ Classifier with Transformer """

    def __init__(self, cfg, n_labels):
        super().__init__()
        self.transformer = BottledBERT(cfg)
        self.fc = nn.Linear(cfg.dim, cfg.dim)
        self.drop = nn.Dropout(cfg.drop_out)
        self.classifier = nn.Linear(cfg.dim, n_labels)

    def forward(self, input_ids, segment_ids, input_mask, prune_emb):
        h = self.transformer(input_ids, segment_ids, input_mask, prune_emb)
        # only use the first h in the sequence
        pooled_h = nn.Tanh()(self.fc(h[:, 0]))
        logits = self.classifier(self.drop(pooled_h))
        return logits

