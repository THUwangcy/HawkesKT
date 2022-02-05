# -*- coding: UTF-8 -*-

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from models.BaseModel import BaseModel
from utils import utils


class SAKT(BaseModel):
    extra_log_args = ['num_layer', 'num_head']

    @staticmethod
    def parse_model_args(parser, model_name='SAKT'):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--num_layer', type=int, default=1,
                            help='Self-attention layers.')
        parser.add_argument('--num_head', type=int, default=4,
                            help='Self-attention heads.')
        return BaseModel.parse_model_args(parser, model_name)

    def __init__(self, args, corpus):
        super().__init__(model_path=args.model_path)
        self.skill_num = int(corpus.n_skills)
        self.question_num = int(corpus.n_problems)
        self.emb_size = args.emb_size
        self.num_head = args.num_head
        self.dropout = args.dropout

        self.inter_embeddings = nn.Embedding(self.skill_num * 2, self.emb_size)
        self.question_embeddings = nn.Embedding(self.question_num, self.emb_size)

        self.attn_blocks = nn.ModuleList([
            TransformerLayer(d_model=self.emb_size, d_feature=self.emb_size//self.num_head, d_ff=self.emb_size,
                             dropout=self.dropout, n_heads=self.num_head, kq_same=False, gpu=args.gpu)
            for _ in range(args.num_layer)
        ])

        self.out = nn.Linear(self.emb_size, 1)
        self.loss_function = nn.BCELoss(reduction='sum')

    def forward(self, feed_dict):
        seqs = feed_dict['inter_seq']          # [batch_size, real_max_step]
        questions = feed_dict['quest_seq']     # [batch_size, real_max_step]
        labels = feed_dict['label_seq']        # [batch_size, real_max_step]

        mask_labels = labels * (labels > -1).long()
        seq_data = self.inter_embeddings(seqs + mask_labels * self.skill_num)
        q_data = self.question_embeddings(questions)

        y = seq_data
        for block in self.attn_blocks:
            y = block(mask=1, query=q_data, key=y, values=y)
        prediction = self.out(y).squeeze(-1).sigmoid()

        out_dict = {'prediction': prediction[:, :-1], 'label': labels[:, 1:].double()}
        return out_dict

    def loss(self, feed_dict, outdict):
        prediction = outdict['prediction'].flatten()
        label = outdict['label'].flatten()
        mask = label > -1
        loss = self.loss_function(prediction[mask], label[mask])
        return loss

    def get_feed_dict(self, corpus, data, batch_start, batch_size, phase):
        batch_end = min(len(data), batch_start + batch_size)
        real_batch_size = batch_end - batch_start
        inter_seqs = data['skill_seq'][batch_start: batch_start + real_batch_size].values
        quest_seqs = data['problem_seq'][batch_start: batch_start + real_batch_size].values
        label_seqs = data['correct_seq'][batch_start: batch_start + real_batch_size].values

        feed_dict = {
            'inter_seq': torch.from_numpy(utils.pad_lst(inter_seqs)),            # [batch_size, real_max_step]
            'quest_seq': torch.from_numpy(utils.pad_lst(quest_seqs)),            # [batch_size, real_max_step]
            'label_seq': torch.from_numpy(utils.pad_lst(label_seqs, value=-1)),  # [batch_size, real_max_step]
        }
        return feed_dict


class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_feature, d_ff, n_heads, dropout, kq_same, gpu=''):
        super().__init__()
        """
            This is a Basic Block of Transformer paper. It containts one Multi-head attention object. 
            Followed by layer norm and postion wise feedforward net and dropout layer.
        """
        self.gpu = gpu
        # Multi-Head Attention Block
        self.masked_attn_head = MultiHeadAttention(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same, gpu=gpu)

        # Two layer norm layer and two droput layer
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, mask, query, key, values):
        seqlen, batch_size = query.size(1), query.size(0)
        nopeek_mask = np.triu(
            np.ones((1, 1, seqlen, seqlen)), k=mask).astype('uint8')
        src_mask = (torch.from_numpy(nopeek_mask) == 0)
        src_mask = src_mask.cuda() if self.gpu != '' else src_mask
        if mask == 0:  # If 0, zero-padding is needed.
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=True)
        else:
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=False)

        query = query + self.dropout1((query2))
        query = self.layer_norm1(query)
        query2 = self.linear2(self.dropout(
            self.activation(self.linear1(query))))
        query = query + self.dropout2((query2))
        query = self.layer_norm2(query)
        return query


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, bias=True, gpu=''):
        super().__init__()
        """
        It has projection layer for getting keys, queries and values. Followed by attention and a connected layer.
        """
        self.d_model = d_model
        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same
        self.gpu = gpu

        self.v_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        if kq_same is False:
            self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.proj_bias = bias
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, q, k, v, mask, zero_pad):
        bs = q.size(0)

        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        if self.kq_same is False:
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next
        scores = self.attention(q, k, v, self.d_k, mask, self.dropout, zero_pad)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).reshape(bs, -1, self.d_model)
        output = self.out_proj(concat)

        return output

    def attention(self, q, k, v, d_k, mask, dropout, zero_pad):
        """
        This is called by Multi-head attention object to find the values.
        """
        scores = torch.matmul(q, k.transpose(-2, -1)) / d_k ** 0.5  # BS, head, seqlen, seqlen
        bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)
        scores.masked_fill_(mask == 0, -np.inf)
        scores = F.softmax(scores, dim=-1)  # BS,head,seqlen,seqlen
        if zero_pad:
            pad_zero = torch.zeros(bs, head, 1, seqlen).double()
            pad_zero = pad_zero.cuda() if self.gpu != '' else pad_zero
            scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)
        scores = dropout(scores)
        output = torch.matmul(scores, v)
        return output
