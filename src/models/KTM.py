# -*- coding: UTF-8 -*-

import numpy as np
import torch
import torch.nn.functional as F

from models.BaseModel import BaseModel
from models.DKTForgetting import DKTForgetting
from utils import utils


class KTM(BaseModel):
    @staticmethod
    def parse_model_args(parser, model_name='KTM'):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        return BaseModel.parse_model_args(parser, model_name)

    def __init__(self, args, corpus):
        self.dataset = args.dataset
        self.problem_num = int(corpus.n_problems)
        self.skill_num = int(corpus.n_skills)
        self.emb_size = args.emb_size
        self.dropout = args.dropout
        self.max_step = args.max_step
        # total dimension of sparse feature vector
        self.num_features = self.problem_num + self.skill_num * 3 + 3
        BaseModel.__init__(self, model_path=args.model_path)

    def _init_weights(self):
        self.W = torch.nn.Embedding(self.num_features, 1)
        self.V = torch.nn.Embedding(self.num_features, self.emb_size)
        self.global_bias = torch.nn.Parameter(torch.Tensor([0.]))

        self.loss_function = torch.nn.BCELoss()

    def forward(self, feed_dict):
        idxs = feed_dict['idx']              # [batch_size, real_max_step, max_num_idx]
        vals = feed_dict['val']              # [batch_size, real_max_step, max_num_idx]
        labels = feed_dict['label_seq']      # [batch_size, real_max_step]

        w = self.W(idxs).squeeze()
        predictions = (w * vals).sum(-1) + self.global_bias

        if self.emb_size > 0:
            v = self.V(idxs)
            interaction = v * vals.unsqueeze(-1)
            predictions += 0.5 * (interaction.sum(-2) ** 2 - (interaction ** 2).sum(-2)).sum(-1)

        prediction = torch.sigmoid(predictions)

        out_dict = {'prediction': prediction[:, 1:], 'label': labels[:, 1:].double()}
        return out_dict

    def loss(self, feed_dict, outdict):
        lengths = feed_dict['length'] - 1
        indice = torch.argsort(lengths, dim=-1, descending=True)
        predictions, labels, lengths = outdict['prediction'][indice], outdict['label'][indice], lengths[indice]
        predictions = torch.nn.utils.rnn.pack_padded_sequence(predictions, lengths, batch_first=True).data
        labels = torch.nn.utils.rnn.pack_padded_sequence(labels, lengths, batch_first=True).data
        loss = self.loss_function(predictions, labels)
        return loss

    def get_feed_dict(self, corpus, data, batch_start, batch_size, phase):
        batch_end = min(len(data), batch_start + batch_size)
        real_batch_size = batch_end - batch_start
        user_seqs = data['skill_seq'][batch_start: batch_start + real_batch_size].values
        label_seqs = data['correct_seq'][batch_start: batch_start + real_batch_size].values
        problem_seqs = data['problem_seq'][batch_start: batch_start + real_batch_size].values
        time_seqs = data['time_seq'][batch_start: batch_start + real_batch_size].values

        lengths = np.array(list(map(lambda lst: len(lst), user_seqs)))

        fm_idxs, fm_vals = self.construct_sparse_feature(
            user_seqs, label_seqs, problem_seqs, time_seqs, lengths
        )

        feed_dict = {
            'idx': torch.from_numpy(fm_idxs).long(),                     # [batch_size, real_max_step, max_num_idx]
            'val': torch.from_numpy(fm_vals),                            # [batch_size, real_max_step, max_num_idx]
            'length': torch.from_numpy(lengths),                         # [batch_size]
            'label_seq': torch.from_numpy(utils.pad_lst(label_seqs))     # [batch_size]
        }
        return feed_dict

    def construct_sparse_feature(self, user_seqs, label_seqs, problem_seqs, time_seqs, lengths):
        real_batch_size = user_seqs.shape[0]
        t1_seq, t2_seq, t3_seq = DKTForgetting.get_time_features(user_seqs, time_seqs)

        max_step, max_num_idx = np.max(lengths), 2 + min(self.max_step - 1, self.skill_num * 2) + 4
        fm_idxs, fm_vals = np.zeros((real_batch_size, max_step, max_num_idx)), \
                           np.zeros((real_batch_size, max_step, max_num_idx))
        real_num_idx = 0
        for i in range(real_batch_size):
            win_attemp_dict, fail_attemp_dict = dict(), dict()
            for j in range(lengths[i]):
                id, base = 0, 0
                # Question id
                fm_idxs[i][j][id] = base + problem_seqs[i][j]
                fm_vals[i][j][id] = 1
                base += self.problem_num
                # Skill id
                id += 1
                fm_idxs[i][j][id] = base + user_seqs[i][j]
                fm_vals[i][j][id] = 1
                base += self.skill_num
                # Win
                for s, num in win_attemp_dict.items():
                    id += 1
                    fm_idxs[i][j][id] = base + s
                    fm_vals[i][j][id] = num
                base += self.skill_num
                # Fail
                for s, num in fail_attemp_dict.items():
                    id += 1
                    fm_idxs[i][j][id] = base + s
                    fm_vals[i][j][id] = num
                base += self.skill_num
                # Time
                id += 1
                fm_idxs[i][j][id] = base
                fm_vals[i][j][id] = t1_seq[i][j]
                id += 1
                fm_idxs[i][j][id] = base + 1
                fm_vals[i][j][id] = t2_seq[i][j]
                id += 1
                fm_idxs[i][j][id] = base + 2
                fm_vals[i][j][id] = t3_seq[i][j]

                # Update information
                if label_seqs[i][j]:
                    if user_seqs[i][j] not in win_attemp_dict:
                        win_attemp_dict[user_seqs[i][j]] = 0
                    win_attemp_dict[user_seqs[i][j]] += 1
                else:
                    if user_seqs[i][j] not in fail_attemp_dict:
                        fail_attemp_dict[user_seqs[i][j]] = 0
                    fail_attemp_dict[user_seqs[i][j]] += 1
                real_num_idx = max(real_num_idx, id + 1)

        fm_idxs = fm_idxs[:, :, :real_num_idx]
        fm_vals = fm_vals[:, :, :real_num_idx]
        return fm_idxs, fm_vals
