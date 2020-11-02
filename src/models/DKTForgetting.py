# -*- coding: UTF-8 -*-

import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F

from models.BaseModel import BaseModel
from utils import utils


class DKTForgetting(BaseModel):
    extra_log_args = ['hidden_size', 'num_layer']

    @staticmethod
    def parse_model_args(parser, model_name='DKTForgetting'):
        parser.add_argument('--emb_size', type=int, default=32,
                            help='Size of embedding vectors.')
        parser.add_argument('--hidden_size', type=int, default=32,
                            help='Size of hidden vectors in LSTM.')
        parser.add_argument('--num_layer', type=int, default=1,
                            help='Number of GRU layers.')
        return BaseModel.parse_model_args(parser, model_name)

    def __init__(self, args, corpus):
        self.problem_num = int(corpus.n_problems)
        self.skill_num = int(corpus.n_skills)
        self.emb_size = args.emb_size
        self.hidden_size = args.hidden_size
        self.num_layer = args.num_layer
        self.dropout = args.dropout
        BaseModel.__init__(self, model_path=args.model_path)

    def _init_weights(self):
        self.skill_embeddings = torch.nn.Embedding(self.skill_num * 2, self.emb_size)
        self.rnn = torch.nn.LSTM(
            input_size=self.emb_size + 3, hidden_size=self.hidden_size, batch_first=True,
            num_layers=self.num_layer
        )
        self.fin = torch.nn.Linear(3, self.emb_size)
        self.fout = torch.nn.Linear(3, self.hidden_size)
        self.out = torch.nn.Linear(self.hidden_size + 3, self.skill_num)

        self.loss_function = torch.nn.BCELoss()

    def forward(self, feed_dict):
        seq_sorted = feed_dict['skill_seq']  # [batch_size, max_step]
        labels_sorted = feed_dict['label_seq']  # [batch_size, max_step]
        lengths = feed_dict['length']  # [batch_size]
        repeated_time_gap_seq = feed_dict['repeated_time_gap_seq']  # [batch_size, max_step]
        sequence_time_gap_seq = feed_dict['sequence_time_gap_seq']  # [batch_size, max_step]
        past_trial_counts_seq = feed_dict['past_trial_counts_seq']  # [batch_size, max_step]

        embed_history_i = self.skill_embeddings(seq_sorted + labels_sorted * self.skill_num)
        fin = self.fin(torch.cat((repeated_time_gap_seq, sequence_time_gap_seq, past_trial_counts_seq), dim=2))
        embed_history_i = torch.cat(
            (embed_history_i.mul(fin), repeated_time_gap_seq, sequence_time_gap_seq, past_trial_counts_seq), dim=2
        )
        embed_history_i_packed = torch.nn.utils.rnn.pack_padded_sequence(embed_history_i, lengths - 1, batch_first=True)
        output, hidden = self.rnn(embed_history_i_packed, None)

        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        fout = self.fout(torch.cat((repeated_time_gap_seq, sequence_time_gap_seq, past_trial_counts_seq), dim=2))
        output = torch.cat((
            output.mul(fout[:, 1:, :]), repeated_time_gap_seq[:, 1:, :],
            sequence_time_gap_seq[:, 1:, :], past_trial_counts_seq[:, 1:, :]
        ), dim=2)
        pred_vector = self.out(output)
        target_item = seq_sorted[:, 1:]
        prediction_sorted = torch.gather(pred_vector, dim=-1, index=target_item.unsqueeze(dim=-1)).squeeze(dim=-1)
        label = labels_sorted[:, 1:]

        prediction_sorted = torch.sigmoid(prediction_sorted)

        prediction = prediction_sorted[feed_dict['inverse_indice']]
        label = label[feed_dict['inverse_indice']].double()

        out_dict = {'prediction': prediction, 'label': label}
        return out_dict

    def loss(self, feed_dict, outdict):
        indice = feed_dict['indice']
        lengths = feed_dict['length'] - 1
        predictions, labels = outdict['prediction'][indice], outdict['label'][indice]
        predictions = torch.nn.utils.rnn.pack_padded_sequence(predictions, lengths, batch_first=True).data
        labels = torch.nn.utils.rnn.pack_padded_sequence(labels, lengths, batch_first=True).data
        loss = self.loss_function(predictions, labels)
        return loss

    def get_feed_dict(self, corpus, data, batch_start, batch_size, phase):
        batch_end = min(len(data), batch_start + batch_size)
        real_batch_size = batch_end - batch_start
        user_ids = data['user_id'][batch_start: batch_start + real_batch_size].values
        user_seqs = data['skill_seq'][batch_start: batch_start + real_batch_size].values
        label_seqs = data['correct_seq'][batch_start: batch_start + real_batch_size].values
        time_seqs = data['time_seq'][batch_start: batch_start + real_batch_size].values
        
        sequence_time_gap_seq, repeated_time_gap_seq, past_trial_counts_seq = \
            self.get_time_features(user_seqs, time_seqs)
        
        lengths = np.array(list(map(lambda lst: len(lst), user_seqs)))
        indice = np.array(np.argsort(lengths, axis=-1)[::-1])
        inverse_indice = np.zeros_like(indice)
        for i, idx in enumerate(indice):
            inverse_indice[idx] = i
            
        feed_dict = {
            'user_id': torch.from_numpy(user_ids[indice]),
            'skill_seq': torch.from_numpy(utils.pad_lst(user_seqs[indice])),  # [batch_size, max_step]
            'label_seq': torch.from_numpy(utils.pad_lst(label_seqs[indice])),  # [batch_size, max_step]
            'repeated_time_gap_seq': torch.from_numpy(repeated_time_gap_seq[indice]),  # [batch_size, max_step]
            'sequence_time_gap_seq': torch.from_numpy(sequence_time_gap_seq[indice]),  # [batch_size, max_step]
            'past_trial_counts_seq': torch.from_numpy(past_trial_counts_seq[indice]),  # [batch_size, max_step]
            'length': torch.from_numpy(lengths[indice]),  # [batch_size]
            'inverse_indice': torch.from_numpy(inverse_indice),
            'indice': torch.from_numpy(indice),
        }
        return feed_dict

    @staticmethod
    def get_time_features(user_seqs, time_seqs):
        skill_max = max([max(i) for i in user_seqs])
        inner_max_len = max(map(len, user_seqs))
        repeated_time_gap_seq = np.zeros([len(user_seqs), inner_max_len, 1], np.double)
        sequence_time_gap_seq = np.zeros([len(user_seqs), inner_max_len, 1], np.double)
        past_trial_counts_seq = np.zeros([len(user_seqs), inner_max_len, 1], np.double)
        for i in range(len(user_seqs)):
            last_time = None
            skill_last_time = [None for _ in range(skill_max)]
            skill_cnt = [0 for _ in range(skill_max)]
            for j in range(len(user_seqs[i])):
                sk = user_seqs[i][j] - 1
                ti = time_seqs[i][j]

                if skill_last_time[sk] is None:
                    repeated_time_gap_seq[i][j][0] = 0
                else:
                    repeated_time_gap_seq[i][j][0] = ti - skill_last_time[sk]
                skill_last_time[sk] = ti

                if last_time is None:
                    sequence_time_gap_seq[i][j][0] = 0
                else:
                    sequence_time_gap_seq[i][j][0] = (ti - last_time)
                last_time = ti

                past_trial_counts_seq[i][j][0] = (skill_cnt[sk])
                skill_cnt[sk] += 1

        repeated_time_gap_seq[repeated_time_gap_seq < 0] = 1
        sequence_time_gap_seq[sequence_time_gap_seq < 0] = 1
        repeated_time_gap_seq[repeated_time_gap_seq == 0] = 1e4
        sequence_time_gap_seq[sequence_time_gap_seq == 0] = 1e4
        past_trial_counts_seq += 1
        sequence_time_gap_seq *= 1.0 / 60
        repeated_time_gap_seq *= 1.0 / 60

        sequence_time_gap_seq = np.log(sequence_time_gap_seq)
        repeated_time_gap_seq = np.log(repeated_time_gap_seq)
        past_trial_counts_seq = np.log(past_trial_counts_seq)
        return sequence_time_gap_seq, repeated_time_gap_seq, past_trial_counts_seq
