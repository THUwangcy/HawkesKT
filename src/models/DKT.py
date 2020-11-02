# -*- coding: UTF-8 -*-

import numpy as np
import torch
import torch.nn.functional as F

from models.BaseModel import BaseModel
from utils import utils


class DKT(BaseModel):
    extra_log_args = ['hidden_size', 'num_layer']

    @staticmethod
    def parse_model_args(parser, model_name='DKT'):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--hidden_size', type=int, default=64,
                            help='Size of hidden vectors in LSTM.')
        parser.add_argument('--num_layer', type=int, default=1,
                            help='Number of GRU layers.')
        return BaseModel.parse_model_args(parser, model_name)

    def __init__(self, args, corpus):
        self.skill_num = int(corpus.n_skills)
        self.emb_size = args.emb_size
        self.hidden_size = args.hidden_size
        self.num_layer = args.num_layer
        self.dropout = args.dropout
        BaseModel.__init__(self, model_path=args.model_path)

    def _init_weights(self):
        self.skill_embeddings = torch.nn.Embedding(self.skill_num * 2, self.emb_size)
        self.rnn = torch.nn.LSTM(
            input_size=self.emb_size, hidden_size=self.hidden_size, batch_first=True,
            num_layers=self.num_layer
        )
        self.out = torch.nn.Linear(self.hidden_size, self.skill_num)

        self.loss_function = torch.nn.BCELoss()

    def forward(self, feed_dict):
        seq_sorted = feed_dict['skill_seq']     # [batch_size, history_max]
        labels_sorted = feed_dict['label_seq']  # [batch_size, history_max]
        lengths = feed_dict['length']           # [batch_size]

        embed_history_i = self.skill_embeddings(seq_sorted + labels_sorted * self.skill_num)
        embed_history_i_packed = torch.nn.utils.rnn.pack_padded_sequence(embed_history_i, lengths - 1, batch_first=True)
        output, hidden = self.rnn(embed_history_i_packed, None)

        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
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

        lengths = np.array(list(map(lambda lst: len(lst), user_seqs)))
        indice = np.array(np.argsort(lengths, axis=-1)[::-1])
        inverse_indice = np.zeros_like(indice)
        for i, idx in enumerate(indice):
            inverse_indice[idx] = i

        feed_dict = {
            'user_id': torch.from_numpy(user_ids[indice]),
            'skill_seq': torch.from_numpy(utils.pad_lst(user_seqs[indice])),    # [batch_size, num of items to predict]
            'label_seq': torch.from_numpy(utils.pad_lst(label_seqs[indice])),   # [batch_size, num of items to predict]
            'length': torch.from_numpy(lengths[indice]),                        # [batch_size]
            'inverse_indice': torch.from_numpy(inverse_indice),
            'indice': torch.from_numpy(indice)
        }
        return feed_dict
