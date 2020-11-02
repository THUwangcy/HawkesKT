# -*- coding: UTF-8 -*-

import os
import sys
import math
import pickle
import argparse
import logging
import numpy as np
import pandas as pd


class DataReader(object):
    @staticmethod
    def parse_data_args(parser):
        parser.add_argument('--path', type=str, default='../data/',
                            help='Input data dir.')
        parser.add_argument('--dataset', type=str, default='ASSISTments_09-10',
                            help='Choose a dataset.')
        parser.add_argument('--sep', type=str, default='\t',
                            help='sep of csv file.')
        parser.add_argument('--kfold', type=int, default=5,
                            help='K-fold number.')
        parser.add_argument('--max_step', type=int, default=50,
                            help='Max time steps per sequence.')
        return parser

    def __init__(self, args):
        self.prefix = args.path
        self.sep = args.sep
        self.k_fold = args.kfold
        self.max_step = int(args.max_step)

        self.dataset = args.dataset
        self.data_df = {
            'train': pd.DataFrame(), 'dev': pd.DataFrame(), 'test': pd.DataFrame()
        }

        logging.info('Reading data from \"{}\", dataset = \"{}\" '.format(self.prefix, self.dataset))
        self.inter_df = pd.read_csv(os.path.join(self.prefix, self.dataset, 'interactions.csv'), sep=self.sep)

        # aggregate by user
        user_wise_dict = dict()
        cnt, n_inters = 0, 0
        for user, user_df in self.inter_df.groupby('user_id'):
            df = user_df[:self.max_step]  # consider the first 50 interactions
            user_wise_dict[cnt] = {
                'user_id': user,
                'skill_seq': df['skill_id'].values.tolist(),
                'correct_seq': [round(x) for x in df['correct']],
                'time_seq': df['timestamp'].values.tolist(),
                'problem_seq': df['problem_id'].values.tolist()
            }
            cnt += 1
            n_inters += len(df)
        self.user_seq_df = pd.DataFrame.from_dict(user_wise_dict, orient='index')

        self.n_users = max(self.inter_df['user_id'].values) + 1
        self.n_skills = max(self.inter_df['skill_id']) + 1
        self.n_problems = max(self.inter_df['problem_id']) + 1

        logging.info('"n_users": {}, "n_skills": {}, "n_problems": {}, "n_interactions": {}'.format(
            self.n_users, self.n_skills, self.n_problems, n_inters
        ))

    def gen_fold_data(self, k):
        assert k < self.k_fold
        n_examples = len(self.user_seq_df)
        fold_size = math.ceil(n_examples / self.k_fold)
        fold_begin = k * fold_size
        fold_end = min((k + 1) * fold_size, n_examples)
        self.data_df['test'] = self.user_seq_df.iloc[fold_begin:fold_end]
        residual_df = pd.concat([self.user_seq_df.iloc[0:fold_begin], self.user_seq_df.iloc[fold_end:n_examples]])
        dev_size = int(0.1 * len(residual_df))
        dev_indices = np.random.choice(residual_df.index, dev_size, replace=False)  # random
        self.data_df['dev'] = self.user_seq_df.iloc[dev_indices]
        self.data_df['train'] = residual_df.drop(dev_indices)
        logging.info('# Train: {}, # Dev: {}, # Test: {}'.format(
            len(self.data_df['train']), len(self.data_df['dev']), len(self.data_df['test'])
        ))

    def show_columns(self):
        logging.info('Data columns:')
        logging.info(self.user_seq_df.iloc[np.random.randint(0, len(self.user_seq_df))])


if __name__ == '__main__':
    logging.basicConfig(filename='../../log/test.txt', level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    parser = argparse.ArgumentParser(description='')
    parser = DataReader.parse_data_args(parser)

    args, extras = parser.parse_known_args()
    args.path = '../../data/'
    np.random.seed(2019)
    data = DataReader(args)
    data.gen_fold_data(k=0)
    data.show_columns()

    corpus_path = os.path.join(args.path, args.dataset, 'Corpus_{}.pkl'.format(args.max_step))
    logging.info('Save corpus to {}'.format(corpus_path))
    pickle.dump(data, open(corpus_path, 'wb'))
