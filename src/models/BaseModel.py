# -*- coding: UTF-8 -*-

import torch
import time
from tqdm import tqdm
import logging
from sklearn.metrics import *
import numpy as np
import torch.nn.functional as F
import os

from utils.utils import *


class BaseModel(torch.nn.Module):
    reader = 'DataReader'
    runner = 'KTRunner'
    extra_log_args = []

    @staticmethod
    def parse_model_args(parser, model_name='BaseModel'):
        parser.add_argument('--model_path', type=str, default='',
                            help='Model save path.')
        return parser

    @staticmethod
    def pred_evaluate_method(p, l, metrics):
        evaluations = dict()
        bi_p = (p > 0.5).astype(int)
        for metric in metrics:
            if metric == 'rmse':
                evaluations[metric] = np.sqrt(mean_squared_error(l, p))
            elif metric == 'mae':
                evaluations[metric] = mean_absolute_error(l, p)
            elif metric == 'auc':
                evaluations[metric] = roc_auc_score(l, p)
            elif metric == 'f1':
                evaluations[metric] = f1_score(l, bi_p)
            elif metric == 'accuracy':
                evaluations[metric] = accuracy_score(l, bi_p)
            elif metric == 'precision':
                evaluations[metric] = precision_score(l, bi_p)
            elif metric == 'recall':
                evaluations[metric] = recall_score(l, bi_p)
        return evaluations

    @staticmethod
    def init_weights(m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                torch.nn.init.normal_(m.bias, mean=0.0, std=0.01)
        elif type(m) == torch.nn.Embedding:
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
        elif type(m) in [torch.nn.GRU, torch.nn.LSTM, torch.nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)

    @staticmethod
    def batch_to_gpu(batch):
        if torch.cuda.device_count() > 0:
            for key in batch:
                batch[key] = batch[key].cuda()
        return batch

    def __init__(self, model_path='../model/Model/Model.pt'):
        super(BaseModel, self).__init__()
        self.model_path = model_path
        self._init_weights()
        self.optimizer = None

    def _init_weights(self):
        pass

    def forward(self, feed_dict):
        pass

    def get_feed_dict(self, corpus, data, batch_start, batch_size, train):
        pass

    def prepare_batches(self, corpus, data, batch_size, phase):
        num_example = len(data)
        total_batch = int((num_example + batch_size - 1) / batch_size)
        assert(num_example > 0)
        # for batch in range(total_batch):
        #     yield self.get_feed_dict(corpus, data, batch * batch_size, batch_size, phase)
        batches = []
        for batch in tqdm(range(total_batch), leave=False, ncols=100, mininterval=1, desc='Prepare Batches'):
            batches.append(self.get_feed_dict(corpus, data, batch * batch_size, batch_size, phase))
        return batches

    def count_variables(self):
        total_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_parameters

    def save_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        dir_path = os.path.dirname(model_path)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        torch.save(self.state_dict(), model_path)
        logging.info('Save model to ' + model_path)

    def load_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        self.load_state_dict(torch.load(model_path))
        self.eval()
        logging.info('Load model from ' + model_path)

    def customize_parameters(self):
        # customize optimizer settings for different parameters
        weight_p, bias_p = [], []
        for name, p in filter(lambda x: x[1].requires_grad, self.named_parameters()):
            if 'bias' in name:
                bias_p.append(p)
            else:
                weight_p.append(p)
        optimize_dict = [{'params': weight_p}, {'params': bias_p, 'weight_decay': 0}]
        return optimize_dict

    def actions_before_train(self):
        total_parameters = self.count_variables()
        logging.info('#params: %d' % total_parameters)

    def actions_after_train(self):
        pass
