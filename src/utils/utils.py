# -*- coding: UTF-8 -*-

import os
import logging
import datetime
import numpy as np


def format_arg_str(args, exclude_lst, max_len=20):
    linesep = os.linesep
    arg_dict = vars(args)
    keys = [k for k in arg_dict.keys() if k not in exclude_lst]
    values = [arg_dict[k] for k in keys]
    key_title, value_title = 'Arguments', 'Values'
    key_max_len = max(map(lambda x: len(str(x)), keys))
    value_max_len = min(max(map(lambda x: len(str(x)), values)), max_len)
    key_max_len, value_max_len = max([len(key_title), key_max_len]), max([len(value_title), value_max_len])
    horizon_len = key_max_len + value_max_len + 5
    res_str = linesep + '=' * horizon_len + linesep
    res_str += ' ' + key_title + ' ' * (key_max_len - len(key_title)) + ' | ' \
               + value_title + ' ' * (value_max_len - len(value_title)) + ' ' + linesep + '=' * horizon_len + linesep
    for key in sorted(keys):
        value = arg_dict[key]
        if value is not None:
            key, value = str(key), str(value).replace('\t', '\\t')
            value = value[:max_len-3] + '...' if len(value) > max_len else value
            res_str += ' ' + key + ' ' * (key_max_len - len(key)) + ' | ' \
                       + value + ' ' * (value_max_len - len(value)) + linesep
    res_str += '=' * horizon_len
    return res_str


def format_metric(metric):
    assert(type(metric) == dict)
    format_str = []
    for name in np.sort(list(metric.keys())):
        m = metric[name]
        if type(m) is float or type(m) is np.float or type(m) is np.float32 or type(m) is np.float64:
            format_str.append('{}:{:<.4f}'.format(name, m))
        elif type(m) is int or type(m) is np.int or type(m) is np.int32 or type(m) is np.int64:
            format_str.append('{}:{}'.format(name, m))
    return ','.join(format_str)


def pad_lst(lst, value=0, dtype=np.int64):
    inner_max_len = max(map(len, lst))
    result = np.ones([len(lst), inner_max_len], dtype) * value
    for i, row in enumerate(lst):
        for j, val in enumerate(row):
            result[i][j] = val
    return result


def check_dir(file_name):
    dir_path = os.path.dirname(file_name)
    if not os.path.exists(dir_path):
        print('make dirs:', dir_path)
        os.makedirs(dir_path)


def get_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def strictly_increasing(l):
    return all(x < y for x, y in zip(l, l[1:]))


def strictly_decreasing(l):
    return all(x > y for x, y in zip(l, l[1:]))


def non_increasing(l):
    return all(x >= y for x, y in zip(l, l[1:]))


def non_decreasing(l):
    return all(x <= y for x, y in zip(l, l[1:]))


def monotonic(l):
    return non_increasing(l) or non_decreasing(l)
