# -*- coding: UTF-8 -*-

import os
import subprocess
import pandas as pd
import argparse
import re
import traceback
import numpy as np


# Run each fold and get the average metrics.
# e.g. run the following cmd in the "src" directory
# python utils/run_exp.py --in_f run.sh --out_f exp.csv


def parse_args():
    parser = argparse.ArgumentParser(description="Run")
    parser.add_argument('--log_dir', nargs='?', default='../log/',
                        help='Log save dir.')
    parser.add_argument('--cmd_dir', nargs='?', default='./',
                        help='Command dir.')
    parser.add_argument('--in_f', nargs='?', default='run.sh',
                        help='Input commands.')
    parser.add_argument('--out_f', nargs='?', default='WSDM-cf.csv',
                        help='Output csv.')
    parser.add_argument('--kfold', type=int, default=5,
                        help='K-fold number.')
    parser.add_argument('--skip', type=int, default=0,
                        help='skip number.')
    parser.add_argument('--total', type=int, default=-1,
                        help='total number.')
    parser.add_argument('--gpu', type=str, default='0',
                        help='Set CUDA_VISIBLE_DEVICES')
    return parser.parse_args()


def find_info(result):
    info = {}
    # prefix = 'INFO:root:'
    prefix = ''
    for line in result:
        if line.startswith(prefix + "Namespace("):
            info['Args'] = line
        elif line.startswith('Best Iter(dev)'):
            line = line.replace(' ', '')
            p = re.compile('BestIter\(dev\)=(\d*)')
            info['Best Iter'] = p.search(line).group(1)
            p = re.compile('\[([\d\.]+)s\]')
            info['Time'] = p.search(line).group(1)
            p = re.compile('valid=\(([\w:\.\d,]+)\)')
            info['Dev'] = p.search(line).group(1)
        elif line.startswith('Test After Training'):
            line = line.replace(' ', '')
            p = re.compile('TestAfterTraining:([\w:\.\d,]+)')
            info['Test'] = p.search(line).group(1)
    return info


def main():
    args = parse_args()
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    columns = ['Model', 'Description', 'Dev', 'Test', 'Fold',
               'Best Iter', 'Time', 'Run CMD', 'Args']
    skip, total = args.skip, args.total
    df = pd.DataFrame(columns=columns)
    in_f = open(os.path.join(args.cmd_dir, args.in_f), 'r')
    lines = in_f.readlines()
    for cmd in lines:
        cmd = cmd.strip()
        if cmd == '' or cmd.startswith('#') or cmd.startswith('export'):
            continue
        # cmd = eval(cmd)
        p = re.compile('--model_name (\w+)')
        model_name = p.search(cmd).group(1)
        for i in range(args.kfold):
            try:
                command = cmd
                if command.find(' --kfold') == -1:
                    command += ' --kfold {}'.format(args.kfold)
                if command.find(' --fold') == -1:
                    command += ' --fold {}'.format(i)
                if command.find(' --gpu ') == -1:
                    command += ' --gpu ' + args.gpu
                print(command)
                if skip > 0:
                    skip -= 1
                    continue

                result = subprocess.check_output(command, shell=True)
                result = result.decode('utf-8')
                result = [line.strip() for line in result.split(os.linesep)]
                # print result
                info = find_info(result)
                info['Fold'] = i
                info['Run CMD'] = command
                if args.kfold == 1:
                    info['Model'] = model_name
                # print info
                row = [info[c] if c in info else '' for c in columns]
                df.loc[len(df)] = row
                df.to_csv(os.path.join(args.log_dir, args.out_f), index=False)
                print(df[columns[:7]])

                total -= 1
                if total == 0:
                    break
            except Exception as e:
                traceback.print_exc()
                continue
        if args.kfold > 1:
            info = {'Model': model_name}
            tests = df['Test'].tolist()[-args.kfold:]
            avg_tests = dict()
            for t in tests:
                for s in t.split(','):
                    (metric, val) = s.split(':')
                    if metric not in avg_tests:
                        avg_tests[metric] = list()
                    avg_tests[metric].append(float(val))
            tests = ['{}:{:<.4f}'.format(key, np.average(lst)) for key, lst in avg_tests.items()]
            info['Test'] = ','.join(tests)
            row = [info[c] if c in info else '' for c in columns]
            df.loc[len(df)] = row
            print(df[columns[:7]])
        for i in range(3):
            row = [''] * len(columns)
            df.loc[len(df)] = row
        df.to_csv(os.path.join(args.log_dir, args.out_f), index=False)
        if total == 0:
            break


if __name__ == '__main__':
    main()
