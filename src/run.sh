#!/usr/bin/env bash

export PYTHONPATH=../

# ASSISTments_09-10
 python main.py --model_name DKT --emb_size 64 --hidden_size 64 --max_step 50 --lr 1e-3 --l2 0 --gpu 1 --dataset ASSISTments_09-10

 python main.py --model_name SAKT --emb_size 64 --max_step 50 --lr 1e-4 --l2 1e-5 --dropout 0 --gpu 1 --dataset ASSISTments_09-10

 python main.py --model_name DKTForgetting --emb_size 64 --hidden_size 64 --max_step 50 --lr 1e-3 --l2 0 --gpu 1 --dataset ASSISTments_09-10

 python main.py --model_name KTM --emb_size 64 --max_step 50 --lr 1e-3 --l2 0 --gpu 1 --dataset ASSISTments_09-10

 python main.py --model_name AKT --emb_size 64 --max_step 50 --lr 1e-4 --l2 1e-4 --gpu 1 --dataset ASSISTments_09-10

 python main.py --model_name HawkesKT --emb_size 64 --max_step 50 --lr 5e-3 --l2 1e-5 --time_log 5 --gpu 1 --dataset ASSISTments_09-10


# ASSISTments_12-13
 python main.py --model_name DKT --emb_size 64 --hidden_size 64 --max_step 50 --lr 1e-3 --l2 0 --gpu 1 --dataset ASSISTments_12-13

 python main.py --model_name SAKT --emb_size 64 --max_step 50 --lr 1e-3 --l2 1e-5 --gpu 1 --dataset ASSISTments_12-13

 python main.py --model_name DKTForgetting --emb_size 64 --hidden_size 64 --max_step 50 --lr 1e-3 --l2 0 --gpu 1 --dataset ASSISTments_12-13

 python main.py --model_name KTM --emb_size 64 --max_step 50 --lr 1e-4 --l2 0 --gpu 1 --dataset ASSISTments_12-13

 python main.py --model_name AKT --emb_size 64 --max_step 50 --lr 1e-3 --l2 1e-5 --gpu 1 --dataset ASSISTments_12-13

 python main.py --model_name HawkesKT --emb_size 64 --max_step 50 --lr 1e-3 --l2 0 --time_log 5 --gpu 1 --dataset ASSISTments_12-13


# slepemapy_cz
 python main.py --model_name DKT --emb_size 64 --hidden_size 64 --max_step 50 --lr 1e-3 --l2 1e-5 --gpu 0 --dataset slepemapy_cz

 python main.py --model_name SAKT --emb_size 64 --hidden_size 64 --max_step 50 --lr 1e-3 --l2 1e-5 --gpu 1 --dataset slepemapy_cz

 python main.py --model_name DKTForgetting --emb_size 64 --hidden_size 64 --max_step 50 --lr 1e-3 --l2 1e-5 --gpu 1 --dataset slepemapy_cz

 python main.py --model_name KTM --emb_size 64 --max_step 50 --lr 1e-4 --l2 0 --gpu 1 --dataset slepemapy_cz

 python main.py --model_name AKT --emb_size 64 --max_step 50 --lr 1e-3 --l2 0 --gpu 1 --dataset slepemapy_cz

 python main.py --model_name HawkesKT --emb_size 64 --max_step 50 --lr 1e-4 --l2 0 --time_log 5 --gpu 1 --dataset slepemapy_cz