#!/bin/sh
mkdir -p log
rm -rf log/*
#rm -rf models/* tmp/*
CUDA_VISIBLE_DEVICES=3 nohup python -u run.py --demo\
    --algo MLSTM\
    --g 3 > log/train.log 2>&1 &
