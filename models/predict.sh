#!/bin/sh
#rm -rf models/* tmp/*
CUDA_VISIBLE_DEVICES=4 nohup  python run.py --predict\
    --algo MLSTM\
    --g 4\
    --test_files /data1/wangpeijian/test1set/preprocessed/zhidao.test1.json > log/predict.log 2>&1 &
