#!/bin/bash
dvc run -n evaluate \
    -d code/train_and_eval/model.py -d code/train_and_eval/evaluate.py \
    -d /raid/nicolas/cifar10/data/cifar-prepared -d logs/cifar/version_0/ \
    python code/train_and_eval/evaluate.py logs/cifar/version_0 /raid/nicolas/cifar10/data/cifar-prepared
