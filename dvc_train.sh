#!/bin/bash
dvc run -n train \
    -p train.max_epochs \
    -d code/train_and_eval/model.py -d code/train_and_eval/train.py -d /raid/nicolas/cifar10/data/cifar-prepared \
    -o logs/cifar/version_0/ \
    python code/train_and_eval/train.py /raid/nicolas/cifar10/data/cifar-prepared logs cifar
