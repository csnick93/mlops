#!/bin/bash
dvc run -n prepare \
    -p prepare.val_split \
    -d code/data_preparation/extract_data.py -d /raid/nicolas/cifar10/data/cifar-10-batches-py \
    --external \
    -o /raid/nicolas/cifar10/data/cifar-prepared \
    python code/data_preparation/extract_data.py \
            /raid/nicolas/cifar10/data/cifar-10-batches-py \
            /raid/nicolas/cifar10/data/cifar-prepared
