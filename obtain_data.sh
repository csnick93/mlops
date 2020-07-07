#!/bin/bash
wget https://www.cs.toronto.edu/\~kriz/cifar-10-python.tar.gz -O $1/cifar-10-python.tar.gz && \
    tar -xvf $1/cifar-10-python.tar.gz -C $1 && \
    rm $1 && \
    git add cifar-10-batches-py.dvc && \
    git commit -m 'tracking data' && git push

