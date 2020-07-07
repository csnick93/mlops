#!/bin/bash

# init dvc
dvc init && \
    git add .dvc && \
    git commit -m 'initializing dvc' && \
    git push


# externalizing cache directory
mkdir -p /raid/nicolas/cifar10/dvc-cache && \
    dvc cache dir /raid/nicolas/cifar10/dvc-cache

# download data
mkdir -p /raid/nicolas/cifar10/data && \
    ./obtain_data.sh /raid/nicolas/cifar10/data

# track data
dvc add /raid/nicolas/cifar10/data/cifar-10-batches-py --external && \
    git add cifar-10-batches-py.dvc && \
    git commit -m 'setup dvc and track data' && \
    git push


# enabling remote storage
dvc remote add -d storage s3://adv-tec.sample-data/cifar10/ && \
    git add -u && git commit -m 'configuring remote storage' &&
    dvc push && git push

# need to run pipeline once to register
./dvc_prepare.sh && \
    ./dvc_train.sh && \
    ./dvc_evaluate.sh

# commit and push dvc config files
git add dvc.yaml dvc.lock &&
    git commit -m 'registering pipeline' && \
    git push
