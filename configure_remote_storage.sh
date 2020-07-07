#!/bin/bash
dvc remote add -f -d storage s3://adv-tec.sample-data/cifar10/ && \
    git commit .dvc/config -m 'configure remote storage' && \
    dvc push && git push
