#!/bin/bash
docker run -it --rm -v /raid/nicolas/cifar10:/raid/nicolas/cifar10 \
                    -v /home/nicolas/.ssh/:/root/.ssh/ \
                    -v /home/nicolas/.aws/:/home/user/.aws/ \
                    --gpus all \
                    cifar10_setup:v0 
# need to update repo after
git pull
