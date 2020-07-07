#!/bin/bash
docker run -t --rm -v /raid/nicolas/cifar10:/raid/nicolas/cifar10 \
                    -v /home/nicolas/.ssh/:/root/.ssh/ \
                    -v /home/nicolas/.aws/:/home/user/.aws/ \
                    --gpus all \
                    cifar10:v0
