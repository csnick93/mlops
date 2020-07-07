#!/bin/bash
dvc repro;
# register ssh remote if not yet registered
if [[ $(git config remote.ssh.url) == "" ]]; then
    git remote add ssh ssh://git@gitlab.advertima.com:2222/tec/mlopsplayground.git;
fi
# need to commit updated dvc config files in case something changed
if [[ ! -z $(git status -uno -s) ]]; then
    git add -u && \
        git commit -m 'running dvc pipeline' && \
        git push ssh master
fi
