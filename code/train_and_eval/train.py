import os
import sys
import shutil
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from sys import path  # NOQA
path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # NOQA
from helpers.utils import load_params  # NOQA
from train_and_eval.model import CifarModel


if __name__ == '__main__':
    params = load_params('train')
    assert(len(sys.argv) ==
           4), "Need to provide data directory, log directory and log name"
    data_folder = sys.argv[1]
    assert(os.path.exists(data_folder)), "Data directory does not exist"
    log_dir = sys.argv[2]
    log_name = sys.argv[3]

    # need to make sure log folder does not exist yet,
    #   otherwise we would won't know which saved model is ours
    log_folder = os.path.join(log_dir, log_name)
    if os.path.exists(log_folder):
        shutil.rmtree(log_folder)

    model = CifarModel(data_folder)
    logger = TensorBoardLogger(log_dir, name=log_name)
    trainer = Trainer(logger=logger,
                      auto_lr_find=False,
                      gpus=1,
                      num_nodes=1,
                      max_epochs=params['max_epochs'],
                      distributed_backend='ddp')
    trainer.fit(model)
