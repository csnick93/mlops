import os
import sys
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from sys import path  # NOQA
path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # NOQA
from helpers.utils import load_params, load_logger_from_dir   # NOQA
from train_and_eval.model import CifarModel


def _get_checkpoint_file(log_dir):
    checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    assert(os.path.exists(checkpoint_dir)), "No checkpoint directory found"
    checkpoint_files = [f for f in os.listdir(
        checkpoint_dir) if f.endswith('.ckpt')]
    assert(len(checkpoint_files) == 1), "Expecting exactly one checkpoint"
    return os.path.join(checkpoint_dir, checkpoint_files[0])


def load_model(log_dir):
    checkpoint_file = _get_checkpoint_file(log_dir)
    model = CifarModel.load_from_checkpoint(checkpoint_file)
    return model


if __name__ == '__main__':
    assert(len(sys.argv) == 3)
    log_dir = sys.argv[1]
    data_path = sys.argv[2]
    model = load_model(log_dir)
    model.data_path = data_path
    logger = load_logger_from_dir(log_dir)
    trainer = Trainer(deterministic=True, logger=logger)
    trainer.test(model)
