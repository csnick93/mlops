import yaml
import warnings
import os
import mlflow
from pytorch_lightning.loggers import TensorBoardLogger


def load_logger_from_dir(log_path):
    log_path = log_path.rstrip('/')
    root, version = os.path.split(log_path)
    log_dir, log_name = os.path.split(root)
    logger = TensorBoardLogger(log_dir,
                               name=log_name,
                               version=int(version.lstrip('version_')))
    return logger


def checkpoint_file_from_logger(logger):
    checkpoint_dir = os.path.abspath(
        os.path.join(logger.save_dir, logger._name,
                     f'version_{logger._version}', 'checkpoints'))
    checkpoint_files = [
        f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')
    ]
    assert (
        len(checkpoint_files) == 1), 'Not exactly one checkpoint file found'
    return os.path.join(checkpoint_dir, checkpoint_files[0])


def tensorboard_file_from_logger(logger):
    logger_dir = os.path.abspath(
        os.path.join(logger.save_dir, logger._name,
                     f'version_{logger._version}'))
    event_files = [f for f in os.listdir(logger_dir) if f.startswith('events')]
    assert(len(event_files) > 0), 'no events found'
    if len(event_files) > 1:
        warnings.warn('Multiple events found, taking oldest one')
        event_files = sorted(event_files, key=lambda x: int(x.split('.')[3]))
    event_file = event_files[0]
    return os.path.join(logger_dir, event_file)


def connect_to_experiment(experiment_name):
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        run_id = 1
    else:
        experiment_id = experiment.experiment_id
        run_id = get_max_run_id(experiment_id)
    return experiment_id, run_id


def get_max_run_id(experiment_id):
    runs = mlflow.search_runs(experiment_ids=[experiment_id])
    run_id = len(runs)
    return run_id


def load_params(keyword):
    params_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'params.yaml')
    with open(params_file, 'r') as f:
        all_params = yaml.safe_load(f)
    return all_params[keyword]
