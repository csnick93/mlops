import sys
import os
import seaborn as sn
import pandas as pd
import numpy as np
import torch
import torchvision
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pytorch_lightning.core.lightning import LightningModule
from PIL import Image
import matplotlib.pyplot as plt
from sklearn import metrics
import mlflow
from mlflow.pytorch import log_model
from mlflow import log_param, log_metric, log_artifact
from sys import path  # NOQA
path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # NOQA
from helpers.utils import connect_to_experiment, checkpoint_file_from_logger, tensorboard_file_from_logger  # NOQA

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
           'ship', 'truck')

# mlflow settings
REMOTE_SERVER_URI = 'http://62.96.249.154:5040/'
mlflow.set_tracking_uri(REMOTE_SERVER_URI)
EXPERIMENT_NAME = 'cifar10'


class Cifar10(Dataset):
    def __init__(self, data_path, transformations):
        assert (len(transformations) > 0)
        super(Cifar10, self).__init__()
        self.transformer = transforms.Compose(transformations)
        self.images, self.labels = self._read_in_data(data_path)

    def _read_in_data(self, data_path):
        labels = []
        images = []
        for label_folder in os.listdir(data_path):
            for filename in os.listdir(os.path.join(data_path, label_folder)):
                label = int(label_folder)
                image = Image.open(
                    os.path.join(data_path, label_folder, filename))
                image = self.transformer(image)
                labels.append(label)
                images.append(image)
        return images, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


class Cifar10Inspector:
    def __init__(self, data_path, batch_size=16):
        self._setup_loader(data_path, batch_size)

    def _setup_loader(self, data_path, batch_size):
        test_path = os.path.join(data_path, 'test')
        transformations = [
            transforms.ToTensor(),
        ]
        dataset = Cifar10(test_path, transformations=transformations)
        self.loader = iter(
            DataLoader(dataset,
                       shuffle=True,
                       batch_size=batch_size,
                       num_workers=0))

    def _imshow(self, img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    def inspect(self):
        images, labels = self.loader.next()
        class_labels = [self.classes[label] for label in labels]
        print('\t'.join(class_labels))
        self._imshow(torchvision.utils.make_grid(images))


class CifarModel(LightningModule):
    def __init__(self, data_path, batch_size=32, learning_rate=0.001):
        super(CifarModel, self).__init__()
        self.data_path = data_path
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.transformations = [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(),
                               lr=self.learning_rate,
                               momentum=0.9)

    def train_dataloader(self):
        train_path = os.path.join(self.data_path, 'train')
        dataset = Cifar10(train_path, transformations=self.transformations)
        loader = DataLoader(dataset,
                            batch_size=self.batch_size,
                            shuffle=True,
                            num_workers=4)
        return loader

    def val_dataloader(self):
        val_path = os.path.join(self.data_path, 'val')
        dataset = Cifar10(val_path, transformations=self.transformations)
        loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=4)
        return loader

    def test_dataloader(self):
        test_path = os.path.join(self.data_path, 'test')
        dataset = Cifar10(test_path, transformations=self.transformations)
        loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=4)
        return loader

    def _accuracy(self, y_hat, y):
        return ((torch.max(y_hat, 1)[1] == y).sum().type(torch.float32) /
                y.shape[0])

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return {
            'val_loss': F.cross_entropy(y_hat, y),
            'val_accuracy': self._accuracy(y_hat, y)
        }

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_accuracy = torch.stack([x['val_accuracy'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss, 'val_accuracy': avg_accuracy}
        return {
            'val_loss': avg_loss,
            'val_accuracy': avg_accuracy,
            'log': tensorboard_logs
        }

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return {
            'test_loss': F.cross_entropy(y_hat, y),
            'test_accuracy': self._accuracy(y_hat, y)
        }

    def test_epoch_end(self, outputs):
        conf_mat_path = self._create_confusion_matrix_plot()
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_accuracy = torch.stack([x['test_accuracy']
                                    for x in outputs]).mean()

        metrics = [("loss", float(avg_loss.cpu().numpy())),
                   ("accuracy", float(avg_accuracy.cpu().numpy()))]
        artifacts = [conf_mat_path, checkpoint_file_from_logger(
            self.logger), tensorboard_file_from_logger(self.logger)]
        self._log_to_mlflow(metrics, artifacts)

        return {
            'test_loss': avg_loss,
            'test_accuracy': avg_accuracy,
        }

    def _log_to_mlflow(self, metrics, artifacts):
        exp_id, _ = connect_to_experiment(EXPERIMENT_NAME)
        with mlflow.start_run(run_name='test_run',  # self.run_name,
                              experiment_id=exp_id):
            log_param("learning_rate", self.learning_rate)
            log_param("batch_size", self.batch_size)

            for metric in metrics:
                log_metric(*metric)

            for artifact in artifacts:
                log_artifact(artifact)

    def _create_confusion_matrix_plot(self):
        confusion_matrix = self._compute_confusion_matrix()
        _, conf_mat_path = self._draw_confusion_matrix(confusion_matrix)
        # self.logger.experiment.add_image('confusion matrix',
        #                                  conf_mat.transpose((2, 0, 1)))

        return conf_mat_path

    def _compute_confusion_matrix(self):
        test_loader = self.test_dataloader()
        y = np.empty(0, np.int)
        y_hat = np.empty(0, np.int)
        for data in test_loader:
            x_, y_ = data
            pred = self(x_)
            computed_labels = torch.max(pred, 1)[1].numpy()
            y_hat = np.concatenate((y_hat, computed_labels))
            y = np.concatenate((y, y_))
        confusion_matrix = metrics.confusion_matrix(y, y_hat)
        return confusion_matrix

    def _draw_confusion_matrix(self, conf_mat, normalized=True):
        conf_mat = conf_mat / np.sum(conf_mat)
        conf_mat = np.round(conf_mat, 2)
        df_cm = pd.DataFrame(conf_mat, CLASSES, CLASSES)
        sn.set(font_scale=1.4)
        sn.heatmap(df_cm, cmap='Blues', annot=True, annot_kws={"size": 12})
        out_file = os.path.join(self.logger.save_dir, self.logger._name,
                                'version_%i' % self.logger._version,
                                'conf_matrix.png')
        plt.tight_layout()
        plt.savefig(out_file)
        im = np.asarray(Image.open(out_file))
        return im, out_file
