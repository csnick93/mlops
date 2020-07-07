import pickle
import sys
from tqdm import tqdm
import os
import cv2
from sklearn.model_selection import train_test_split
from sys import path  # NOQA
path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # NOQA
from helpers.utils import load_params  # NOQA


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def save_image(filepath, im):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    cv2.imwrite(filepath, im)


def extract_batch(batch):
    images = []
    labels = []
    unpickled_batch = unpickle(batch)
    num_entries = len(unpickled_batch[b'labels'])
    for i in range(num_entries):
        im = unpickled_batch[b'data'][i]
        im = im.reshape((3, 32, 32)).transpose(1, 2, 0)
        label = unpickled_batch[b'labels'][i]
        images.append(im)
        labels.append(label)
    return images, labels


def save_to_disk(images, labels, folder):
    assert(len(images) == len(labels))
    for i in range(len(images)):
        sub_dir = os.path.join(folder, str(labels[i]))
        os.makedirs(sub_dir, exist_ok=True)
        cv2.imwrite(os.path.join(sub_dir, '%06i.png' % i), images[i])


def train_val_split(images, labels, split_size=0.2):
    train_images, train_labels, val_images, val_labels = train_test_split(
        images, labels, test_size=split_size)
    return train_images, train_labels, val_images, val_labels


def extract_training_and_validation_batches(cifar_folder, train_folder,
                                            val_folder, val_split):
    batches = [
        os.path.join(cifar_folder, b) for b in os.listdir(cifar_folder)
        if 'data' in b
    ]
    for batch in tqdm(batches):
        images, labels = extract_batch(batch)
    input_args = train_val_split(
        images, labels, val_split)
    save_to_disk(input_args[0], input_args[2],
                 train_folder)
    save_to_disk(input_args[1], input_args[3],
                 val_folder)


def extract_testing_batch(cifar_folder, test_folder):
    testing_batch = os.path.join(cifar_folder, 'test_batch')
    images, labels = extract_batch(testing_batch)
    save_to_disk(images, labels, test_folder)


def extract_data(cifar_folder, target_folder, val_split):
    print('Extracting training and validation data')
    extract_training_and_validation_batches(
        cifar_folder, os.path.join(target_folder, 'train'),
        os.path.join(target_folder, 'val'), val_split)
    print('Extracting test data')
    extract_testing_batch(cifar_folder, os.path.join(target_folder, 'test'))


if __name__ == '__main__':
    params = load_params('prepare')
    assert(len(sys.argv) == 3), "Need to provide input and output directory"
    raw_data_folder = os.path.abspath(sys.argv[1])
    prepared_data_folder = os.path.abspath(sys.argv[2])
    extract_data(raw_data_folder, prepared_data_folder, params['val_split'])
