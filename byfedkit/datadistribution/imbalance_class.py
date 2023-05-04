import random
import torch
from torchvision import datasets, transforms
import numpy as np


def _index_dict(dataset):
    """
    get {index: [labels]} dict of a dataset
    the key is a certain label of the dataset, the value is the index list of this label in the dataset
    like:
    {1:[1,3,4], 2:[2,5,7],...}
    :param dataset: given dataset
    :return: index->label dict
    """
    index_dict = {}
    for i in range(len(dataset)):
        label = torch.tensor(dataset.targets[i]).item()

        if label not in index_dict.keys():
            index_dict[label] = []
        index_dict[label].append(i)
    return index_dict


def load_mnist_index(client_num, class_per_client):
    pass


def load_mnist_full(client_num, class_per_client):
    pass