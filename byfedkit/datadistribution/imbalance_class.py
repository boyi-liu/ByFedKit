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


def load_mnist_index(client_num, class_per_client, class_num):
    """
    load random index of mnist, not the actual dataset
    :param client_num:
    :param class_per_client:
    :param class_num:
    :return:
    """
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset_train = datasets.MNIST('data/mnist/', train=True, download=True, transform=trans_mnist)
    dataset_test = datasets.MNIST('data/mnist/', train=False, download=True, transform=trans_mnist)

    shard_num = class_per_client * client_num // class_num

    index_dict_train = _index_dict(dataset_train)
    index_dict_test = _index_dict(dataset_test)

    shards_train = []
    shards_test = []

    for k in sorted(index_dict_train.keys()):
        index_k = np.array(index_dict_train[k])
        np.random.shuffle(index_k)

        chunks = np.array_split(index_k, shard_num)
        for _ in chunks:
            shards_train.append(_)

    for k in sorted(index_dict_test.keys()):
        index_k = np.array(index_dict_test[k])
        np.random.shuffle(index_k)

        chunks = np.array_split(index_k, shard_num)
        for _ in chunks:
            shards_test.append(_)

    labels = np.arange(class_num)
    np.random.shuffle(labels)
    select_list = np.repeat(labels, shard_num)  # [1,1,3,3,2,2,4,4,6,6,8,8,...]
    for ind, _ in enumerate(select_list):
        select_list[ind] = (_-1) * shard_num + ind % shard_num  # [0,1,4,5,2,3,...]

    finals_train = [[] for _ in range(client_num)]
    finals_test = [[] for _ in range(client_num)]

    for ind, s in enumerate(select_list):
        client_id = ind % client_num
        finals_train[client_id].extend(shards_train[s])
        finals_test[client_id].extend(shards_test[s])
    return finals_train, finals_test

def load_mnist_full(client_num, class_per_client):
    pass

load_mnist_index(10, 10, 10)