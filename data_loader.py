import bisect
import os

import numpy as np
import numpy.random as nr
import torch
from fuzzywuzzy import fuzz
from PIL import Image
from scipy import io
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from torchvision import datasets, transforms

MAPPING = {
    "mnist": (datasets.MNIST, [1000] * 10),
    "cifar10": (datasets.CIFAR10, [1000] * 10),
    "cifar100": (datasets.CIFAR100, [100] * 10),
    "svhn": (datasets.SVHN, [1000] * 10),
    "sun397": (datasets.SUN397, [1000] * 10),
    # Add more dataset mappings as needed
}

def get_class_nsamples(dataset_name):
    if dataset_name in MAPPING:
        return MAPPING[dataset_name]
    else:
        fuzzy_match = get_fuzzy_match(dataset_name, MAPPING.keys())
        if fuzzy_match is not None:
            return MAPPING[fuzzy_match]
        else:
            raise ValueError("Unsupported dataset name: " + dataset_name)

DATA_ROOT = os.path.expanduser('./data')

def instantiate_unknown_class(class_type, **kwargs):
    # Get the constructor signature of the class
    constructor = class_type.__init__

    # Get the parameters of the constructor
    parameters = list(constructor.__code__.co_varnames)[1:]  # Exclude 'self' parameter

    # Filter the keyword arguments based on the constructor parameters
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in parameters}

    # Instantiate the class with the filtered keyword arguments
    instance = class_type(**filtered_kwargs)

    return instance


def make_longtailed_imb(max_num, class_num, gamma):
    mu = np.power(1/gamma, 1/(class_num - 1))
    print(mu)
    class_num_list = []
    for i in range(class_num):
        class_num_list.append(int(max_num * np.power(mu, i)))

    return list(class_num_list)


def get_val_test_data(dataset, num_sample_per_class, shuffle=False, random_seed=0):
    """
    Return a list of indices for validation and test from a dataset.
    Input: A test dataset (e.g., CIFAR-10)
    Output: validation_list and test_list
    """
    length = dataset.__len__()
    num_sample_per_class = list(num_sample_per_class)
    num_samples = num_sample_per_class[0] # Suppose that all classes have the same number of test samples

    val_list = []
    test_list = []
    indices = list(range(0, length))
    if shuffle:
        nr.shuffle(indices)
    for i in range(0, length):
        index = indices[i]
        _, label = dataset.__getitem__(index)
        if num_sample_per_class[label] > (9 * num_samples / 10):
            val_list.append(index)
            num_sample_per_class[label] -= 1
        else:
            test_list.append(index)
            num_sample_per_class[label] -= 1

    return val_list, test_list


def get_oversampled_data(dataset, num_sample_per_class, random_seed=0):
    """
    Return a list of imbalanced indices from a dataset.
    Input: A dataset (e.g., CIFAR-10), num_sample_per_class: list of integers
    Output: oversampled_list ( weights are increased )
    """
    length = dataset.__len__()
    num_sample_per_class = list(num_sample_per_class)
    num_samples = list(num_sample_per_class)

    selected_list = []
    indices = list(range(0,length))
    for i in range(0, length):
        index = indices[i]
        _, label = dataset.__getitem__(index)
        if num_sample_per_class[label] > 0:
            selected_list.append(1 / num_samples[label])
            num_sample_per_class[label] -= 1

    return selected_list


def get_imbalanced_data(dataset, num_sample_per_class, shuffle=False, random_seed=0):
    """
    Return a list of imbalanced indices from a dataset.
    Input: A dataset (e.g., CIFAR-10), num_sample_per_class: list of integers
    Output: imbalanced_list
    """
    length = dataset.__len__()
    num_sample_per_class = list(num_sample_per_class)
    selected_list = []
    indices = list(range(0,length))

    for i in range(0, length):
        index = indices[i]
        _, label = dataset.__getitem__(index)
        if num_sample_per_class[label] > 0:
            selected_list.append(index)
            num_sample_per_class[label] -= 1

    return selected_list


def get_oversampled(dataset, num_sample_per_class, batch_size, TF_train, TF_test):
    print("Building {} CV data loader with {} workers".format(dataset, 8))
    ds = []

    dataset_, num_test_samples = get_class_nsamples(dataset)
    train_set = instantiate_unknown_class(dataset_, root=DATA_ROOT, train=True, split='train', download=False, transform=TF_train)
    try:
        targets = np.array(train_set.targets)
    except:
        targets = np.array(train_set.labels)
   
    classes, class_counts = np.unique(targets, return_counts=True)
    nb_classes = len(classes)

    imbal_class_counts = [int(i) for i in num_sample_per_class]
    class_indices = [np.where(targets == i)[0] for i in range(nb_classes)]

    imbal_class_indices = [class_idx[:class_count] for class_idx, class_count in zip(class_indices, imbal_class_counts)]
    imbal_class_indices = np.hstack(imbal_class_indices)

    train_set.targets = targets[imbal_class_indices]
    train_set.data = train_set.data[imbal_class_indices]

    assert len(train_set.targets) == len(train_set.data)

    train_in_idx = get_oversampled_data(train_set, num_sample_per_class)
    train_in_loader = DataLoader(train_set, batch_size=batch_size,
                                 sampler=WeightedRandomSampler(train_in_idx, len(train_in_idx)), num_workers=8)
    ds.append(train_in_loader)

    test_set = instantiate_unknown_class(dataset_, root=DATA_ROOT, train=False, split='test', download=False, transform=TF_test)

    val_idx, test_idx = get_val_test_data(test_set, num_test_samples)
    val_loader = DataLoader(test_set, batch_size=100,
                            sampler=SubsetRandomSampler(val_idx), num_workers=8)
    test_loader = DataLoader(test_set, batch_size=100,
                             sampler=SubsetRandomSampler(test_idx), num_workers=8)
    ds.append(val_loader)
    ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds

    return ds


def get_imbalanced(dataset, num_sample_per_class, batch_size, TF_train, TF_test):
    print("Building CV {} data loader with {} workers".format(dataset, 8))
    ds = []

    dataset_, num_test_samples = get_class_nsamples(dataset)
    train_set = instantiate_unknown_class(dataset_, root=DATA_ROOT, train=True, split='train', download=True, transform=TF_train)

    train_in_idx = get_imbalanced_data(train_set, num_sample_per_class)
    train_in_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                  sampler=SubsetRandomSampler(train_in_idx), num_workers=8)
    ds.append(train_in_loader)

    test_set = instantiate_unknown_class(dataset_, root=DATA_ROOT, train=False, split='test', download=False, transform=TF_test)

    val_idx, test_idx= get_val_test_data(test_set, num_test_samples)
    val_loader = torch.utils.data.DataLoader(test_set, batch_size=100,
                                                  sampler=SubsetRandomSampler(val_idx), num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100,
                                                  sampler=SubsetRandomSampler(test_idx), num_workers=8)
    ds.append(val_loader)
    ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds

    return ds


def smote(data, targets, n_class, n_max):
    aug_data = []
    aug_label = []

    for k in range(1, n_class):
        indices = np.where(targets == k)[0]
        class_data = data[indices]
        class_len = len(indices)
        class_dist = np.zeros((class_len, class_len))

        # Augmentation with SMOTE ( k-nearest )
        if smote:
            for i in range(class_len):
                for j in range(class_len):
                    class_dist[i, j] = np.linalg.norm(class_data[i] - class_data[j])
            sorted_idx = np.argsort(class_dist)

            for i in range(n_max - class_len):
                lam = nr.uniform(0, 1)
                row_idx = i % class_len
                col_idx = int((i - row_idx) / class_len) % (class_len - 1)
                new_data = np.round(
                    lam * class_data[row_idx] + (1 - lam) * class_data[sorted_idx[row_idx, 1 + col_idx]])

                aug_data.append(new_data.astype('uint8'))
                aug_label.append(k)

    return np.array(aug_data), np.array(aug_label)


def get_smote(dataset,  num_sample_per_class, batch_size, TF_train, TF_test):
    print("Building CV {} data loader with {} workers".format(dataset, 8))
    ds = []

    dataset_, num_test_samples = get_class_nsamples(dataset)
    train_set = instantiate_unknown_class(dataset_, root=DATA_ROOT, train=True, split='train', download=False, transform=TF_train)

    try:
        targets = np.array(train_set.targets)
    except TypeError:
        targets = np.array(train_set.labels)

    classes, class_counts = np.unique(targets, return_counts=True)
    nb_classes = len(classes)

    imbal_class_counts = [int(i) for i in num_sample_per_class]
    class_indices = [np.where(targets == i)[0] for i in range(nb_classes)]

    imbal_class_indices = [class_idx[:class_count] for class_idx, class_count in zip(class_indices, imbal_class_counts)]
    imbal_class_indices = np.hstack(imbal_class_indices)

    train_set.targets = targets[imbal_class_indices]
    train_set.data = train_set.data[imbal_class_indices]

    assert len(train_set.targets) == len(train_set.data)

    class_max = max(num_sample_per_class)
    aug_data, aug_label = smote(train_set.data, train_set.targets, nb_classes, class_max)

    train_set.targets = np.concatenate((train_set.targets, aug_label), axis=0)
    train_set.data = np.concatenate((train_set.data, aug_data), axis=0)

    print("Augmented data num = {}".format(len(aug_label)))
    print(train_set.data.shape)

    train_in_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
    ds.append(train_in_loader)

    test_set = instantiate_unknown_class(dataset_, root=DATA_ROOT, train=False, split='test', download=False, transform=TF_test)

    val_idx, test_idx = get_val_test_data(test_set, num_test_samples)
    val_loader = torch.utils.data.DataLoader(test_set, batch_size=100,
                                             sampler=SubsetRandomSampler(val_idx), num_workers=8)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100,
                                              sampler=SubsetRandomSampler(test_idx), num_workers=8)
    ds.append(val_loader)
    ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds

    return ds