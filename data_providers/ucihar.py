
import os
import numpy as np
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from data_providers.base_provider import *


class UCIHARDataProvider(DataProvider):
    multiplier = 4
    stem_multiplier = 3
    def __init__(self, data_path='.', save_path=None, train_batch_size=256, test_batch_size=256, valid_size=5000,
                 n_worker=0):

        self._save_path = save_path
        train_dataset, test_dataset, Xtest, Ytest = load_dataset(data_path + 'data/ucihar/')
        self.Xtest, self.Ytest = Xtest, Ytest
        target = train_dataset.tensors[1] 

        if valid_size is not None:
            if isinstance(valid_size, float):
                valid_size = int(valid_size * len(train_dataset))
            else:
                assert isinstance(valid_size, int), 'invalid valid_size: %s' % valid_size

            train_indexes, valid_indexes = self.random_sample_valid_set(
                train_dataset.tensors[1], valid_size, self.n_classes,
            )
            train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indexes)
            valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indexes)

            valid_dataset = train_dataset

            self.train = torch.utils.data.DataLoader(
                train_dataset, batch_size=train_batch_size, sampler=train_sampler,
                num_workers=n_worker, pin_memory=True,
            )
            self.valid = torch.utils.data.DataLoader(
                valid_dataset, batch_size=test_batch_size, sampler=valid_sampler,
                num_workers=n_worker, pin_memory=True,
            )
        else:
            self.train = torch.utils.data.DataLoader(
                train_dataset, batch_size=train_batch_size, shuffle=True,
                num_workers=n_worker, pin_memory=True,
            )
            self.valid = None

        self.test = torch.utils.data.DataLoader(
            test_dataset, batch_size=test_batch_size, shuffle=False,
            num_workers=n_worker, pin_memory=True,
        )

        if self.valid is None:
            self.valid = self.test

    @staticmethod
    def name():
        return 'ucihar'

    @property
    def data_shape(self):
        return 6, self.data_length  # C, length

    @property
    def n_classes(self):
        return 6

    @property
    def target_names(self):
        LABELS = [
        "WALKING",
        "WALKING_UPSTAIRS",
        "WALKING_DOWNSTAIRS",
        "SITTING",
        "STANDING",
        "LAYING"
        ]
        return LABELS

    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = './dataset/ucihar'
        return self._save_path

    @property
    def data_url(self):
        raise ValueError('unable to download UCI-HAR')

    @property
    def train_path(self):
        return os.path.join(self.save_path, 'train')

    @property
    def valid_path(self):
        return os.path.join(self._save_path, 'val')

    @property
    def normalize(self):
        return transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])

    @property
    def data_length(self):
        return 128