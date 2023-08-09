
import os
import numpy as np
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from data_providers.base_provider import *


class OPPDataProvider(DataProvider):
    multiplier = 4
    stem_multiplier = 3
    def __init__(self, data_path='.', save_path=None, train_batch_size=256, test_batch_size=256, valid_size=None,
                 n_worker=0):

        self._save_path = save_path
        train_dataset, test_dataset, Xtest, Ytest = load_dataset(data_path + 'data/opportunity/')
        Xval = np.load(data_path + 'data/opportunity/val_x.npy')
        Yval = np.load(data_path + 'data/opportunity/val_y.npy')
        valid_dataset = TensorDataset(torch.tensor(Xval).float(), 
                                    torch.tensor(Yval).long())

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
            self.valid = torch.utils.data.DataLoader(
                valid_dataset, batch_size=test_batch_size, shuffle=False,
                num_workers=n_worker, pin_memory=True,
            )

        self.test = torch.utils.data.DataLoader(
            test_dataset, batch_size=test_batch_size, shuffle=False,
            num_workers=n_worker, pin_memory=True,
        )

    @staticmethod
    def name():
        return 'opportunity'

    @property
    def data_shape(self):
        return 113, self.data_length  # C, length

    @property
    def n_classes(self):
        return 18 # 18 with null class

    @property
    def target_names(self):
        LABELS = [
            'Null',
            'Open Door 1',
            'Open Door 2',
            'Close Door 1',
            'Close Door 2',
            'Open Fridge',
            'Close Fridge',
            'Open Dishwasher',
            'Close Dishwasher',
            'Open Drawer 1',
            'Close Drawer 1',
            'Open Drawer 2',
            'Close Drawer 2',
            'Open Drawer 3',
            'Close Drawer 3',
            'Clean Table',
            'Drink from Cup',
            'Toggle Switch'
        ]
        return LABELS

    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = './dataset/opportunity'
        return self._save_path

    @property
    def data_url(self):
        raise ValueError('unable to download Opportunity')

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
        return 16

if __name__ == "__main__":
    dataset = OPPDataProvider()
    dataset.data_shape
    for input, target in dataset.train:
        input, target
        break