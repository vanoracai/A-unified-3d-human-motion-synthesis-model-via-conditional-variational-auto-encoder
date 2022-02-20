from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.utils.data
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
import importlib

def get_dataset_moudule_by_name(dataset_name):
    model_file_name = "dataloader." + 'data_' + dataset_name + "_loader"
    modellib = importlib.import_module(model_file_name)
    dataset = None
    for name, cls in modellib.__dict__.items():
        if name.lower() == dataset_name.lower():
            dataset = cls
    if dataset is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (model_file_name, dataset_name))
        exit(0)
    return dataset

class CustomDatasetDataLoader(object):
    @property
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt, dataset_class, train=True):
        self.opt = opt
        # load dataset module using name
        self.dataset = dataset_class
        # dataset = get_dataset_moudule_by_name(opt.dataset_name)
        # self.dataset = dataset(opt, train)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=train,
            num_workers=int(opt.nThreads),
            )
        return self.dataset, self.dataloader

    # def load_data(self):
    #     return self.dataloader

    # def shuffle_data(self):
    #     self.dataset.shuffle_data()

    def __len__(self):
        return len(self.dataset)


class DistributedDataLoader(object):
    def initialize(self, opt, dataset_class,train=True):
        # print("Use distributed dataloader")

        self.dataset = dataset_class
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        self.train_sampler = DistributedSampler(self.dataset, world_size, rank)

        num_workers = opt.nThreads
        # assert opt.batchSize % world_size == 0
        if train:
            batch_size = opt.batchSize #// world_size
        else:
            batch_size = opt.test_batchSize
        shuffle = False
        drop_last = train

        self.data_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            sampler=self.train_sampler,
            )
        return self.dataset, self.data_loader

    # def load_data(self):
    #     return self.data_loader
    #
    # def shuffle_data(self):
    #     self.dataset.shuffle_data()

    def __len__(self):
        return len(self.dataset)


def CreateDataLoader(opt, train=True, actions=None):
    dataset = get_dataset_moudule_by_name(opt.dataset_name)
    dataset_class = dataset(opt, train, actions)
    if opt.dist:
        data_loader = DistributedDataLoader()
    else:
        data_loader = CustomDatasetDataLoader()
    dataset, dataloader = data_loader.initialize(opt, dataset_class,train)
    return dataset, dataloader

