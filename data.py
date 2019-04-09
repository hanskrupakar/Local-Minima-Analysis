import torch
import torch.utils.data
import numpy as np

from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

import logging
logging.basicConfig(format='%(name)s %(asctime)s %(levelname)s %(message)s',
                    filename='log.txt', filemode='a', level=logging.INFO)
logger = logging.getLogger(__name__)

import sys
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)s %(asctime)s %(levelname)s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def get_mnist_loaders(cuda_flag, dataset_path, val=False, validation_size=5000, batch_size=64, test_batch_size=1000):
   
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda_flag and torch.cuda.is_available() else {}
    
    train_dataset = datasets.MNIST(dataset_path, train=True, download=True,
                        transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))]))

    indices = list(range(len(train_dataset)))
    np.random.shuffle(indices)
    
    if not val:
        ptr = len(train_dataset)
    else:
        ptr = len(train_dataset) - validation_size

    train_sampler, val_sampler = SubsetRandomSampler(indices[:ptr]), SubsetRandomSampler(indices[ptr:])

    train_loader = torch.utils.data.DataLoader(train_dataset,
                        batch_size=batch_size, sampler=train_sampler, **kwargs)

    if val:
        val_loader = torch.utils.data.DataLoader(train_dataset,
                            batch_size=batch_size, sampler=val_sampler, **kwargs)
    
    test_loader = torch.utils.data.DataLoader(
                        datasets.MNIST(dataset_path, train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
                        batch_size=test_batch_size, shuffle=True, **kwargs)
    
    logger.info('Using MNIST dataset for this experiment')

    if not val:
        return train_loader, test_loader
    else:
        return train_loader, val_loader, test_loader
if __name__=='__main__':
    
    train, test = get_mnist_loaders(True, 'data/')
    logger.info (train, test)

    train, test = get_mnist_loaders(False, 'data/')
    logger.info (train, test)    
