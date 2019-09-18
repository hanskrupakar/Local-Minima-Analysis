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

from matplotlib import pyplot as plt
import PIL

def get_mnist_loaders(cuda_flag, dataset_path, val=False, rotation_aug=False, validation_size=5000, batch_size=64, test_batch_size=1000, num_test_images=10000):
   
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda_flag and torch.cuda.is_available() else {}
    
    T = [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))]

    if rotation_aug:
        T.insert(0, transforms.RandomRotation(90, resample=PIL.Image.BILINEAR))

    train_dataset = datasets.MNIST(dataset_path, train=True, download=True, transform=transforms.Compose(T))

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
    
    test_dataset = datasets.MNIST(dataset_path, train=False, transform=transforms.Compose(T))
    
    test_indices = list(range(len(test_dataset)))

    assert num_test_images <= len(test_dataset)

    test_sampler = SubsetRandomSampler(test_indices[:num_test_images])
    test_loader = torch.utils.data.DataLoader(
                        test_dataset,
                        batch_size=test_batch_size, sampler=test_sampler, **kwargs)
    
    logger.info('Using MNIST dataset for this experiment')

    if not val:
        return train_loader, test_loader
    else:
        return train_loader, val_loader, test_loader

def visualize_dataset(data, gray=True):
    
    img, _ = next(iter(data))
    fig, ax = plt.subplots(len(img))
    if gray:
        for idx in range(len(img)):
            ax[idx].imshow(img.numpy()[idx,0,:,:])
        plt.show()

def get_cifar10_loaders(cuda_flag, dataset_path, val=False, validation_size=5000, batch_size=64, test_batch_size=1000):
    
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda_flag and torch.cuda.is_available() else {}
    
    train_dataset = datasets.CIFAR10(dataset_path, train=True, download=True,
                        transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]))

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
                        datasets.CIFAR10(dataset_path, train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                        ])),
                        batch_size=test_batch_size, shuffle=True, **kwargs)
    
    logger.info('Using CIFAR10 dataset for this experiment')

    if not val:
        return train_loader, test_loader
    else:
        return train_loader, val_loader, test_loader

def get_svhn_loaders(cuda_flag, dataset_path, val=False, validation_size=5000, batch_size=64, test_batch_size=1000):
    
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda_flag and torch.cuda.is_available() else {}
    
    train_dataset = datasets.SVHN(dataset_path, split='train' if not val else 'test', download=True,
                        transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

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
                        datasets.CIFAR10(dataset_path, train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                        ])),
                        batch_size=test_batch_size, shuffle=True, **kwargs)
    
    logger.info('Using SVHN dataset for this experiment')

    if not val:
        return train_loader, test_loader
    else:
        return train_loader, val_loader, test_loader


if __name__=='__main__':
    
    import cv2

    train, test = get_svhn_loaders(True, 'data/')
    print (train, test)
    
    for img, lab in train:
        print (lab[0])
        image = (img[0]*255).numpy().astype(np.uint8).transpose((1,2,0))
        cv2.imshow('f', image); cv2.waitKey()

    train, test = get_cifar10_loaders(False, 'data/')
    print (train, test)    
