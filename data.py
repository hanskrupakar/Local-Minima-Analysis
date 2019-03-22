import torch
import torch.utils.data

from torchvision import datasets, transforms

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

def get_mnist_loaders(cuda_flag, dataset_path, batch_size=64, test_batch_size=1000):
   
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda_flag and torch.cuda.is_available() else {}
    
    train_loader = torch.utils.data.DataLoader(
                            datasets.MNIST(dataset_path, train=True, download=True,
                            transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                        ])),
                        batch_size=batch_size, shuffle=True, **kwargs)
    
    test_loader = torch.utils.data.DataLoader(
                        datasets.MNIST(dataset_path, train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
                        batch_size=test_batch_size, shuffle=True, **kwargs)
    
    logger.info('Using MNIST dataset for this experiment')
    return train_loader, test_loader

if __name__=='__main__':
    
    train, test = get_mnist_loaders(True, 'data/')
    logger.info (train, test)

    train, test = get_mnist_loaders(False, 'data/')
    logger.info (train, test)    
