from .model import Net
from .data import get_mnist_loaders
from torch.nn import functional as F
import argparse
import torch
from torch import optim
import os

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

from abc import ABCMeta, abstractmethod
from .losses import LossClass
from .decay import LRDecay

class Trainer:    
    
    def __init__(self, model, optimizer, data_loaders, total_epochs, \
                 log_every=10, validate=True, gpu=None, lr_decayer=None, name='mnist1'):
    
        '''
    model: Pytorch Module that takes x and gives y_pred
    optimizer: Gradient descent optimization algorithm used (as Pytorch optimizer)
    data_loaders: (train_data_loader, test_data_loader) set (Pytorch DataLoaders)
    total_epochs: Total number of epochs to run training
    log_every: Number of batches for a single entry in training log
    validate: Will test on test_data_loader every epoch if set as True
    cpu: Will train on CPU if set True
    lr_decayer: Learning Rate decayer object with decay() method that takes optimizer and epoch as arguments 
                and returns an optimizer with reduced lr or otherwise for every epoch
    name: Unique name of folder to save model-specific weights
        '''
   
        self.model = model
        self.optimizer = optimizer
        self.train_loader, self.test_loader = data_loaders
        self.total_epochs = total_epochs

        self.log_every = log_every
        self.validate = validate
        self.gpu = gpu
        self.lr_decayer = lr_decayer

        if self.lr_decayer is not None:
            assert isinstance(self.lr_decayer, LRDecay)
        
        self.name = name
        
        self.loss_classes = []
        self.prepare()

    def prepare(self):
        
        if self.gpu is None:
            dev = 'cpu'
        elif isinstance(self.gpu, int):
            dev = 'cuda:%d'%(self.gpu)
        elif isinstance(self.gpu, (list, tuple)):
            dev = 'cuda'

        self.device = torch.device(dev)
        self.model = self.model.to(self.device)
        
        if not os.path.isdir('models/%s'%self.name):
            os.makedirs('models/%s'%self.name)

    def add_loss(self, loss_class):
        
        assert isinstance(loss_class, LossClass), 'Add LossClass subclasses only'
        self.loss_classes.append(loss_class)

    def train_epoch(self, epoch):
    
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)

            loss = self.compose_losses(data, target, output)
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.log_every == 0:
                logger.info('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader), loss.item()))
    
    def save_model(self, loss, correct, total, fname):

        save_dict = dict()
        save_dict['model'] = self.model.state_dict()
        save_dict['loss'] = loss
        save_dict['correct'] = correct
        save_dict['total'] = total 

        torch.save(save_dict, fname)

    def load_model(self, path, stats=None, weight=False):
    
        if self.gpu is None:
            if not weight:
                model_dict = torch.load(path, map_location='cpu')
            else:
                model = Net()
                model.load_state_dict(path)
        else:
            if not weight:
                model_dict = torch.load(path)
            else:
                model = Net()
                model.load_state_dict(path)
        
        if not weight:
            model, loss, correct, total = model_dict['model'], model_dict['loss'], model_dict['correct'], model_dict['total']
            return model, loss, correct, total
        else:
            return model

    def print_loss_name(self, name):        
        print ('Using loss: %s'%(name))

    def compose_losses(self, x, y, y_pred, test=False):
       
        losses = []
        for loss_obj in self.loss_classes: 

            data_out = loss_obj.loss_params(x, y, y_pred)

            if test:
                data_out[1]['reduction'] = 'sum'
            
            losses.append(loss_obj.loss_fn()(*data_out[0], **data_out[1]))

        loss = torch.stack(losses, dim=0).mean()
        return loss
    
    def test(self, data_loader=None, log=True, ntimes=None):

        self.model.eval()
        
        test_loss = 0
        correct = 0

        DL = self.test_loader if data_loader is None else data_loader

        ntimes = 1 if ntimes is None else ntimes

        with torch.no_grad():
            for _ in range(ntimes):
                for data, target in DL:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    test_loss += self.compose_losses(data, target, output, test=True)
                    pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                    correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= float(len(self.test_loader.dataset))*ntimes
        correct /= ntimes

        if log:
            logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
                test_loss, correct, len(self.test_loader.dataset),
                100. * correct / float(len(self.test_loader.dataset))))

        return test_loss, correct

    def train(self):
         
        assert len(self.loss_classes) > 0, 'Use at least one loss function by using the subclass'
        self.model.train()
   
        for epoch in range(1, self.total_epochs + 1):
            
            self.train_epoch(epoch)
            
            if self.validate:
                loss, correct = self.test()
            else:
                loss, correct = 0, 0 
        
            self.save_model(loss, correct, len(self.test_loader.dataset), "models/%s/%s_%d.pt"%(self.name, self.name, epoch))
        
            if self.lr_decayer is not None:
                self.optimizer = self.lr_decayer.decay(self.optimizer, epoch)

if __name__=='__main__':
    
    #loss_fns = [F.cross_entropy]
    #loss_params = [lambda x, y, y_pred: ((y_pred, y), {})]
    
    # Applying invariance by explicit difference
    #def train_rotation_inv_ms(model, t_loader, angle=0., l_par=1, l_rate=0.001, num_epochs=10, device=DEVICE, l_rot=1):

    #loss_fns.append(F.mse_loss)
    
    '''
    def rotation_inv_diff(x, y, y_pred, **kwargs):
        
        print (kwargs)

        gamma = torch.tensor([kwargs['angle']], dtype=torch.float,requires_grad=False).to(kwargs['device'])
        cg = torch.cos(gamma)
        sg = torch.sin(gamma)
        zero = torch.tensor([0.]).to(device)
        Mt = torch.reshape(torch.cat([cg,sg,zero,-sg,cg,zero]), (2,3)).expand(x.size(0),2,3)
        grid_rot = F.affine_grid(Mt, x.size())
        rotated_batch = F.grid_sample(img_batch, grid_rot)
        rot_pred = kwargs['model'](rotated_batch)

        return (kwargs['l_rot']*y_pred, kwargs['l_rot']*rot_pred)
    '''
    #rot_kwargs = {'angle': 0, 'model': model, 'device': device, 'l_rot': 1}
    
    #loss_params.append(rotation_inv_diff)
    #loss_kwargs = (None, rot_kwargs)
    
    #run_training(model, optimizer, data_loaders, args.epochs, loss_fns, loss_params, loss_kwargs, lr_decayer=lr_decayer, name=args.name)
