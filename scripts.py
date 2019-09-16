from minima_exploration.decay import LRDecay
from minima_exploration.losses import LossClass

import argparse
import torch
from torch import optim
import math

from minima_exploration.data import get_mnist_loaders, visualize_dataset
from minima_exploration.model import Net
from minima_exploration.main import Trainer
import torch.nn.functional as F
import copy

class Decay(LRDecay):
    def __init__(self, total_epochs, rate, init_lr=0.1):
        self.total_epochs = total_epochs
        self.rate = rate
        self.init_lr = init_lr

        self.flag = self.total_epochs // 2
        self.future_step = max(self.flag // 2, 2)

    def decay(self, optimizer, epoch):
        
        if epoch == self.flag and epoch != self.total_epochs-1:
            lr = max(self.init_lr * self.rate, 1e-5)
            self.flag += self.future_step
            self.future_step = max(self.future_step//2, 2)
            
            logger.info ("Reducing learning rate from %0.7f to %0.7f"%(self.init_lr, lr))
            self.init_lr = lr
            
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        return optimizer

class CrossEntropy(LossClass):
    
    def __init__(self, trainer, name):
        self.trainer = trainer
        trainer.print_loss_name(name)

    def loss_fn(self):
        return F.cross_entropy
    
    def loss_params(self, x, y, y_pred):
        return (y_pred, y), {} 


ap = argparse.ArgumentParser(description='Interface for comparison of vectors of weight matrices of several local minima')
    
ap.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
ap.add_argument('--test-batch-size', type=int, default=1000, help='input batch size for testing (default: 1000)')
ap.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 10)')
ap.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
ap.add_argument('--cpu', action='store_true', default=False, help='disables CUDA training')
ap.add_argument('--log-interval', type=int, default=10, help='how many batches to wait before logging training status')
ap.add_argument('--name', default='mnist', help='Name of dataset to store models in folder')
args = ap.parse_args()
    
device = torch.device("cuda" if not args.cpu and torch.cuda.is_available() else "cpu")
    
data_loaders = get_mnist_loaders(cuda_flag=not args.cpu, dataset_path='data/', rotation_aug=False,
                                 batch_size=args.batch_size, test_batch_size=args.test_batch_size)   

rotation_data_loaders = get_mnist_loaders(cuda_flag=not args.cpu, dataset_path='data/', rotation_aug=True,
                                 batch_size=args.batch_size, test_batch_size=args.test_batch_size)   

#visualize_dataset(data_loaders[1])
#visualize_dataset(rotation_data_loaders[1])

model = Net()
model_rot = copy.deepcopy(model)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.7)
optimizer_rot = optim.SGD(model_rot.parameters(), lr=args.lr, momentum=0.7)

lr_decayer = Decay(args.epochs, 0.25, args.lr)
    
trainer = Trainer(model, optimizer, data_loaders, args.epochs, \
                  log_every=10, validate=True, gpu=0, lr_decayer=None, name='mnist1')

trainer_rot = Trainer(model_rot, optimizer_rot, rotation_data_loaders, args.epochs, \
                  log_every=10, validate=True, gpu=1, lr_decayer=None, name='mnist_rot1')

loss_obj1 = CrossEntropy(trainer, 'cross_entropy')   
trainer.add_loss(loss_obj1)

trainer.train()

trainer_rot.add_loss(loss_obj1)
trainer_rot.train()

print ("BEFORE AUGMENTATION:")
trainer.test()
trainer.test(data_loader=rotation_data_loaders[1], ntimes=10)
print ("AFTER AUGMENTATION:")
trainer_rot.test(data_loader=data_loaders[1])
trainer_rot.test(ntimes=10)

class RotationInvariance(LossClass):
    
    def __init__(self, trainer, angle, l_rot):
        self.trainer = trainer
        self.angle = angle
        self.l_rot = l_rot
        
        self.trainer.print_loss_name('rotation_inv(angle:%f)'%(angle))

    def loss_params(self, x, y, y_pred):

        gamma = torch.tensor([self.angle], dtype=torch.float,requires_grad=False).to(self.trainer.device)
        cg = torch.cos(gamma)
        sg = torch.sin(gamma)
        zero = torch.tensor([0.]).to(self.trainer.device)
        Mt = torch.reshape(torch.cat([cg,sg,zero,-sg,cg,zero]), (2,3)).expand(x.size(0),2,3)
        grid_rot = F.affine_grid(Mt, x.size())
        rotated_batch = F.grid_sample(x, grid_rot)
        rot_pred = self.trainer.model(rotated_batch)

        return (self.l_rot*y_pred, self.l_rot*rot_pred), {}

    def loss_fn(self):
        return F.mse_loss

angle = math.pi / 8
while angle <= math.pi/2:
    loss_obj2 = RotationInvariance(trainer, -angle, 1/10)
    trainer.add_loss(loss_obj2)
    
    loss_obj3 = RotationInvariance(trainer, angle, 1/10)
    trainer.add_loss(loss_obj3)

    angle *= 2

trainer.train()

print ("AFTER MINIMA EXPLORATION:")
trainer.test()
trainer.test(data_loader=rotation_data_loaders[1], ntimes=10)
