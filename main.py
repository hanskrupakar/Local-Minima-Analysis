from model import Net
from data import get_mnist_loaders
from torch.nn import functional as F
import argparse
import torch
from torch import optim
import os

class Decay:
    def __init__(self, total_epochs, rate, init_lr=0.1):
        self.total_epochs = total_epochs
        self.rate = rate
        self.init_lr = init_lr

        self.flag = self.total_epochs // 2

    def decay(self, optimizer, epoch):
        
        if epoch == self.flag and epoch != self.total_epochs-1:
            lr = max(self.init_lr / self.rate, 1e-5)
            self.flag += self.flag//2
            
            print ("Reducing learning rate from %0.7f to %0.7f"%(self.init_lr, lr))
            self.init_lr = lr

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

def train(model, train_loader, optimizer, epoch, device, log_interval=10):
    
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, test_loader, device):

    model.eval()
    
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

if __name__=='__main__':
    
    ap = argparse.ArgumentParser(description='Interface for comparison of vectors of weight matrices of several local minima')
    
    ap.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    ap.add_argument('--test-batch-size', type=int, default=1000, help='input batch size for testing (default: 1000)')
    ap.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 10)')
    ap.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    ap.add_argument('--cpu', action='store_true', default=False, help='disables CUDA training')
    ap.add_argument('--log-interval', type=int, default=10, help='how many batches to wait before logging training status')
    ap.add_argument('--name', default='mnist', help='Name of dataset to store models in folder')
    args = ap.parse_args()
    
    device = torch.device("cuda:0" if not args.cpu and torch.cuda.is_available() else "cpu")

    model = Net()
    
    lr_decayer = Decay(args.epochs, 0.25, args.lr)

    if not args.cpu and torch.cuda.is_available():
        model = model.to(device)
        model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.7)
    
    train_loader, test_loader = get_mnist_loaders(cuda_flag=not args.cpu, dataset_path='data/', 
                                    batch_size=args.batch_size, test_batch_size=args.test_batch_size)

    if not os.path.isdir('models'):
        os.mkdir('models')

    if not os.path.isdir('models/%s'%args.name):
        os.mkdir('models/%s'%args.name)

    for epoch in range(1, args.epochs + 1):

        train(model, train_loader, optimizer, epoch, device, args.log_interval)
        test(model, test_loader, device)

        lr_decayer.decay(optimizer, epoch)

        torch.save(model.state_dict(),"models/%s/%s_%d.pt"%(args.name, args.name, epoch))
