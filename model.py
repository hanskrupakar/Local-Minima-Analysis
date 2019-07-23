from torch.nn import Module
from torch.nn import Conv2d, Linear, BatchNorm2d
from torch.nn import functional as F

from torch.nn import ModuleList
import torch

class Net(Module):

    def __init__(self, input_size=28, input_channels=1):
        
        super(Net, self).__init__()

        self.input_size = input_size

        self.conv_params = [{'in_channels': input_channels, 
                             'out_channels': 32, 
                             'kernel_size': 3, 
                             'stride': 1, 
                             'padding': 0,
                             'bias': True},
                            
                            {'in_channels': 32, 
                             'out_channels': 128, 
                             'kernel_size': 3, 
                             'stride': 1, 
                             'padding': 0,
                             'bias': True},
                             
                             {'in_channels': 128, 
                             'out_channels': 64, 
                             'kernel_size': 3, 
                             'stride': 1, 
                             'padding': 0,
                             'bias': True},
                             
                             {'in_channels': 64, 
                             'out_channels': 10, 
                             'kernel_size': 3, 
                             'stride': 1, 
                             'padding': 0,
                             'bias': True},
                             ]
 
        self.conv_layers = []
        
        self.size = self.input_size

        for p in self.conv_params:
            self.conv_layers.append(conv_block(**p))
            self.size = self.calculate_output_size(self.size, p['kernel_size'], p['stride'], p['padding'])
            self.size = self.calculate_output_size(self.size, p['kernel_size'], p['stride'], p['padding'])
        
        self.conv_layers = ModuleList(self.conv_layers)

        self.fc1 = Linear(self.size*self.size*self.conv_params[-1]['out_channels'], 500)
        self.fc2 = Linear(500, 10)

    def calculate_output_size(self, input_size, kernel_size, stride=1, padding=0, typ='conv'):
        
        if typ == 'conv':
            return int((input_size + 2*padding - kernel_size)/stride) + 1
        elif typ == 'pool':
            return int(input_size/kernel_size)

    def forward(self, x):
        conv_buffer = x

        for conv in self.conv_layers:
            conv_buffer = conv(conv_buffer)

        conv_buffer = conv_buffer.view(-1, self.size*self.size*self.conv_params[-1]['out_channels'])

        fc1 = self.fc1(conv_buffer)
        fc2 = self.fc2(fc1)

        return fc2

class conv_block(Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        
        super(conv_block, self).__init__()

        self.conv1 = Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        #self.bn1 = BatchNorm2d(out_channels)
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=bias)
        #self.bn2 = BatchNorm2d(out_channels)
    
    def forward(self, x):
        
        c1 = self.conv1(x)
        #c1_b = self.bn1(c1)
        c1_ba = F.relu(c1)

        c2 = self.conv2(c1_ba)
        #c2_b = self.bn2(c2)
        c2_ba = F.relu(c2)

        return c2_ba
    
    def scale(self, const=2):
        
        with torch.no_grad():
            self.conv1.weight.masked_scatter_(
                    torch.ones(self.conv1.weight.size(), device=self.conv1.weight.device, dtype=torch.uint8), 
                    self.conv1.weight*const)
            #self.conv1.bias.masked_scatter_(
            #        torch.ones(self.conv1.bias.size(), device=self.conv1.bias.device, dtype=torch.uint8), 
            #        self.conv1.bias*const)

            self.conv2.weight.masked_scatter_(
                    torch.ones(self.conv2.weight.size(), device=self.conv2.weight.device, dtype=torch.uint8), 
                    self.conv2.weight*(1.0/const))
            #self.conv2.bias.masked_scatter_(
            #        torch.ones(self.conv2.bias.size(), device=self.conv2.bias.device, dtype=torch.uint8), 
            #        self.conv2.bias*(1.0/const))

if __name__=='__main__':
    
    net = Net()

    import torch
    x = torch.ones(20, 1, 28,28)

    y = net.forward(x)

    print (y.size())
