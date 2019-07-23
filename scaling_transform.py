from main import load_model, test
from data import get_mnist_loaders
from model import Net

import argparse 
import torch

def scale_by_constant(model, const, layer=1):
    
    assert layer < len(model.conv_layers)

    # sanity check
    # print (torch.sum(model.conv_layers[layer].conv1.weight), torch.sum(model.conv_layers[layer].conv1.bias))
    
    model.conv_layers[layer].scale(const)
    
    # sanity check
    # print (torch.sum(model.conv_layers[layer].conv1.weight), torch.sum(model.conv_layers[layer].conv1.bias))

if __name__ == '__main__':
        
    ap = argparse.ArgumentParser()
    ap.add_argument('model', help='Path to model weights file to load')
    ap.add_argument('factor', type=float, help='Factor to use for scalar multiplication')
    ap.add_argument('layer', type=int, nargs='+', help='Layer (and layer+1) of network to apply transforms * N followed by * 1/N')
    args = ap.parse_args()
    
    devc = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(devc)

    modelw, _, _, _ = load_model(args.model, cpu = devc == "cpu")
    
    model = Net()
    model.load_state_dict(modelw)
    model = model.to(device)

    _, test_loader = get_mnist_loaders(cuda_flag= devc != "cpu", dataset_path='data/', 
                                       batch_size=4, test_batch_size=100)
    
    loss, correct = test(model, test_loader, device)
    print ("Number of correct predictions before scaling: %d"%(correct))
    
    for layer in args.layer:
        scale_by_constant(model, args.factor, layer)

    loss, correct = test(model, test_loader, device)
    print ("Number of correct predictions after scaling: %d"%(correct))
