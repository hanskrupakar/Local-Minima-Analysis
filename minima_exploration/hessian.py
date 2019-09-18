import torch
from main import load_model
from data import get_mnist_loaders
from model import Net
import argparse
import os
import glob

import matplotlib.pyplot as plt

def gradient(**kwargs):
    
    outputs = kwargs['outputs']
    inputs = kwargs['inputs']
    grad_outputs= None if "grad_outputs" not in kwargs else kwargs['grad_outputs']
    retain_graph= None if "retain_graph" not in kwargs else  kwargs['retain_graph']
    create_graph= None if "create_graph" not in kwargs else  kwargs['create_graph']

    if torch.is_tensor(inputs):
        inputs = [inputs]
    else:
        inputs = list(inputs)
    
    grads = torch.autograd.grad(outputs, inputs, grad_outputs,
                                allow_unused=True,
                                retain_graph=retain_graph,
                                create_graph=create_graph)
    grads = [x if x is not None else torch.zeros_like(y) for x, y in zip(grads, inputs)]
    
    return torch.cat([x.contiguous().view(-1) for x in grads])

def hessian(output, inputs, out=None, allow_unused=False, create_graph=False):
   
    assert output.ndimension() == 0

    if torch.is_tensor(inputs):
        inputs = [inputs]
    else:
        inputs = list(inputs)

    n = sum(p.numel() for p in inputs)
    if out is None:
        out = output.new_zeros(n, n)

    ai = 0
    for i, inp in enumerate(inputs):
        [grad] = torch.autograd.grad(output, inp, create_graph=True, allow_unused=allow_unused)
        grad = torch.zeros_like(inp) if grad is None else grad
        grad = grad.contiguous().view(-1)

        for j in range(inp.numel()):
            if grad[j].requires_grad:
                row = gradient(outputs=grad[j], inputs=inputs[i:], retain_graph=True, create_graph=create_graph)[j:]
            else:
                row = grad[j].new_zeros(sum(x.numel() for x in inputs[i:]) - j)

            out[ai, ai:].add_(row.type_as(out))  # ai's row
            if ai + 1 < n:
                out[ai + 1:, ai].add_(row[1:].type_as(out))  # ai's column
            del row
            ai += 1
        del grad

    return out

if __name__=='__main__':
        
    ap = argparse.ArgumentParser()
    ap.add_argument('--models_dir', help='Path to model checkpoints', default='models/mnist/')
    args = ap.parse_args()

    cuda_flag = torch.cuda.is_available()
    device = torch.device('cuda' if cuda_flag else 'cpu')
    
    _, test_loader = get_mnist_loaders(cuda_flag=cuda_flag, dataset_path='data/', 
                                           batch_size=4, test_batch_size=5,
                                           num_test_images=10)


    zero_limit = 1e-5

    for f in glob.glob(os.path.join(args.models_dir, '*')):
        
        modelw, _, _, _ = load_model(f, cpu = not cuda_flag)    
        model = Net()
        model.load_state_dict(modelw)
        model = model.to(device)
        
        x, y = next(iter(test_loader))
        x, y = x.to(device), y.to(device)
        l = torch.nn.functional.cross_entropy(model(x), y)
        
        for n, p in model.named_parameters():
            hess = hessian(l, p)
            eigvals, eigvecs = torch.eig(hess)
            
            ctx = 0
            for e in eigvals:
                if e[0] < zero_limit and e[1] < zero_limit:
                   ctx += 1

            print (eigvals.size(0), ctx)
