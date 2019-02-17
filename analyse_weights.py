import torch
import glob

from pprint import pprint

weights = []

for i, f in enumerate(glob.glob('models/*')):
    
    name = f.split('/')[1]
    ord_files = sorted(glob.glob(f+'/*'), key=lambda x: int(x.split('_')[-1][:-3]))
    
    for j, epoch_W in enumerate(ord_files):
        
        model = torch.load(epoch_W)
        
        layers = []

        for key in model:
            if 'conv' in key and 'weight' in key:
                layers.append(model[key])

        weights.append(layers)

for w1, w2 in zip(weights[0], weights[1]):
    
    print (w1.size(), w2.size())
    print (torch.sum(torch.abs(w1-w2)))
