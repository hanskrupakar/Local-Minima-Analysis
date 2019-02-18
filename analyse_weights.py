import torch
import glob

from pprint import pprint
import numpy as np

weights = []

for i, f in enumerate(glob.glob('models/*')):
    
    name = f.split('/')[1]
    ord_files = sorted(glob.glob(f+'/*'), key=lambda x: int(x.split('_')[-1][:-3]))
    
    network, network_params = [], []
    for j, epoch_W in enumerate(ord_files):
        
        model_dict = torch.load(epoch_W)
        
        model = model_dict['model']
        loss, correct, total = model_dict['loss'], model_dict['correct'], model_dict['total']
        network_params.append([loss, correct, total])

        layers, layer_keys = [], []

        for key in model:
            if 'num_batches_tracked' not in key:
                layer_keys.append(key)
                layers.append(model[key])

        network.append(layers)
    weights.append(network)

abs_sum = lambda x: torch.sum(torch.abs(x)).item()

idx1, idx2 = 0, 0
while idx1 == idx2:
    idx1, idx2 = np.random.randint(len(weights)), np.random.randint(len(weights))

for key, w1, w2 in zip(layer_keys, weights[idx1][-1], weights[idx2][-1]):
    
    print (key, ': SHAPE=', list(w1.size()))
    print ('Losses = %0.6f (%d/%d)  %0.6f (%d/%d)'%(*network_params[idx1], *network_params[idx2]))
    print ('Sum of Absolute values of parameters =', abs_sum(w1), abs_sum(w2)) 
    prodofsizes = np.prod(list(w1.size()))
    print ('Sum of Absolute value difference (|w1-w2|) = ', abs_sum(w1-w2))
    print ('\n')
