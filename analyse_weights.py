import torch
import glob

import numpy as np
import os

def get_models(models_path):  
    
    weights, params = [], []
    for i, f in enumerate(glob.glob(os.path.join(models_path, '*'))):

        name = f.split('/')[1]
        ord_files = sorted(glob.glob(os.path.join(f,'*')), key=lambda x: int(x.split('_')[-1][:-3]))

        network, network_params = [], []
        for j, epoch_W in enumerate(ord_files):

            model_dict = torch.load(epoch_W)
        

            loss, correct, total = model_dict['loss'], model_dict['correct'], model_dict['total']
            network_params.append([loss, correct, total])

            layers, layer_keys = [], []

            for key in model_dict['model']:
                if 'num_batches_tracked' not in key:
                    layer_keys.append(key)
                    layers.append(model_dict['model'][key])
            network.append(layers)
        params.append(network_params)
        weights.append(network)
    
    return weights, params, layer_keys 

def get_random_model_keys(weights):

    idx1, idx2 = 0, 0
    while idx1 == idx2:
        idx1, idx2 = np.random.randint(len(weights)), np.random.randint(len(weights))

    return idx1, idx2

abs_sum = lambda x: torch.sum(torch.abs(x)).item()

def display_stats(weights, params, layer_keys, idx1, idx2, epoch=-1): # Taking last epoch of checkpoints

    loss1, correct1, total1 = params[idx1][epoch]
    loss2, correct2, total2 = params[idx2][epoch]

    print ("Losses = %0.5f (%d/%d) ; %0.5f (%d/%d)"%(loss1, correct1, total1, loss2, correct2, total2))

    for key, w1, w2 in zip(layer_keys, weights[idx1][epoch], weights[idx2][epoch]):

        print ('\nLayer:', key)  
        print ('Sum of Absolute values of parameters =', abs_sum(w1), abs_sum(w2))
        print ('Sum of Absolute difference (|w1-w2|) =', abs_sum(w1-w2), '\n')

weights, params, layer_keys = get_models('models')
w1_idx, w2_idx = get_random_model_keys(weights)

display_stats(weights, params, layer_keys, w1_idx, w2_idx)
