import torch
import glob

import numpy as np
import os

import logging
logging.basicConfig(level = logging.INFO, filename='log.txt', filemode='a', format='%(name)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

import sys
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)s %(asctime)s %(levelname)s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class TrainedModels:
    
    def __init__(self, models_dir, pytorch_model=False):
        self.models_dir = models_dir
        self.pytorch_model = pytorch_model

    def get_models(self, cpu=False):  
        
        weights, params = [], []
        for i, f in enumerate(glob.glob(os.path.join(self.models_dir, '*'))):

            name = f.split('/')[1]
            ord_files = sorted(glob.glob(os.path.join(f,'*')), key=lambda x: int(x.split('_')[-1][:-3]))

            network, network_params = [], []
            for j, epoch_W in enumerate(ord_files):

                if cpu:
                    model_dict = torch.load(epoch_W, map_location='cpu')
                else:
                    model_dict = torch.load(epoch_W)

                loss, correct, total = model_dict['loss'], model_dict['correct'], model_dict['total']
                network_params.append([loss, correct, total])

                layers, layer_keys = [], []
                
                if self.pytorch_model:
                    model = model_dict['model']
                    network.append(model)
                    layer_keys = model.keys()
                else:
                    for key in model_dict['model']:
                        if 'num_batches_tracked' not in key:
                            layer_keys.append(key)
                            layers.append(model_dict['model'][key])
                    network.append(layers)

            params.append(network_params)
            weights.append(network)
    
        return weights, params, layer_keys 

    def get_random_model_keys(self, weights):

        idx1, idx2 = 0, 0
        
        while idx1 == idx2:
            idx1, idx2 = np.random.randint(len(weights)), np.random.randint(len(weights))

        return idx1, idx2

    def display_stats(self, weights, params, layer_keys, idx1, idx2, epoch=-1): # Taking last epoch of checkpoints

        abs_sum = lambda x: torch.sum(torch.abs(x)).item()

        loss1, correct1, total1 = params[idx1][epoch]
        loss2, correct2, total2 = params[idx2][epoch]

        logger.info ("Losses = %0.5f (%d/%d) ; %0.5f (%d/%d)"%(loss1, correct1, total1, loss2, correct2, total2))

        for key, w1, w2 in zip(layer_keys, weights[idx1][epoch], weights[idx2][epoch]):

            logger.info ("\nLayer: %s"% (key))  
            
            if not self.pytorch_model:
                mod_w1, mod_w2 = abs_sum(w1), abs_sum(w2)
                mod_del = abs_sum(w1-w2)
            else:
                we1, we2 = weights[idx1][epoch][w1], weights[idx2][epoch][w2]
                mod_w1, mod_w2 = abs_sum(we1), abs_sum(we2)
                mod_del = abs_sum(we1-we2)
            
            logger.info ("Sum of Absolute values of parameters = %0.5f %0.5f"%(mod_w1, mod_w2))
            logger.info ("Sum of Absolute difference (|w1-w2|) = %0.5f \n"%(mod_del))
            
if __name__=='__main__':
    
    model_handler = TrainedModels('models', True)

    weights, params, layer_keys = model_handler.get_models()

    w1_idx, w2_idx = model_handler.get_random_model_keys(weights)

    model_handler.display_stats(weights, params, layer_keys, w1_idx, w2_idx)
