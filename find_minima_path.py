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

from analyse_weights import TrainedModels
import numpy as np

class PathFinder:
    
    # model_handler: see analyse_weights.py
    # nongradient_steps: number of divisions between w1 and w2 defining step size for non-gradient update towards better w
    # del_w_limit: Minimum difference between the weight vectors to determine if they aren't the same w point
    # del_loss_limit: Maximum difference between the loss values to consider the weights to belong to a common path
    # epoch: Epoch number to take as converged w for w1 and w2
    def __init__(self, model_handler, nongradient_steps, del_w_limit, del_loss_limit, epoch=-1):  
        
        weights, params, layer_keys = model_handler.get_models()
        w1_idx, w2_idx = model_handler.get_random_model_keys(weights)
        
        params1 = params[w1_idx][epoch]
        params2 = params[w2_idx][epoch]
        
        w1 = {'weight': weights[w1_idx][epoch], 'loss': params1[0], 'correct': params1[1], 'total': params1[2]} 
        w2 = {'weight': weights[w2_idx][epoch], 'loss': params2[0], 'correct': params2[1], 'total': params2[2]} 
        self.weights = [w1, w2]
        
        self.nongrad_steps = nongradient_steps
        self.del_w_lim = del_w_limit
        self.del_loss_lim = del_loss_limit
        
    def find(self):
        
        losses = [self.weights[0]['loss'], self.weights[1]['loss']]
        min_w = self.weights[np.argmin(losses)]['weight']
        

if __name__=='__main__':
    
    model_handler = TrainedModels('models', True)
    
    p = PathFinder(model_handler=model_handler,
                    nongradient_steps=50,
                    del_w_limit=1e-2,
                    del_loss_limit=1e-2)
    p.find()
