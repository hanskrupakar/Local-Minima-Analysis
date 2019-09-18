import torch
from torch.nn import Module
from torch.nn import functional as F
from analyse_weights import TrainedModels
from main import save_model, load_model, test
from data import get_mnist_loaders

import argparse
import numpy as np
import copy
import os

import logging
logging.basicConfig(filename='log.txt', filemode='a', level=logging.INFO,
                    format='%(name)s %(asctime)s %(levelname)s %(message)s')                              
logger = logging.getLogger(__name__)

import sys
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)s %(asctime)s %(levelname)s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

optim_str = ['Adadelta', 'Adagrad', 'Adam', 'Adam', 'Adamax', 'RMSprop', 'SGD', 'SGD']
optim_params = [
            {'lr':1e-4, 'rho': 0.9, 'eps': 1e-06},
            {"lr":1e-5, "lr_decay":0, "weight_decay":0},
            {"lr":1e-6, "betas":(0.9, 0.999), "eps": 1e-08, "amsgrad":False},
            {"lr":1e-6, "betas":(0.9, 0.999), "eps": 1e-08, "amsgrad":True},
            {"lr":2e-6, "betas":(0.9, 0.999), "eps": 1e-08},
            {"lr":1e-5, "alpha":0.99, "eps":1e-08},
            {"lr":1e-5, "momentum":0.9, "nesterov":False},
            {"lr":1e-5, "momentum":0.9, "nesterov":True}
         ]

def get_random_optimizer(model):
 
    optims = [getattr(torch.optim, x) for x in optim_str]
    optim_idx = np.random.randint(len(optims))
    optim = optims[optim_idx](model.parameters(), **optim_params[optim_idx])

    return optim

class PathFinder(Module):
    
    # model_src, model_dest: pytorch models encapsulated with local Model container
    def __init__(self, model_src, model_dest, device, step_min=1e-3, del_loss=4e-3, split_size=10, window=5):
        
        assert all([x.size()==y.size() for x,y in zip(model_src.model.parameters(), model_dest.model.parameters())])

        self.src_model = model_src
        self.target_model = model_dest
        self.window = window
        self.step_min = step_min
        self.del_loss = del_loss
        self.split_size = split_size
        self.device = device
        
        self.src_model.display_results('src-model (W_0)')
        self.target_model.display_results('tgt-model (W_n)')
        
        self._modules = self.src_model.model._modules
        self._parameters = self.src_model.model._parameters
        self._buffers = self.src_model.model._buffers
        
        self._modules.update(self.target_model.model._modules)
        self._parameters.update(self.target_model.model._parameters)
        self._buffers.update(self.target_model.model._buffers)
        
        # split_size: Number of parts determining size of non-gradient step along straight line
        # target_model: Destination for minima path
        # window: Set of previous paths to find next minima
        # step_min: Minimum 
        # del_loss: Minimum difference in loss to be considered part of the minima path
    
    def forward(self, x, gradient=False):
        
        if not gradient:
            
            logger.info('Took non-gradient step towards target model (L/%d of vector line difference in Euclidean space)'%(self.split_size))
            
            with torch.no_grad():  
                for params, paramt in zip(self.src_model.model.parameters(), self.target_model.model.parameters()):
                    params += (paramt - params) / self.split_size
        else:
            return self.src_model.model(x)
    
    def replace_model(self, model, source=False):
    # source: True if source model needs to be replaced
    # model: Model instance 

        if source:
            self.src_model = model
        else:
            self.target_model = model

    def avg_frob_norm(self, avgmodel):
        
        with torch.no_grad():
            
            nacc, ctx = 0, 0

            for p in avgmodel.parameters():
                basesize = p.size()[2:]
                if len(basesize) < 2:
                    nacc += torch.norm(p) 
                else:
                    nacc += torch.norm(p.view(-1, *basesize))
                ctx+=1

            return (nacc / float(ctx)).item()

class Model:
    
    def __init__(self, model, loss, correct, total):
        
        self.model = model
        self.loss = loss
        self.correct=correct
        self.total=total
    
    def display_results(self, mname):
        logger.info('\n%s model stats:\n\tLoss: %.5f\n\tCorrect: %d/%d\n'%(mname, self.loss, self.correct, self.total) + '.'*50+'\n') 
    
    def edit_metadata(self, loss, correct, total):
        
        self.loss = loss
        self.correct=  correct
        self.total = total

    def __sub__(self, other):
        
        with torch.no_grad():
            diff = copy.deepcopy(self.model)
            for p1, p2 in zip(diff.parameters(), other.model.parameters()):
                p1 -= p2
        
            m = Model(diff, other.loss, other.correct, other.total)
            return m

if __name__=='__main__':
    
    ap = argparse.ArgumentParser()
    ap.add_argument('--cpu', help='Runs on CPU', action='store_true')
    ap.add_argument('--num_steps', type=int, help='Number of iterations to wait for path', default=1000)
    args = ap.parse_args()

    dtrain, dval, dtest = get_mnist_loaders(cuda_flag=not args.cpu and torch.cuda.is_available(), dataset_path='data/', val=True)
    if not args.cpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
        cuda = True
    else:
        device = torch.device('cpu')
        cuda = False

    model_handler = TrainedModels('models', pytorch_model=True)
    weights, params, layer_keys = model_handler.get_models()

    # min, max
    m, M = 0, 0
    # loss values comparison
    if params[0][0][0] > params[1][0][0]:
        m = 1
    else:
        M = 1
    
    src_model = Model(load_model(weights[M][0], weight=True), params[M][0][0], params[M][0][1], params[M][0][2])
    tgt_model = Model(load_model(weights[m][0], weight=True), params[m][0][0], params[m][0][1], params[m][0][2])
    
    p = PathFinder(src_model, tgt_model, device,
                    step_min=1e-3, del_loss=1e-3, split_size=50)

    p = p.to(device)
    
    if m!=1:
        model_handler.model_files = [x for x in reversed(model_handler.model_files)]

    target_loss = p.target_model.loss
    
    ng = False
    pt_ctx = 0

    try:
        os.mkdir('path_weights')
    except Exception:
        pass
    
    get_fname = lambda x: '.'.join(x.split('/')[-1].split('.')[:-1])
    seq_name = get_fname(model_handler.model_files[0])+'_'+get_fname(model_handler.model_files[1])

    try:
        os.mkdir('path_weights/'+ seq_name)
    except Exception:
        pass

    for step in range(args.num_steps):
        
        backup = copy.deepcopy(p.src_model)
        logger.info('Global step: %d'%(step))
        
        if not ng:
            ng = True
            with torch.no_grad():
                p.forward(torch.zeros(1), gradient=False)
                loss_val, correct_val = test(p.src_model.model, dval, device, log=False)
                p.src_model.edit_metadata(loss_val, correct_val, len(dval.dataset))
        
        else:
            cur_loss = p.src_model.loss
            optimizer = get_random_optimizer(p.src_model.model)
            
            ptr = 0

            logger.info('Current difference in loss: %0.5f'%(cur_loss-target_loss))
            logger.info('Difference in loss tolerated: %0.5f'%(p.del_loss))
            while abs(cur_loss - target_loss) > p.del_loss: 
                data, target = next(iter(dtrain))
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()

                pred = p.forward(data, gradient=True)
                cur_loss = F.cross_entropy(pred, target, reduction='sum')
                
                cur_loss.backward()
                optimizer.step()

                logger.info('Pt: {} [{}]\tLoss: {:.6f}'.format(
                pt_ctx, ptr+1, cur_loss))

                if cur_loss < target_loss:
                    logger.info('Replaced current target model with lower loss (%0.5f > %0.5f)'%(target_loss, cur_loss))
                    p.replace_model(Model(p.src_model.model, cur_loss.item(), -1, -1))     
                    target_loss = cur_loss.item()
                    break

                ptr += 1

            pt_ctx += 1
            save_model(p.src_model.model, cur_loss, correct_val, len(dval.dataset), 'path_weights/'+seq_name+'/%s.pt'%(str(pt_ctx).zfill(4)))
            
            ng = False
        
        logger.info('AVG FROB NORM of SRC: %0.5f'%(p.avg_frob_norm(backup.model)))
        logger.info('AVG FROB NORM of TARGET: %0.5f'%(p.avg_frob_norm(p.src_model.model)))
        logger.info('AVG FROB NORM of DIFF: %0.5f'%(p.avg_frob_norm((p.src_model - backup).model)))

