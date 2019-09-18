EPOCH = -1 # Consider last checkpoint for every model
DIFF_RATE = 1
NUM_STEPS = 10
LOG_STEPS = 1
CUDA = True
DATASET_PATH = 'data/'
BATCH_SIZE = 256
TEST_BATCH_SIZE = 1024
NPARTS = 10

from analyse_weights import TrainedModels 
from data import get_mnist_loaders
from model import Net
from main import test

import torch
import numpy as np
import copy

import logging
logging.basicConfig(level=logging.INFO, filename='log.txt', filemode='a', format='%(name)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

import sys
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)s %(levelname)s %(asctime)s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def iterate_along_line_direction(model1, model2, test_loader, device, num_steps=NUM_STEPS, diffrate=DIFF_RATE, logsteps=LOG_STEPS, nparts=NPARTS):
    
    with torch.no_grad():
        diff = []
        for m, d in zip(model1.parameters(), model2.parameters()):
            diff.append((m-d)/nparts)
        
        for idx in range(num_steps):
        
            for model_w, diff_w in zip(model2.parameters(), diff):
            #for model_w, other_w in zip(model1.parameters(), model2.parameters()):
                #k1, w1 = model_w
                #k2, w2 = other_w
                
                # Move model2 along the direction of the better model
                model_w.add_(diff_w * diffrate)
                #other_w = other_w - (diffrate * (other_w-model_w))

            if (idx+1) % logsteps == 0:
                loss, correct = test(model2, test_loader, device)
                
                '''
                try:
                    if loss > ploss:
                        logger.info ("Loss value increased since last step; TERMINATING!!")
                        exit()
                except NameError:
                    ploss = loss

                ploss = loss
                '''
                logger.info ('Epoch %d: Loss = %0.5f'%(idx+1, loss))

def iterate_along_triangle(model1, model2, model3, test_loader, device, interpolate=0.1, num_steps=NUM_STEPS, diffrate=DIFF_RATE, logsteps=LOG_STEPS):
    
    assert 0 < interpolate < 1
        
    with torch.no_grad():
        
        interpolated_model = copy.deepcopy(model2)
        
        for inter in np.arange(0, 1, interpolate):
        
            logger.info ("\nINTERPOLATION CONSTANT: %f\n\n"%(inter))
            for actual, other_w1, other_w2 in zip(interpolated_model.parameters(), model2.parameters(), model3.parameters()):    
                #actual[1]._sub(actual[1])
                actual.sub_(actual)
                actual.add_(inter*other_w1 + (1-inter)*other_w2)
        
            iterate_along_line_direction(interpolated_model, model1, test_loader, device, num_steps, diffrate, logsteps)
    
def swap(x1, x2):
    return x2, x1

if __name__=='__main__':

    device = torch.device("cuda:0" if CUDA and torch.cuda.is_available() else "cpu")

    model_handler = TrainedModels('models', True)

    # weights: NUM_SAVED_MODELS * NUM_EPOCHS * NUM_LAYERS
    # params: NUM_SAVED_MODELS * NUM_EPOCHS * [loss, num_correct, num_total]
    weights, params, keys = model_handler.get_models(not CUDA or not torch.cuda.is_available())
    
    idx1, idx2 = model_handler.get_random_model_keys(weights)
    
    idx3 = idx1
    while idx3 == idx1 or idx3 == idx2:
        idx3, _ = model_handler.get_random_model_keys(weights)

    if params[idx1][EPOCH][0] > params[idx2][EPOCH][0]:
        # Swap idx1 with idx2 (idx1 represents the better model in terms of loss)
        idx1, idx2 = swap(idx1, idx2)
    
    model_handler.display_stats(weights, params, keys, idx1, idx2)
    model_handler.display_stats(weights, params, keys, idx2, idx3)
    model_handler.display_stats(weights, params, keys, idx1, idx3)
    
    train_loader, test_loader = get_mnist_loaders(cuda_flag=CUDA, dataset_path=DATASET_PATH, 
                                        batch_size=BATCH_SIZE, test_batch_size=TEST_BATCH_SIZE)

    
    location = 'cpu' if not CUDA or not torch.cuda.is_available() else 'gpu'
    
    logger.info ('Loss values in order: %0.5f, %0.5f, %0.5f'%(params[idx1][EPOCH][0], params[idx2][EPOCH][0], params[idx3][EPOCH][0]))

    model1 = Net()
    model1.load_state_dict(weights[idx1][EPOCH])

    model2 = Net()
    model2.load_state_dict(weights[idx2][EPOCH])
    
    model3 = Net()
    model3.load_state_dict(weights[idx3][EPOCH])

    model1.to(device)
    model2.to(device)
    model3.to(device)

    #iterate_along_line_direction(model1, model2, test_loader, device)
    iterate_along_triangle(model1, model2, model3, test_loader, device)
