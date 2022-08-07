import numpy as np
import os
import sys
sys.path.append('../')
import pickle
from common.io import _load
from common.timer import Timer
import torch
from tqdm import tqdm


def load(fname):
    root = '../prob_regressor_data/'
    time = Timer()
    time.start()
    data = _load(os.path.join(root, fname))
    time.stop(f'load {fname}')
    g = torch.tensor(data['g']).to('cuda')
    time.stop(f'load g to GPU')
    N,D = g.shape
    F = 0
    for i  in tqdm(range(N)):
        gg = g[i]
        F += torch.ger(gg, gg) 
    time.stop(f'cal torch.ger(gg, gg) for {N} times')
    data['F'] = F.cpu().numpy()/N
    time.stop(f'dump to CPU')
    data['g'] = []
    import pdb;pdb.set_trace()
    pickle.dump(data, open(os.path.join(root, fname.split('.')[0] + '.new1.pkl'), 'wb'))
    time.stop(f'dump to disk')

load('mlp_mnist_10000samples_10000batches.npy')
#load('../prob_regressor_data/resnet20_cifar10_1000samples_1000batches_0seed.pkl')
