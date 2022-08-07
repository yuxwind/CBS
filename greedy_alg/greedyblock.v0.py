import os
import sys
sys.path.append('../')

import argparse
import numpy as np
import torch
from tqdm import tqdm
import math
import glob
import pathlib

from common.io import _load, find_float_in_str
from common.timer import Timer
from common.debug_memory import print_active_tensors
from greedy_alg.mat_utils import product_efficient_v1
from greedy_alg.greedy_io import get_greedy_exp_name, get_greedy_exp_paths
from greedy_alg.greedy_io import load_idx_2_module, get_module, get_modules, is_fc, in_same_block
from greedy_alg.greedy import GreedyPruner
from greedy_alg.greedy_options import get_parse, debug_args 

DEBUG = False
ROOT  = os.path.dirname(pathlib.Path(__file__).parent.resolve())

class GreedyBlockPruner(GreedyPruner):
    def __init__(self, sparsity=0.6, wgh_path=None, weights=None, grads=None, fisher_matrix=None, device='cuda', args=None):
        super(GreedyBlockPruner, self).__init__(sparsity=sparsity, 
                wgh_path=wgh_path, weights=weights, grads=grads, 
                fisher_matrix=fisher_matrix, device=device, args=args)
        self.get_layer_of_weights()

    def get_layer_of_weights(self):
        self.layer_idx_of_w = torch.zeros(self.w.shape).to(self.device)
        l_idx = 0
        for k,v in self.idx_2_module.items():
            self.layer_idx_of_w[k[0]:k[1]] = l_idx
            l_idx += 1

    def mask_ij(self, i, j):
        return self.layer_idx_of_w[i] == self.layer_idx_of_w[j]

    def mask_IJ(self, I, J, unique_I = True):
        ''' whether the weights of indices I and J belongs to a same layer
            return:
                mask: |I| * |J|
        '''
        I_idx = self.layer_idx_of_w[I]
        if unique_I:
            I_idx = torch.unique(I_idx)
        J_idx = self.layer_idx_of_w[J]
        #import pdb;pdb.set_trace()
        mask = I_idx.repeat(len(J),1).T == J_idx
        return mask
   
    def get_F_ij(self, i, j):
        if not self.mask_ij(i,j):
            return 0.0
        if self.decompose_F:
            return (self.G[:,i] * self.G[:,j]).mean(dim=0)
        else:
            return self.F[i,j]

    def cal_HW(self, I, J, is_HW=True, part=1):
        '''calculate the basic operation:
                I: the rows of H
                J: the columns of H
            is_HW = True:  H_IJ @ W_J => |I| 
            is_HW = False: W_I  @ H_IJ = H_JI @ W_I => |J|
        '''
        # swap the I and J to calculate WH
        if not is_HW:
            tmp = I
            I   = J
            J   = tmp
        rst   = torch.zeros([len(I)]).to(self.device)
        mask  = self.mask_IJ(I,J)
        I_idx = torch.unique(self.layer_idx_of_w[I])
        I = I.to(self.device)
        J = J.to(self.device)

        if self.decompose_F:
            #gw     = self.G[:,J] @ self.w[J] # N x 1
            for k, Ii in enumerate(I_idx):
                j  = J[mask[k]]
                if len(j) > 0:
                    gw = product_efficient_v1(self.G[:,j],  self.w[j], num_parts=part)
                    i  = self.layer_idx_of_w[I] == Ii 
                    rst[i] = (self.G[:, I[i]] * gw[:,None]).mean(dim = 0)
                    del gw
            return rst.squeeze()
        else:
            N_I    = len(I)
            N_part = math.ceil(N_I / part)
            rst    = torch.zeros(N_I).to(self.device)
            w_J    = self.w[J]
            for i in range(part):
                s = i * N_part
                e = min(N_I, (i+1)*N_part)
                set_idx = [ii for ii in range(s,e)]
                idx     = I[set_idx]
                if len(idx) > 0:
                    rst[set_idx] =  self.F[np.ix_(idx, J)] @ w_J 
            return rst
    
    def debug(self):
        keys = list(self.idx_2_module.keys())
        I1 = keys[1]
        I2 = keys[4]
        I3 = keys[6]
        i1 = torch.randint(I1[0],I1[1], (10,))
        j1 = torch.randint(I1[0],I1[1], (20,))
        i2 = torch.randint(I2[0],I2[1], (30,))
        j2 = torch.randint(I2[0],I2[1], (40,))
        i3 = torch.randint(I3[0],I3[1], (50,))
        j3 = torch.randint(I3[0],I3[1], (60,))



        I = torch.cat((i1, i3))
        J = torch.cat((j1, j2))
        hw1 = self.cal_HW(I,J)
        hw2 = torch.zeros(hw1.shape).to(hw1.device)
        hw2[:len(i1)] = self.cal_HW(i1, j1)
        hw2[len(i1):] = 0.0
        assert (hw1 == hw2).all()
        import pdb;pdb.set_trace()

if __name__ == '__main__':
    args = get_parse()
    args.method = 'greedyblock'
    debug_args(args)
    greedy_pruner = GreedyBlockPruner(sparsity=args.sparsity, wgh_path=args.wgh_path, device='cuda', args=args)
    if args.only_calculate_delta_w:
        greedy_pruner.cal_delta_w()
    else:
        greedy_pruner.prune()	
