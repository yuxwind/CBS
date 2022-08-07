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
from greedy_alg.mat_utils import product_efficient_v1, g_prod_gw_efficient
from greedy_alg.greedy_io import get_greedy_exp_name, get_greedy_exp_paths
from greedy_alg.greedy_io import load_idx_2_module, get_module, get_modules, is_fc, in_same_block
from greedy_alg.greedy import GreedyPruner, load_fisher_inv
from greedy_alg.greedy_options import get_parse, debug_args 

DEBUG = True
ROOT  = os.path.dirname(pathlib.Path(__file__).parent.resolve())

class GreedyLayerPruner(GreedyPruner):
    def __init__(self, sparsity=0.6, wgh_path=None, weights=None, grads=None, fisher_matrix=None, device='cuda', args=None):
        super(GreedyLayerPruner, self).__init__(sparsity=sparsity, 
                wgh_path=wgh_path, weights=weights, grads=grads, 
                fisher_matrix=fisher_matrix, device=device, args=args)
        timer = Timer()
        self.gw = None
        self.pruners = []
        self.set_pruned_list = []
        greedy_dir = args.greedy_dir
        greedy_path = args.greedy_path
        
        self.prune_init()
        mask = torch.ones(self.N_w)
        mask[self.set_pruned] = 0
        
        for l in range(self.N_layer):
            s,e   = self.layer_2_w[l]
            l_idx = torch.arange(s,e) 
            l_w   = self.w[s:e]
            l_g   = self.G[:,s:e]
            l_set_pruned = (mask[s:e] == 0).nonzero().squeeze().cpu()
            args.greedy_dir = greedy_dir + f'.layer{l}'
            args.greedy_path = None
            if len(l_set_pruned) > 0:
                self.set_pruned_list.append(l_set_pruned)
                l_pruner = GreedyPruner(sparsity=sparsity, weights=l_w, grads=l_g, args=args)
                l_pruner.prune_init()
                self.pruners.append(l_pruner)
        args.greedy_dir = greedy_dir
        args.greedy_path = greedy_path

    def prune_one_iter(self, itr=1):
        sets_pruned = []
        N_pruner = len(self.pruners)
        to_stops = torch.zeros(N_pruner).bool()
        timer = Timer()
        import pdb;pdb.set_trace()
        for l in range(N_pruner):
            print(f'========={l}-th layer===========')
            if not to_stops[l]:
                to_stops[l] = self.pruners[l].prune_one_iter(itr)
                self.set_pruned_list[l] = self.pruners[l].set_pruned + self.layer_2_w[l][0]
        self.cur_obj = self.cal_objective()
        self.iter_meta = None
        self.set_pruned = torch.cat(self.set_pruned_list)
        to_stop = to_stops.all()
        import pdb;pdb.set_trace()
        return to_stop, self.set_pruned

if __name__ == '__main__':
    args = get_parse()
    args.method = 'greedyblock'
    debug_args(args)
    greedy_pruner = GreedyBlockPruner(sparsity=args.sparsity, wgh_path=args.wgh_path, device='cuda', args=args)
    greedy_pruner.init_with_woodfisherblock()
    greedy_pruner.cal_delta_w_not_comb() 
    #greedy_pruner.debug() 
    #if args.only_calculate_delta_w:
    #    greedy_pruner.cal_delta_w()
    #else:
    #    greedy_pruner.prune()	
