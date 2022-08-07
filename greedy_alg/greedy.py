import os
import sys
#sys.path.append('../')
sys.path.append(os.path.join(sys.path[0], '..'))

import argparse
import numpy as np
import torch
from tqdm import tqdm
import math
import glob
import pathlib
import copy

from common.io import _load, find_float_in_str
from common.timer import Timer
from common.debug_memory import print_active_tensors
from greedy_alg.mat_utils import product_efficient_v1, g_prod_gw_efficient
from greedy_alg.greedy_io import get_greedy_exp_name, get_greedy_exp_paths
from greedy_alg.greedy_io import load_idx_2_module, get_module, get_modules, in_same_block
from greedy_alg.greedy_options import get_parse, debug_args

DEBUG = False
ROOT  = os.path.dirname(pathlib.Path(__file__).parent.resolve())

import contextlib
@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

def load_fisher_weight(wgh_path, load_F = False, load_G = True):
    data = _load(wgh_path)
    W, F, G = [],[],[]
    if 'F' in data and load_F:
        F = data['F']
    if 'g' in data and load_G:
        if 'mlp_mnist' in wgh_path:
            G = data['g'] * math.sqrt(data['g'].shape[0])
        else:
            if type(data['g']) == str:
                data_g = _load(data['g'])
                G    = data_g['all_grads']
            else:
                G    = data['g']
    if 'W' in data:
        W = data['W']
    if 'F' in data:
        del data['F']
        torch.cuda.empty_cache()
    return W, F, G

def load_fisher_inv(fisher_inv_path):
    data = _load(fisher_inv_path)
    return data['f_inv']

class Meta():
    def __init__(self, idx_2_module_path):
        self.metas  = []
        self.swap_cnt        = 0
        self.idx_2_module = load_idx_2_module(idx_2_module_path)
    
    @staticmethod
    def set(b_obj=None, a_obj=None, i=None, j=None, reduction=None, i_layer='', j_layer=''):
        meta_type = [('obj_before_swap', float), ('obj_after_swap',float), 
                ('reduction', float), ('swap_in_i', int), ('swap_out_j', int),
                ('swap_in_layer', np.chararray), ('swap_out_layer', np.chararray)]
        if b_obj is None:
            return np.array([], dtype=meta_type)
        meta = np.array((b_obj, a_obj, reduction, i, j, i_layer, j_layer), dtype=meta_type)
        return meta

    def add(self, meta):
        meta['swap_in_layer'] = get_module(self.idx_2_module, meta['swap_in_i'])
        meta['swap_out_layer'] = get_module(self.idx_2_module, meta['swap_out_j'])
        self.metas.append(meta)
        self.swap_cnt += 1

    def get(self,):
        meta_type = [('obj_before_swap', float), ('obj_after_swap',float), 
                ('reduction', float), ('swap_in_i', int), ('swap_out_j', int),
                ('swap_in_layer', np.chararray), ('swap_out_layer', np.chararray)]
        if len(self.metas) == 0:
            return np.array([], dtype=meta_type)
        return np.stack(self.metas)

class GreedyPruner():
    def __init__(self, sparsity=0.6, wgh_path=None, weights=None, grads=None, fisher_matrix=None, 
                device='cuda', args=None):
        self.args     = args
        self.wgh_path = wgh_path
        self.greedy_path= args.greedy_path
        self.greedy_dir = args.greedy_dir
        self.idx_2_module_path = args.idx_2_module_path
        self.fisher_inv_path   = args.fisher_inv_path
        self.woodfisher_mask_path = args.woodfisher_mask_path

        self.device = device
        self.init_method = args.init_method
        self.load_F = False
        self.load_G = True
        self.decompose_F = self.load_G
        self.use_simulated_data = False
        self.swap_one_per_iter = args.swap_one_per_iter
        if DEBUG:
            self.load_F = True 
            self.decompose_F = True
            #self.use_simulated_data = True
        if not self.use_simulated_data:
            #print_active_tensors()
            if self.wgh_path is not None:
                self.w, self.F, self.G = load_fisher_weight(self.wgh_path, self.load_F, self.load_G)
                print_active_tensors()
                self.w = torch.tensor(self.w).to(self.device)
                if self.F is not None:
                    self.F = torch.tensor(self.F).to(self.device)
                #TODO: debug for potential overfloat problem in mobilenet 
                #self.G = torch.tensor(self.G) * 10
                self.G = torch.tensor(self.G) 
                if not self.args.offload_grads:
                    self.G = self.G.to(self.device)
            else:
                self.w = weights
                self.F = fisher_matrix
                self.G = grads
        if self.args.has_first_order_term_in_obj: 
            self.G_mean = self.G.mean(dim=0).to(self.device)
        else:
            self.G_mean = None
            #print_active_tensors()
        
        self.set_pruned = []
        self.set_remaining = []
        self.set_I = []
        self.set_J = []
        self.sparsity = sparsity
        self.max_iter = args.max_iter
        self.RANGE = args.range
        self.MAX_NO_MATCH = args.max_no_match
        self.THRESHOLD = args.threshold
        self.last_iter = None
        self.iter      = 0
        self.iter_time = 0
        self.accumulate_time = 0
        self.iter_meta = None
        self.cur_obj   = None

        self.set_greedy_dir()
        self.set_greedy_path()
        
        # DEBUG
        self.into_remaining  = []
        self.outof_remaining = []
        self.sort_with_magnitude()
        self.idx_2_module = load_idx_2_module(self.idx_2_module_path)
        self.N_layer = len(self.idx_2_module.keys())
        self.N_sample, self.N_w = self.G.shape
        self.get_layer_of_weights() 
        #if not self.use_simulated_data:
        #    self.cal_H_diag()
        #    self.wg = self.G * self.w

    def get_layer_of_weights(self):
        # Map from the weight index to chich the layer it belongs to
        self.w_2_layer = torch.zeros(self.w.shape).int().to(self.device)
        self.layer_2_w = torch.zeros([self.N_layer,2]).int().to(self.device)
        # Indicate the index of the first weight in l_idx layer
        self.layer_start_idx = torch.zeros(len(self.idx_2_module.keys()))
        l_idx = 0
        for k,v in self.idx_2_module.items():
            self.w_2_layer[k[0]:k[1]] = l_idx
            self.layer_2_w[l_idx]     = torch.tensor(k)
            self.layer_start_idx[l_idx] = k[0]
            l_idx += 1

    def debug_init_with_simulated_data(self):
        N = 8
        self.w = torch.arange(N).to(self.device).float()
        if  self.load_F:
            self.F = torch.arange(N**2).reshape(N,N).to(self.device).float()
        if self.load_G:
            N_samples = 3
            self.G = torch.arange(N_samples * N).reshape(N_samples, N).to(self.device).float()
            self.F = 0
            for i in range(N_samples):
                self.F += torch.ger(self.G[i], self.G[i])
            self.F = self.F / float(N_samples)
            self.cal_H_diag()
        self.set_pruned = torch.tensor([2,6,4]).long()
        self.set_remaining = torch.tensor([1,3,7,5,0]).long()
        #self.set_pruned = torch.tensor([2,3]).long()
        #self.set_remaining = torch.tensor([0,1]).long()
   
    def init_with_random(self):
        N_w = len(self.w)
        N_pruned = int(N_w * self.sparsity)
        perm = torch.randperm(N_w)
        self.set_pruned = perm[:N_pruned]
        self.set_remaining = perm[N_pruned:]

    def init_with_wg(self):
        g_mean = self.G.mean(dim=0).to(self.device)
        wg     = (g_mean * self.w).abs()
        num_params_to_keep = int(self.w.shape[0] * (1-self.sparsity))
        tops,out = torch.topk(wg, num_params_to_keep, sorted=True)
        threshold = tops[-1]
        del tops
        del out
        torch.cuda.empty_cache()
        self.set_pruned = (wg.abs() <= threshold).nonzero().squeeze().cpu()
        self.set_remaining = (wg > threshold).nonzero().squeeze().cpu()


    def init_with_magnitude(self):
        num_params_to_keep = int(self.w.shape[0] * (1-self.sparsity))
        tops,out = torch.topk(self.w.abs(), num_params_to_keep, sorted=True)
        threshold = tops[-1]
        del tops
        del out
        torch.cuda.empty_cache()
        self.set_pruned = (self.w.abs() <= threshold).nonzero().squeeze().cpu()
        self.set_remaining = (self.w > threshold).nonzero().squeeze().cpu()

    def init_with_woodfisherblock(self):
        wf_mask = _load(self.woodfisher_mask_path)
        assert(self.sparsity == wf_mask['sparsity'])
        mask = wf_mask['mask'].cpu()
        self.set_pruned = (mask==0).nonzero().squeeze().cpu()
        self.set_remaining = (mask==1).nonzero().squeeze().cpu()

    def init_with_setting(self, pruned_idx):
        mask = torch.ones(self.N_w)
        mask[pruned_idx] = 0
        self.set_pruned = (mask==0).nonzero().squeeze().cpu()
        self.set_remaining = (mask==1).nonzero().squeeze().cpu()
        self.alpha_pruned   = self.cal_alpha_vec(part=4)
        self.beta_remaining = self.cal_beta_vec()
        self.cur_obj        = self.cal_objective() 

    def sort_with_magnitude(self):
        sorted_idx = torch.argsort(-self.w.abs())
        self.idx_2_rank = dict(zip(sorted_idx.cpu().numpy(), range(len(self.w))))

    def cal_alpha(self, i):
        ''' calculate the loss(\deta_w * H * \deta_w) for each pruned parameter
        '''
        w_pruned = self.w[self.set_pruned]
        H_i_row = self.F[i,self.set_pruned]
        H_i_col = self.F[self.set_pruned,i]
        alpha_i = self.w[i] * (H_i_row.dot(w_pruned) + H_i_col.dot(w_pruned)) \
                        - self.w[i] * self.F[i,i] * self.w[i] 
        return alpha_i
    
    def cal_beta(self, j):
        ''' calculate the loss for each remaining parameter
        '''
        w_pruned = self.w[self.set_pruned]
        H_i_row = self.F[j,self.set_pruned]
        H_i_col = self.F[self.set_pruned,j]
        beta_j = self.w[j] * (H_i_row.dot(w_pruned) + H_i_col.dot(w_pruned)) \
                        + self.w[j] * self.F[j,j] * self.w[j]
        return beta_j
    
    def _cal_alpha_vec(self):
        alpha = torch.zeros(len(self.set_pruned)).to(self.device)
        for i in range(len(self.set_pruned)):
            alpha[i] = self.cal_alpha(self.set_pruned[i])
        return alpha

    def _cal_beta_vec(self):
        beta = torch.zeros(len(self.set_remaining)).to(self.device)
        for i in range(len(self.set_remaining)):
            beta[i] = self.cal_beta(self.set_remaining[i])
        return beta
    
    def cal_H_diag(self):
        if self.decompose_F:
            self.H_diag = (self.G * self.G).mean(dim=0).to(self.device) 
        else:
            self.H_diag = self.F.diagonal()

    def get_H_diag(self, I):
        if self.decompose_F:
            if self.args.offload_grads and torch.is_tensor(I):
                I = I.cpu()
            return (self.G[:,I] * self.G[:,I]).mean(dim=0).to(self.device) 
        else:
            return self.F.diagonal()[I]

    def get_F_ij(self, i, j):
        if self.decompose_F:
            if self.args.offload_grads: 
                if torch.is_tensor(i):
                    i = i.cpu()
                if torch.is_tensor(j):
                    j = j.cpu()
            return (self.G[:,i] * self.G[:,j]).mean(dim=0).to(self.device)
        else:
            return self.F[i,j]

    def get_F_IJ(self, I, J):
        if self.decompose_F:
            if self.args.offload_grads:
                if torch.is_tensor(I):
                    I = I.cpu()
                if torch.is_tensor(J):
                    J = J.cpu()
            return (self.G[:,I] * self.G[:,J]).mean(dim=0).to(self.device)
        else:
            return self.F[np.ix_(I, J)]


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
        cap_N = 2.5e+9 
        s_N,w_N = self.G.shape
        J_N     = len(J)
        rest_N  = cap_N - (s_N * w_N)
        part    = int(max(1, 3 * s_N * J_N / rest_N))
        if self.decompose_F:
            gw     = self.G[:,J] @ self.w[J] # N x 1
            #gw     = product_efficient_v1(self.G[:,J],  self.w[J], num_parts=part)
            #gw1     = self.G[:,J] @ self.w[J] # N x 1
            #diff   = (gw1.to(self.device) - gw).abs().max()
            #print(f'cal_HW: debug {diff:.4e}')
            #import pdb;pdb.set_trace()
            rst    = (self.G[:, I] * gw[:,None]).mean(dim = 0)
            #rst    = g_prod_gw_efficient(self.G[:, I], gw) 
            #rst1    = (self.G[:, I] * gw[:,None]).mean(dim = 0)
            #diff   = (rst1.to(self.device) - rst).abs().max()
            #print(f'cal_HW: debug {diff:.4e}')
            #import pdb;pdb.set_trace()
            return rst
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
    
    def cal_alpha_vec(self, part=1):
        '''calculate the loss for all pruned weights
        '''
        # for pruned weights
        alpha = self.cal_HW(self.set_pruned, self.set_pruned, part=4)
        w_k = self.w[self.set_pruned]
        H_kk_diag = self.get_H_diag(self.set_pruned)
        alpha = 2 * w_k * alpha - w_k * H_kk_diag * w_k
        # debug
        if DEBUG and not (alpha == self.cal_alpha_vec1(4)).all():
            print('inconsistent alpha', (alpha - self.cal_alpha_vec1(4)).abs().max())
            import pdb;pdb.set_trace()
        del w_k, H_kk_diag
        torch.cuda.empty_cache()
        return alpha

    def cal_alpha_vec1(self, part=1):
        '''calculate the loss for all pruned weights
        '''
        # for pruned weights
        N_pruned = len(self.set_pruned)
        N_part = math.ceil(N_pruned / part)
        alpha=torch.zeros(N_pruned).to(self.device)
        w_k = self.w[self.set_pruned]
        H_kk_diag = self.F.diagonal()[self.set_pruned]
        for i in range(part):
            s = i * N_part
            e = min(N_pruned, (i+1)*N_part)
            set_idx = [ii for ii in range(s,e)]
            idx = self.set_pruned[set_idx]
            if len(idx) > 0:
                alpha[set_idx] =  self.F[np.ix_(idx, self.set_pruned)] @ w_k 
                alpha[set_idx] += self.F[np.ix_(self.set_pruned, idx)].T @ w_k
        #alpha[set_idx] = w_k * (H_kk @ w_k + w_k @ H_kk) - w_k * H_kk_diag * w_k
        alpha = w_k * alpha - w_k * H_kk_diag * w_k
        # debug
        if DEBUG and not (alpha == self._cal_alpha_vec()).all():
            print('inconsistent alpha', (alpha - self._cal_alpha_vec()).abs().max())
            import pdb;pdb.set_trace()
        del w_k, H_kk_diag
        torch.cuda.empty_cache()
        return alpha
    
    def cal_beta_vec(self):
        '''calculate the loss for the remaining weights
        '''
        w_k = self.w[self.set_pruned]    # pruned weights
        w_x = self.w[self.set_remaining] # remaining weights
        H_xx_diag = self.get_H_diag(self.set_remaining)
        beta = w_x * (self.cal_HW(self.set_remaining, self.set_pruned) \
                      + self.cal_HW(self.set_pruned, self.set_remaining, is_HW=False)) \
                + w_x * H_xx_diag * w_x
        # debug
        if DEBUG and not (beta == self.cal_beta_vec1()).all():
            print('inconsistent beta', (beta - self.cal_beta_vec1()).abs().max())
            import pdb;pdb.set_trace()
        del w_k, w_x, H_xx_diag
        torch.cuda.empty_cache()
        return beta

    def cal_beta_vec1(self):
        '''calculate the loss for the remaining weights
        '''
        w_k = self.w[self.set_pruned]    # pruned weights
        w_x = self.w[self.set_remaining] # remaining weights
        H_kx = self.F[np.ix_(self.set_pruned, self.set_remaining)]
        H_xk = self.F[np.ix_(self.set_remaining, self.set_pruned)]
        H_xx_diag = self.F.diagonal()[self.set_remaining]
        beta = w_x * (H_xk @ w_k + (w_k.T @ H_kx).T) + w_x * H_xx_diag * w_x
        # debug
        if DEBUG and not (beta == self._cal_beta_vec()).all():
            print('inconsistent beta', (beta - self._cal_beta_vec()).abs().max())
            import pdb;pdb.set_trace()
        del w_k, w_x, H_kx, H_xk, H_xx_diag
        torch.cuda.empty_cache()
        return beta

    def cal_gamma(self, i, j):
        #return self.w[i] * self.F[i,j] * self.w[j] + self.w[j] * self.F[j,i] * self.w[i]
        return self.w[i] * self.get_F_ij(i,j) * self.w[j] + self.w[j] * self.get_F_ij(j,i) * self.w[i]

    def cal_set_gamma(self, I, J):
        '''calculate gamma for two sets I and J and sum over J
            I: N1
            J: N2
            gamma: N1
        '''
        # to handle I, J are scaler tensor, list, int
        I = torch.tensor(I) if not torch.is_tensor(I) else I
        J = torch.tensor(J) if not torch.is_tensor(J) else J
        I = I[None].reshape(-1)
        J = J[None].reshape(-1)
        if len(I) == 0 or len(J) == 0:
            return 0
        w_I = self.w[I]
        w_J = self.w[J]
        gamma = w_I * (self.cal_HW(I, J) + self.cal_HW(J,I, is_HW=False))
        if DEBUG and not (gamma == self.cal_set_gamma1(I, J)).all():
            print('inconsistent gamma', (gamma - self.cal_set_gamma1(I, J)).abs().max())
        del I,J,w_I,w_J
        torch.cuda.empty_cache()
        return gamma

    def cal_set_gamma1(self, I, J):
        '''calculate gamma for two sets I and J and sum over J
            I: N1
            J: N2
            gamma: N1
        '''
        # to handle I, J are scaler tensor, list, int
        I = torch.tensor(I) if not torch.is_tensor(I) else I
        J = torch.tensor(J) if not torch.is_tensor(J) else J
        I = I[None].reshape(-1)
        J = J[None].reshape(-1)
        if len(I) == 0 or len(J) == 0:
            return 0
        w_I = self.w[I]
        w_J = self.w[J]
        H_IJ = self.F[np.ix_(I,J)]
        H_JI = self.F[np.ix_(J,I)]
        gamma = w_I * ((H_IJ + H_JI.T) @ w_J)
        del I,J,w_I,w_J, H_IJ, H_JI
        torch.cuda.empty_cache()
        return gamma

    def update_from_I(self, i):
        '''update loss for any parameter by removing terms related to I
        '''
        return self.cal_set_gamma(i, self.set_I) 
    
    def update_from_J(self, i):
        '''update loss for any paramter by adding terms related to I
        '''
        return self.cal_set_gamma(i, self.set_J) 

    def update_after_swap(self, i):
        '''update the loss of parameter i by set_I and set_J
        '''
        return self.update_from_J(i) - self.update_from_I(i)


    def update_all_after_swap(self):
        all_idx = np.arange(len(self.w))
        all_update =  self.update_from_J(all_idx) - self.update_from_I(all_idx)
        # for set_I, correct the update by adding back gamma_ii
        try:
            w_I, H_diag_I = self.w[self.set_I], self.get_H_diag(self.set_I)
        except:
            import pdb;pdb.set_trace()
        all_update[self.set_I] += w_I * H_diag_I * w_I * 2
        del all_idx, w_I, H_diag_I
        torch.cuda.empty_cache()
        # for set_J, correct the update by substracting  gamma_jj
        w_J, H_diag_J = self.w[self.set_J], self.get_H_diag(self.set_J)
        all_update[self.set_J] += -w_J * H_diag_J * w_J * 2
        del w_J, H_diag_J
        torch.cuda.empty_cache()
        return all_update

    def cal_objective1(self):
        obj = 0.0
        for i in self.set_pruned:
            for j in self.set_pruned:
                obj +=  self.w[i] * self.get_F_ij(i, j) * self.w[j]
        return obj

    def cal_objective(self, debug=False):
        if not hasattr(self, 'w') or not hasattr(self, 'G'): 
            self.w, self.F, self.G = load_fisher_weight(self.wgh_path, self.load_F, self.load_G)
            self.w = torch.tensor(self.w).to(self.device)
            self.F = torch.tensor(self.F).to(self.device)
            self.G = torch.tensor(self.G)
            if not self.args.offload_grads: 
                self.G = self.G.to(self.device) 
        if debug:
            w_I = self.w[self.set_pruned_debug]
            obj = w_I.T @ self.cal_HW(self.set_pruned_debug, self.set_pruned_debug)
        else:
            w_I = self.w[self.set_pruned]
            obj = w_I.T @ self.cal_HW(self.set_pruned, self.set_pruned)
        #print(f'DEBUG obj: {obj-self.cal_objective1():.4e}')
        #import pdb;pdb.set_trace()
        return obj

    def cal_objective_after_update(self, delta_w=None):
        if delta_w is None:
            delta_w = self.delta_w
        # set self.w = delta_w to make use of cal_HW() fucntion 
        delta_w = delta_w.clone() * self.args.scale_prune_update
        delta_w[self.set_pruned] = - self.w[self.set_pruned]
        w_ = self.w.clone()
        self.w  = delta_w
        # set delta_w for pruned weights as -w
        all_idx = torch.arange(len(w_))
        obj = delta_w.T @ self.cal_HW(all_idx, all_idx)
        # recover self.w
        self.w = w_
        return obj

    def is_valid_swap(self, alpha_i, beta_j):
        '''Check whether to swap the other elements in set_pruned and set_remaining 
        '''
        DEBUG = False
        if DEBUG:
            b_obj = self.cal_objective(debug=DEBUG)
            print('\n')
            print("before swap:", b_obj)
        i = self.set_pruned[alpha_i].clone()
        j = self.set_remaining[beta_j].clone()
        alpha_new = (self.alpha_pruned[alpha_i] + self.update_after_swap(i)).squeeze()
        beta_new  = (self.beta_remaining[beta_j] + self.update_after_swap(j)).squeeze()
        reduction =  beta_new - self.cal_gamma(i,j) - alpha_new
        valid = -reduction >=self.THRESHOLD  
        
        if DEBUG:
            print(f'i={i}, weights[i]={self.w[i]:.4e}, rank_of_i={self.idx_2_rank[i.item()]}, alpha_new_i={alpha_new:.4e}, alpha_old_i={self.alpha_pruned[alpha_i]:.4e}' ) 
            print(f'j={j}, weights[j]={self.w[j]:.4e}, rank_of_j={self.idx_2_rank[j.item()]}, beta_new_j={beta_new:.4e}, beta_old_j={self.beta_remaining[beta_j]:.4e}' ) 
            print(f'gamma(i,j)={self.cal_gamma(i,j):.4e}, reduction: {-reduction:.4e}')

            # swap it just to calculate obj for debug 
            #self.set_pruned[alpha_i] = j
            if valid:
                self.set_pruned_debug[alpha_i] = j
                a_obj = self.cal_objective(debug=DEBUG)
                print(f'b_obj={b_obj:.4e}, a_obj={a_obj:.4e}')
                print(f'reduction={-reduction:.4e}, b_obj-a_obj={b_obj-a_obj:.4e},', 
                        f'diff={(-reduction)-(b_obj-a_obj):.4e}')
                meta = Meta.set(b_obj, a_obj, reduction, i, j)
            else:
                print('invliad swap')
                meta = None
            # restore swap
            #self.set_pruned[alpha_i] = i
        else:
            meta = None
        
        #allowed = allowed(self.idx_2_module, j)
        #allowed = in_same_block(self.idx_2_module, i,j)
        #print(f'valid={valid}, in_same_layer={allowed} (not required for swapping in this exp)')
        allowed = True
        return valid, allowed, reduction, meta

    def prune_init(self,):
        # step1: get initial sets by magnitude-based pruning
        if self.use_simulated_data:
            self.debug_init_with_simulated_data()
        else:
            if self.init_method == 'wg':
                self.init_with_wg()
            elif self.init_method == 'woodfisherblock':
                self.init_with_woodfisherblock()
            elif self.init_method == 'random':
                self.init_with_random()
            elif self.init_method == 'mag' or self.init_method == 'mag_perb':
                self.init_with_magnitude()
            #elif self.init_method == 'mag_perb':
            #    seed = self.args.result_file
            #    time=[s for s in self.args.result_file.split('/') if 'csv.' in s][0].split('.')[-1]
            #    time= int(time.replace('_', '')) # '%H_%M_%S'
            #    self.constructive_greedy_prune_one_iter(time)
            else:
                raise(f'Unknown init method for greedy: {self.init_method}')
        self.iter_meta = Meta.set()
        # step2: initialize self.alpha_pruned, self.beta_remaining 
        self.alpha_pruned   = self.cal_alpha_vec(part=4)
        self.beta_remaining = self.cal_beta_vec()
        self.cur_obj        = self.cal_objective() 


    def constructive_greedy_prune_one_iter(self, itr=0):
        N_w = len(self.w)
        N_chunk   = max(1, min(int(math.sqrt(N_w)), 1000))
        chunk_size = math.ceil(N_w / N_chunk)
        #chunk_size = max(int(math.sqrt(N_w)), 1000)
        #N_chunk    = math.ceil(N_w / chunk_size)
       

        N_keeped = int(N_w * (1-self.sparsity))
        num_params_to_keep = math.ceil(chunk_size * (1-self.sparsity))
        
        sets_keeped = []
        with temp_seed(itr):
            shf_idx = np.arange(N_w) 
            np.random.shuffle(shf_idx)
            shf_idx = torch.tensor(shf_idx)
           
            for i in range(N_chunk):
                if i < N_chunk - 1:
                    b,e = chunk_size * i, chunk_size * (i+1)
                else:
                    b,e = chunk_size * i, N_w
                    num_params_to_keep = max(0, N_keeped - len(torch.cat(sets_keeped)))
                w = self.w[shf_idx[b:e]]
                try:
                    tops,out = torch.topk(w.abs(), num_params_to_keep, sorted=True)
                except:
                    import pdb;pdb.set_trace()
                if len(tops) > 0:
                    threshold = tops[-1]
                    set_keeped = (w.abs() >= threshold).nonzero().squeeze().cpu()
                    sets_keeped.append(shf_idx[set_keeped + b])
            
            new_remaining = torch.cat(sets_keeped)
            mask = torch.zeros(N_w)
            mask[new_remaining] = 1
            new_pruned = (mask == 0).nonzero().squeeze()

            out, counts = torch.unique(torch.cat([new_pruned, self.set_pruned]), return_counts=True)
            diff        = out[counts==1]
            print(f'================={len(diff)/2} differnce elements ')
        
        self.set_pruned = new_pruned.cpu()
        self.set_remaining = new_remaining.cpu()

    def prune_one_iter(self, itr=0):
        # DEBUG
        MAX_SWAP = 1000000
        test_constructive_greedy = False
        if test_constructive_greedy:
            self.constructive_greedy_prune_one_iter(itr)
            return False
        
        old_obj = self.cur_obj.clone()
        sum_reduct = 0.0
        to_stop = False
        iter_meta = Meta(self.idx_2_module_path)
        self.alpha_pruned, sorted_idx = torch.sort(self.alpha_pruned, descending=True)
        self.set_pruned = self.set_pruned[sorted_idx]
        #self.set_pruned_debug = self.set_pruned.clone()
        self.set_pruned_debug = copy.deepcopy(self.set_pruned)
        _, sorted_idx  = torch.sort(
            self.beta_remaining - self.cal_set_gamma(self.set_remaining, self.set_pruned[0]), 
            descending=False)
        self.beta_remaining = self.beta_remaining[sorted_idx]
        self.set_remaining  = self.set_remaining[sorted_idx]
        self.set_I = []
        self.set_J = []
        #timer.stop('sorting alpha and beta')
        # check the firest elemtnes in set_pruned and set_remaining
        valid, allowed, reduction, meta = self.is_valid_swap(0,0)
        if not valid:
            to_stop = True
            return to_stop 
        else:
            if allowed:
                self.set_I.append(self.set_pruned[0])
                self.set_J.append(self.set_remaining[0])
                sum_reduct += reduction
                print(f'i,j: 0,0, reduction={-reduction:.4e}')
                if DEBUG:
                    iter_meta.add(meta)
                #import pdb;pdb.set_trace() 
        no_match_cnt = 0
        set_I_in_alpha = [0] if allowed else []
        set_J_in_beta  = [0] if allowed else []
        is_in_J = torch.zeros(len(self.set_remaining))
        for i in range(1, len(self.set_pruned)):
        #for i in range(1, 5):
            # DEBUG: only compare the first pair
            #        at most swap MAX_SWAP
            if self.swap_one_per_iter or len(self.set_I) >= MAX_SWAP:
                break 
            # if there are more than no_match_cnt_max weights in set_pruned didn't find matching
            if no_match_cnt > self.MAX_NO_MATCH:
                break
            #timer.stop(f'swap: i={i}') 
            
            for j in range(max(1,i-self.RANGE), min(len(self.set_remaining), i+self.RANGE)):
                if is_in_J[j] == 0:
                    valid, allowed, reduction, meta = self.is_valid_swap(i,j)
                    if valid and allowed:
                        #print(f'i,j: {i},{j}, reduction={reduction}')
                        self.set_I.append(self.set_pruned[i])
                        self.set_J.append(self.set_remaining[j])
                        sum_reduct += reduction
                        set_I_in_alpha.append(i)
                        set_J_in_beta.append(j)
                        is_in_J[j] = 1
                        if DEBUG:
                            iter_meta.add(meta)
                        print(f'swap: i={i}, j={j}, reducetion={-reduction:.4e}')
                        #import pdb;pdb.set_trace()
                        break
            if is_in_J[j] == 0:
                no_match_cnt += 1
        self.iter_meta = iter_meta.get()
        # update alpha and beta
        self.set_I = torch.tensor(self.set_I)
        self.set_J = torch.tensor(self.set_J)
        print(f'len(set_I): {len(self.set_I)}, len(self_J): {len(self.set_J)}')
        if len(self.set_I) > 0:
            update_all = self.update_all_after_swap()
            # update alpha and beta
            self.alpha_pruned += update_all[self.set_pruned]
            self.beta_remaining += update_all[self.set_remaining]
            # Swap set_J (into set_pruned, alpha_pruned), set_I (into set_remaiming, beta_remaining)
            alpha_I = self.alpha_pruned[set_I_in_alpha] 
            beta_J  = self.beta_remaining[set_J_in_beta]
            
            self.alpha_pruned[set_I_in_alpha] = beta_J
            self.set_pruned[set_I_in_alpha] = self.set_J
            #print(f'set_I weights', self.w[self.set_I])
            #print(f'set_J weights', self.w[self.set_J])
            
            self.beta_remaining[set_J_in_beta] = alpha_I
            self.set_remaining[set_J_in_beta] = self.set_I
        # reset set_I and set_J after per iteration
        self.set_I = []
        self.set_J = []
        self.cur_obj = self.cal_objective() 
        print(f'pre_obj={old_obj:.4e}, new_obj={self.cur_obj:.4e}, \n',
                f'pre_obj-new_obj={old_obj-self.cur_obj:.4e}, sum_reduction={-sum_reduct:.4e}, \n',
                f'(pre_obj-new_obj) - sum_reduction = {old_obj-self.cur_obj + sum_reduct:4e}')
        #import pdb;pdb.set_trace()
        return to_stop
    
    def perb_online(self, itr):
        if itr == 0:
            self.prune_init()
        else:   
            # Run mag_perb for a few iterations
            self.constructive_greedy_prune_one_iter(itr)
        return self.set_pruned

    def prune_online(self, itr):
        timer = Timer() 
        timer.start()
        self.iter_time = 0
        to_stop = False
        self.iter = itr
        if itr == 0:
            self.accumulate_time = 0
            #  perb_online() is its initialization 
            if self.init_method == 'mag_perb': 
                pass
            else:
                self.prune_init()
            self.dump_pruned(f'magnitude')
            self.accumulate_time += timer.stop('get intial alpha and beta')
        else:   
            # for one iteration:
            # Run mag_perb for a few iterations
            to_stop = self.prune_one_iter(itr)
            self.iter_time = timer.stop(f'RANGE={self.RANGE}, converged in {iter} iteration')
            self.accumulate_time += self.iter_time
            self.dump_pruned(f'iter_{itr}')
        #if not DEBUG:
        #    self.cal_delta_w()
        print('Time (s) in total: ', self.accumulate_time)
        print('pruned: ', len(self.set_pruned), )
        print('remaining: ', len(self.set_remaining), )
        self.cur_obj = self.cal_objective() 
        print('objective: ', self.cur_obj)
        #import pdb;pdb.set_trace()
        return to_stop, self.set_pruned

    def prune_old(self):
        # get initial sets by magnitude-based pruning
        if self.use_simulated_data:
            self.debug_init_with_simulated_data()
        else:
            if self.init_method == 'wg':
                self.init_with_wg()
            else:
                self.init_with_magnitude()
        #import pdb;pdb.set_trace()
        timer = Timer()
        self.accumulate_time = 0
        self.iter_time = 0
        timer.start()

        self.alpha_pruned   = self.cal_alpha_vec(part=4)
        self.beta_remaining = self.cal_beta_vec()
        self.iter_meta = Meta.set()
        self.dump_pruned(f'magnitude')
        self.accumulate_time += timer.stop('get intial alpha and beta')
        for itr in tqdm(range(self.max_iter)):
            iter_meta = Meta(self.idx_2_module_path)
            self.alpha_pruned, sorted_idx = torch.sort(self.alpha_pruned, descending=True)
            self.set_pruned = self.set_pruned[sorted_idx]
            #self.set_pruned_debug = self.set_pruned.clone()
            _, sorted_idx  = torch.sort(
                self.beta_remaining - self.cal_set_gamma(self.set_remaining, self.set_pruned[0]),
                descending=False)
            self.beta_remaining = self.beta_remaining[sorted_idx]
            self.set_remaining  = self.set_remaining[sorted_idx]
            self.set_I = []
            self.set_J = []
            #timer.stop('sorting alpha and beta')
            # check the firest elemtnes in set_pruned and set_remaining
            valid, allowed, reduction, meta = self.is_valid_swap(0,0)
            #import pdb;pdb.set_trace()
            if not valid:
               break
            else:
                if allowed:
                    self.set_I.append(self.set_pruned[0])
                    self.set_J.append(self.set_remaining[0])
                    print(f'i,j: 0,0, reduction={reduction}')
                    iter_meta.add(meta)
                    #import pdb;pdb.set_trace()
            no_match_cnt = 0
            set_I_in_alpha = [0] if allowed else []
            set_J_in_beta  = [0] if allowed else []
            is_in_J = torch.zeros(len(self.set_remaining))
            for i in range(1, len(self.set_pruned)):
            #for i in range(1, 5):
                if self.swap_one_per_iter:
                    break # DEBUG: only compare the first pair
                # if there are more than no_match_cnt_max weights in set_pruned didn't find matching
                if no_match_cnt > self.MAX_NO_MATCH:
                    break
                #timer.stop(f'swap: i={i}')

                for j in range(max(1,i-self.RANGE), min(len(self.set_remaining), i+self.RANGE)):
                    if is_in_J[j] == 0:
                        valid, allowed, reduction, meta = self.is_valid_swap(i,j)
                        if valid and allowed:
                            #print(f'i,j: {i},{j}, reduction={reduction}')
                            self.set_I.append(self.set_pruned[i])
                            self.set_J.append(self.set_remaining[j])
                            set_I_in_alpha.append(i)
                            set_J_in_beta.append(j)
                            is_in_J[j] = 1
                            iter_meta.add(meta)
                            print(f'swap: i={i}, j={j}')
                            #import pdb;pdb.set_trace()
                            break
                if is_in_J[j] == 0:
                    no_match_cnt += 1
            #print_active_tensors()
            #import pdb;pdb.set_trace()
            #timer.stop(f'swap done: i={i}, j={j}')
            # update alpha and beta
            self.set_I = torch.tensor(self.set_I)
            self.set_J = torch.tensor(self.set_J)
            print(f'len(set_I): {len(self.set_I)}, len(self_J): {len(self.set_J)}')
            if len(self.set_I) > 0:
                update_all = self.update_all_after_swap()
                # update alpha and beta
                self.alpha_pruned += update_all[self.set_pruned]
                self.beta_remaining += update_all[self.set_remaining]
                # Swap set_J (into set_pruned, alpha_pruned), set_I (into set_remaiming, beta_remaining)
                alpha_I = self.alpha_pruned[set_I_in_alpha]
                beta_J  = self.beta_remaining[set_J_in_beta]

                self.alpha_pruned[set_I_in_alpha] = beta_J
                self.set_pruned[set_I_in_alpha] = self.set_J

                self.beta_remaining[set_J_in_beta] = alpha_I
                self.set_remaining[set_J_in_beta] = self.set_I
            else:
                break
            self.set_I = []
            self.set_J = []
            self.iter_time = timer.stop(f'update all alpha and beta')
            self.accumulate_time += self.iter_time
            self.iter_meta = iter_meta.get()
            self.dump_pruned(f'iter_{itr}')
        self.iter_time = timer.stop(f'RANGE={self.RANGE}, converged in {iter} iteration')
        self.last_iter = itr
        #if not DEBUG:
        #    self.cal_delta_w()
        print('Time (s) in total: ', self.accumulate_time + self.iter_time)
        print('pruned: ', self.set_pruned, self.alpha_pruned)
        print('remaining: ', self.set_remaining, self.beta_remaining)
    
    def prune(self):
        timer = Timer() 
        self.accumulate_time = 0
        self.iter_time = 0
        timer.start()

        # init the alpha_pruned, beta_remaining
        self.prune_init()
        self.dump_pruned(f'magnitude')
        self.accumulate_time += timer.stop('get intial alpha and beta')
        for itr in tqdm(range(self.max_iter)):
            self.iter = itr
            to_stop =  self.prune_one_iter(itr)
            #self.iter_time = timer.stop(f'update all alpha and beta')
            self.iter_time = timer.stop(f'RANGE={self.RANGE}, converged in {iter} iteration')
            self.accumulate_time += self.iter_time
            self.dump_pruned(f'iter_{itr}')
            if to_stop:
                break
        self.last_iter = itr
        #if not DEBUG:
        #    self.cal_delta_w()
        print('Time (s) in total: ', self.accumulate_time + self.iter_time)
        print('pruned: ', self.set_pruned, self.alpha_pruned)
        print('remaining: ', self.set_remaining, self.beta_remaining)

    def set_greedy_dir(self):
        if not os.path.exists(self.greedy_dir):
            os.makedirs(self.greedy_dir)

    def dump_pruned(self, info=''):
        info = '-' + info if info != '' else ''
        name = get_greedy_exp_name(self.sparsity, self.THRESHOLD, self.RANGE, self.MAX_NO_MATCH)
        rst_path = os.path.join(self.greedy_dir,  name + info + '.npy')
        np.save(rst_path, {'pruned_idx': self.set_pruned, 'iter_time': self.iter_time, 
                        'accumulated_time': self.accumulate_time,
                        'meta': self.iter_meta,
                        'obj': self.cur_obj})
    
    def load_pruned(self):
        # load the last iteration results at default
        if self.greedy_path is None:
            rst_path = get_greedy_exp_paths(self.greedy_dir, 
                        self.sparsity, self.THRESHOLD, self.RANGE, self.MAX_NO_MATCH)
        else:
            rst_path = self.greedy_path 
        if rst_path is None:
            raise Exception(f'!!! greedy result not exist: {rst_path}')
            return None
        data        = _load(rst_path)
        return data['pruned_idx'].to(self.device)

    def set_greedy_path(self):
        if self.greedy_path is None:
            exp = get_greedy_exp_name(self.sparsity, self.THRESHOLD, self.RANGE, self.MAX_NO_MATCH)
            if self.iter == 0:
                exp += '-magnitude.npy'
            else:
                exp += f'-iter_{self.iter}.npy'
            self.greedy_path = os.path.join(self.greedy_dir, exp)

    def load_delta_w(self):
        rst_path = self.greedy_path
        delta_w = None
        if os.path.exists(rst_path):
            data     = _load(rst_path)
            key = f'delta_w.{self.args.weight_update_method}'
            if key in data.keys():
                delta_w = data[key]
        return delta_w

    def dump_delta_w(self, extra_dict):
        rst_path = self.greedy_path
        if not os.path.exists(self.greedy_path):
            data     = {'pruned_idx': self.set_pruned, 
                        'iter_time': self.iter_time, 'accumulated_time': self.accumulate_time,
                        'meta': self.iter_meta, 'obj': self.cal_objective()}
        else:
            data     = _load(rst_path)
        data[f'delta_w.{self.args.weight_update_method}'] = self.delta_w
        data.update(extra_dict)
        np.save(rst_path, data)

    def cal_delta_w_base(self, F_inv, pruned_idx, w, G_mean=0.0):
        if len(pruned_idx) == 0:
            return torch.zeros(len(w)).to(self.device)
        F_inv_KK = F_inv[pruned_idx,:][:,pruned_idx].to(self.device)
        w_K = w[pruned_idx]
        ## solve the linear system with variables lambda
        #   F_inv_KK @ lambda = w_K        
        offload_solve = True if len(w_K) > 20000 else False
        print(f'len(w_K): {len(w_K)}')
        #try: 
        #    lambda_star = torch.solve(w_K[:, None], F_inv_KK)
        #    offload_solve = True
        #except:
        #    import pdb;pdb.set_trace()
        #    torch.cuda.empty_cache()
        #    lambda_star = torch.solve(w_K[:, None].cpu(), F_inv_KK.cpu())
        #    offload_solve = False 
        if offload_solve:
            #lambda_star = torch.solve(w_K[:, None].cpu(), F_inv_KK.cpu())
            lambda_star = torch.solve(w_K[:, None].cpu(), F_inv_KK.cpu())
        else:
            lambda_star = torch.solve(w_K[:, None], F_inv_KK)
        del F_inv_KK
        torch.cuda.empty_cache()
        lambda_     = torch.zeros(len(w)).to(self.device)
        if offload_solve:
            lambda_[pruned_idx] = lambda_star[0].squeeze().to(self.device)
        else:
            lambda_[pruned_idx] = lambda_star[0].squeeze()
        
        if self.args.has_first_order_term_in_obj:
            print('lamba.norm={lambda_.norm()}, g_mean.norm={G_mean.norm()}')
            lambda_ += G_mean * 0.001
        
        #self.delta_w = - F_inv @ lambda_ 
        F_inv = F_inv.to(self.device)
        delta_w = - product_efficient_v1(F_inv, lambda_, num_parts=4) 
        del lambda_star, lambda_, F_inv
        torch.cuda.empty_cache()

        return delta_w

    def cal_delta_w(self):
        timer = Timer()
        timer.start()
        if self.decompose_F:
            del self.G
        else:
            del self.F
        torch.cuda.empty_cache()
        F_inv = load_fisher_inv(os.path.join(self.fisher_inv_path, 'fisher_inv.pkl'))
        if len(self.set_pruned) > 0:
            pruned_idx = self.set_pruned
        else:
            #if self.args.woodfisher_mask_path is not None:
            #    self.init_with_woodfisherblock()
            #    pruned_idx = self.set_pruned
            #else:
            #    pruned_idx = self.load_pruned()
            pruned_idx = self.load_pruned()
        pre_time = timer.stop('prepare data to obtain delta_w')
        self.delta_w = self.cal_delta_w_base(F_inv, pruned_idx, self.w, self.G_mean) 
        cal_delta_time = timer.stop('obtain delta_w')
        self.dump_delta_w({'pre_cal_delta_time': pre_time, 'cal_delta_time': cal_delta_time}) 
        return self.delta_w

    def cal_delta_w_old(self):
        timer = Timer()
        timer.start()
        if self.decompose_F:
            del self.G
        else:
            del self.F
        torch.cuda.empty_cache()
        F_inv = load_fisher_inv(os.path.join(self.fisher_inv_path, 'fisher_inv.pkl'))
        if len(self.set_pruned) > 0:
            pruned_idx = self.set_pruned
        else:
            pruned_idx = self.load_pruned()
        F_inv_KK = F_inv[pruned_idx,:][:,pruned_idx]
        w_K = self.w[pruned_idx]
        pre_time = timer.stop('prepare data to obtain delta_w')
        ## solve the linear system with variables lambda
        #   F_inv_KK @ lambda = w_K        
        lambda_star = torch.solve(w_K[:, None], F_inv_KK)
        del F_inv_KK
        torch.cuda.empty_cache()
        lambda_     = torch.zeros(len(self.w)).to(self.device)
        lambda_[pruned_idx] = lambda_star[0].squeeze()
        #self.delta_w = - F_inv @ lambda_ 
        self.delta_w = - product_efficient_v1(F_inv, lambda_, num_parts=2) 
        cal_delta_time = timer.stop('obtain delta_w')
        self.dump_delta_w({'pre_cal_delta_time': pre_time, 'cal_delta_time': cal_delta_time}) 

if __name__ == '__main__':    
    args = get_parse()
    args.greedy_method = 'greedy'
    debug_args(args) 
    greedy_pruner = GreedyPruner(sparsity=args.sparsity, wgh_path=args.wgh_path, device='cuda', args = args)
    if args.only_calculate_delta_w:
        greedy_pruner.cal_delta_w()
    else:
        greedy_pruner.prune()
