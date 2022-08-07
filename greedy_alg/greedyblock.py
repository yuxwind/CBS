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

class GreedyBlockPruner(GreedyPruner):
    def __init__(self, sparsity=0.6, wgh_path=None, weights=None, grads=None, fisher_matrix=None, device='cuda', args=None):
        super(GreedyBlockPruner, self).__init__(sparsity=sparsity, 
                wgh_path=wgh_path, weights=weights, grads=grads, 
                fisher_matrix=fisher_matrix, device=device, args=args)
        timer = Timer()
        timer.start()
        #import pdb;pdb.set_trace()
        #self.gw = self.G * self.w
        self.gw = None
        timer.stop('cal self.gw')
        #self.get_F_blocks()
        

    def efficient_mat_mul(self, mat1, mat2):
        N,d1,_ = mat1.shape
        N,_,d2 = mat2.shape
        # To allow 2GB is used at most
        bz      = math.ceil(2e9 / (d1*d2*8))
        K     = math.ceil(N/bz)
        print(f'N={N}, d1xd2={d1}x{d2}, K={K}, bz={bz}')
        rst    = 0
        for kk in range(K):
            b,e = kk * bz, min(N, (kk+1)*bz)
            rst += (mat1[b:e] @ mat2[b:e]).sum(dim=0)
            torch.cuda.empty_cache() 
        rst = rst/N
        return rst

    def get_F_blocks(self):
        self.F_blocks = []
        for i in range(self.N_layer):
            b_idx, e_idx = self.layer_2_w[i]
            g = self.G[:, b_idx:e_idx][:,:,None]
            #self.F_blocks.append((g @ g.transpose(1,2)).mean(dim=0))
            import pdb;pdb.set_trace()
            self.F_blocks.append(self.efficient_mat_mul(g, g.transpose(1,2)))
            import pdb;pdb.set_trace()
            del g
            torch.cuda.empty_cache()
            import pdb;pdb.set_trace()

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

    def mask_ij(self, i, j):
        return self.w_2_layer[i] == self.w_2_layer[j]

    def mask_IJ(self, I, J, unique_I = True):
        ''' whether the weights of indices I and J belongs to a same layer
            return:
                mask: |I| * |J|
        '''
        I_idx = self.w_2_layer[I]
        if unique_I:
            I_idx = torch.unique(I_idx)
        J_idx = self.w_2_layer[J]
        #import pdb;pdb.set_trace()
        mask = I_idx.repeat(len(J),1).T == J_idx
        return mask
   
    def get_F_ij(self, i, j):
        if not self.mask_ij(i,j):
            return 0.0
        if self.decompose_F:
            if self.args.offload_grads:
                if torch.is_tensor(i):
                    i = i.cpu()
                if torch.is_tensor(j):
                    j = j.cpu()
            return (self.G[:,i] * self.G[:,j]).mean(dim=0).to(self.device)
        else:
            return self.F_blocks[i_layer][i-s_layer, j-s_layer]
    
    # NOTE: unnecesary to get G_layer first
    def get_F_ij_new(self, i, j):
        i_layer = self.w_2_layer[i]
        j_layer = self.w_2_layer[j]
        
        if i_layer == j_layer:
            s_layer = self.layer_start_idx[i_layer]
            b,e     = self.layer_2_w[i_layer]
            G_layer = self.G[:, b:e]
            if self.decompose_F:
                rst = (G_layer[:,i] * G_layer[:,j]).mean(dim=0)
            else:
                rst = self.F_blocks[i_layer][i-s_layer, j-s_layer]
        else:
            rst = 0.0
        if DEBUG:
            assert(rst == self.get_F_ij1(i,j), 'Different with old get_F_ij()!!!')
        return rst

    def cal_HW_new(self, I, J, is_HW=True, part=1):
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
        timer = Timer()
        timer.start()
        I = I.to(self.device)
        J = J.to(self.device)
        rst   = torch.zeros([len(I)]).to(self.device)
        #timer.stop('111')
        I_layers = self.w_2_layer[I].unique()
        #timer.stop('112')
        J_layers = self.w_2_layer[J].unique()
        #timer.stop('113')
        IJ_layers = np.intersect1d(I_layers.unique().cpu().numpy(), J_layers.unique().cpu().numpy())
        #timer.stop('114')
        if len(IJ_layers) == 0:
            return rst
        mask_I  = self.mask_IJ(I,J)
        #timer.stop('115')
        di = dict(zip(I_layers.cpu().numpy(), range(len(I_layers))))
        #timer.stop('116')
        mask_J  = self.mask_IJ(J,I)
        #timer.stop('117')
        dj = dict(zip(J_layers.cpu().numpy(), range(len(J_layers))))
        #timer.stop('118')
        #J_w      = self.w[J]
        #print('layers:', IJ_layers)
        #for l_i in IJ_layers:
        #for k, l_i in enumerate(I_layers):
        for l_i in IJ_layers:
            #timer.stop('aaa1')
            j  = J[mask_I[di[l_i]]]
            i  = mask_J[dj[l_i]]
            #timer.stop(f'{l_i} starts 1')
            #I_ = I[I_layers==l_i]
            #timer.stop('111')
            #J_ = J[J_layers==l_i]
            #timer.stop('222')
            ##import pdb;pdb.set_trace()
            ##rst[I_layers==l_i] = (self.G[:,I_] * self.gw[:,J_].sum(dim=1)[:,None]).mean(dim=0)
            #aa = (self.G[:,I_] * self.gw[:,J_].sum(dim=1)[:,None]).mean(dim=0)
            #timer.stop('333')
            #rst[I_layers==l_i] = aa
            #timer.stop('444')
            #rst[I_layers==l_i] = (self.G[:, I[I_layers==l_i]] * self.gw[:,J[J_layers==l_i]].sum(dim=1)[:,None]).mean(dim=0)
            rst[i] = (self.G[:, I[i]] * self.gw[:,j].sum(dim=1)[:,None]).mean(dim=0)
        
        #t1 = timer.stop('++++')
        #import pdb;pdb.set_trace()
        #rst1 = rst.clone()
        #timer.stop('')
        mask  = self.mask_IJ(I,J)
        #timer.stop('aaa')
        I_idx = torch.unique(self.w_2_layer[I])
        I = I.to(self.device)
        J = J.to(self.device)
        #I_layers = self.w_2_layer[I]
        #J_layers = self.w_2_layer[J]
        #IJ_layers = np.intersect1d(I_layers.unique().cpu().numpy(), J_layers.unique().cpu().numpy())
        if self.decompose_F:
            #gw     = self.G[:,J] @ self.w[J] # N x 1
            for k, Ii in enumerate(I_idx):
            #for k, Ii in enumerate(IJ_layers):
                j  = J[mask[k]]
                #timer.stop('aaa1')
                if len(j) > 0:
                    gw = product_efficient_v1(self.G[:,j],  self.w[j], num_parts=part)
                    #timer.stop('aaa2')
                    i  = self.w_2_layer[I] == Ii 
                    #timer.stop('aaa3')
                    rst[i] = (self.G[:, I[i]] * gw[:,None]).mean(dim = 0)
                    #timer.stop('aaa4')
                    del gw
        t2 = timer.stop('====')
        print(len(I), len(J), f'new - old = {t1 - t2}')
            #timer.stop(f'{l_i} stops 1 ###########')
            #timer.stop(f'{l_i}-th layer is done')
            #if len(J_) > 0:
            #     timer.stop('000')
            #     i_in_layer = I_ - self.layer_start_idx[l_i] 
            #     timer.stop('001')
            #     j_in_layer = J_ - self.layer_start_idx[l_i]
            #     timer.stop('002')
            #     j_w        = J_w[J_layers==l_i]
            #     timer.stop('003')
            #     b,e     = self.layer_2_w[l_i]
            #     timer.stop('004')
            #     G_layer = self.G[:, b:e]
            #     timer.stop('005')
            #     if self.decompose_F:
            #         timer.stop('')
            #         gw     = product_efficient_v1(G_layer[:,j_in_layer], j_w, num_parts=part) 
            #         timer.stop('555')
            #         #rst[I_layers==l_i] = (G_layer[:,i_in_layer] * gw[:,None]).mean(dim=0)     
            #         bb = (G_layer[:,i_in_layer] * gw[:,None]).mean(dim=0)     
            #         timer.stop('666')
            #         del gw
            #         if not (torch.isclose(aa,bb).all()):
            #             import pdb;pdb.set_trace()
            #         #rst[I_layers==l_i] = bb
            #     else:
            #         rst[I_layers==l_i] = self.F_block[l_i][np.ix_(i_in_layer, j_in_layer)] @ j_w
            #timer.stop(f'{l_i} stops 2+++++++')
        DEBUG=False
        import pdb;pdb.set_trace()
        if DEBUG:
            if not is_HW:
                is_HW = True
            rst_ = self.cal_HW1(I,J,is_HW, part)
            if not torch.isclose(rst, self.cal_HW1(I,J,is_HW, part)).all(): 
                    print('Different with old cal_HW()!!!')
                    import pdb;pdb.set_trace()
        return rst 

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
        I_idx = torch.unique(self.w_2_layer[I])
        I = I.to(self.device)
        J = J.to(self.device)

        if self.decompose_F:
            #gw     = self.G[:,J] @ self.w[J] # N x 1
            for k, Ii in enumerate(I_idx):
                j  = J[mask[k]]
                if len(j) > 0:
                    gw = product_efficient_v1(self.G[:,j],  self.w[j], num_parts=part)
                    i  = self.w_2_layer[I] == Ii 
                    try:
                        #rst[i] = (self.G[:, I[i]] * gw[:,None]).mean(dim = 0)
                        rst[i] = g_prod_gw_efficient(self.G[:, I[i]], gw) 
                    except:
                        import pdb;pdb.set_trace()
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
   
    def cal_delta_w(self):
        timer = Timer()
        timer.start()
        if self.decompose_F:
            del self.G
            del self.gw
        else:
            del self.F
        torch.cuda.empty_cache()
        # get pruned_idx: load the last iteration if it doesn't exist
        if len(self.set_pruned) > 0:
            pruned_idx = self.set_pruned
        else:
            pruned_idx = self.load_pruned()
        # No pruning results found
        if pruned_idx is None:
            return 

        self.delta_w = torch.zeros_like(self.w).to(self.device)
        # update every layer
        
        for l in range(self.N_layer):
            fp = os.path.join(self.fisher_inv_path, f'block_fisher_inv_{l}.pkl')
            print(fp)
            l_F_inv = load_fisher_inv(fp)
            start, end = self.layer_2_w[l]
            l_w   = self.w[start:end]
            if self.args.has_first_order_term_in_obj:
                l_G_mean = self.G_mean[start:end]
            else:
                l_G_mean = None
            #import pdb;pdb.set_trace()
            l_pruned_idx = pruned_idx[self.w_2_layer[pruned_idx] == l] - start
            if type(l_F_inv) == list:
                N_blk = len(l_F_inv)
                s_, e_ = 0,0
                for b in range(N_blk):
                    s_ = e_
                    e_ = e_ + l_F_inv[b].shape[0]
                    b_w = l_w[s_:e_]
                    if self.args.has_first_order_term_in_obj:
                        b_G_mean = l_G_mean[s_:e_]
                    else:
                        b_G_mean = None
                    b_pruned_idx = l_pruned_idx[
                            torch.logical_and(l_pruned_idx < e_, l_pruned_idx >= s_)] - s_
                    b_delta_w = self.cal_delta_w_base(
                                    l_F_inv[b], b_pruned_idx, b_w, b_G_mean)
                    self.delta_w[start+s_:start+e_] = b_delta_w
            else:
                self.delta_w[start:end] = self.cal_delta_w_base(
                                l_F_inv, l_pruned_idx, l_w, l_G_mean)
            del l_F_inv, l_w, l_pruned_idx
            torch.cuda.empty_cache()

        cal_delta_time = timer.stop('obtain delta_w in combinatorial way')
        self.dump_delta_w({'cal_delta_time': cal_delta_time}) 
        return self.delta_w

    #########################################################################
    # this code is derived from computing global update in pruners/woodfisherblock.py
    #########################################################################
    def cal_delta_w_not_comb_base(self, l_F_inv, l_pruned_idx, flattened_params):
        l_F_inv = l_F_inv.to(self.device)
        _block_fisher_inv_diag = l_F_inv.diagonal()
        pruned_params  = torch.zeros(len(flattened_params)).to(self.device)
        pruned_params[l_pruned_idx] = 1
        scaled_basis_vector = -1 * torch.div(torch.mul(pruned_params, flattened_params), _block_fisher_inv_diag)
        weight_update = l_F_inv @ scaled_basis_vector

        _zero_after_prune = False
        if _zero_after_prune:
            weight_update[pruned_params.bool()] = (-1 * flattened_params[pruned_params.bool()])
        
        del _block_fisher_inv_diag, pruned_params, scaled_basis_vector, l_F_inv
        torch.cuda.empty_cache()

        return weight_update

    def cal_delta_w_not_comb(self):
        timer = Timer()
        timer.start()
        if self.decompose_F:
            del self.G
            del self.gw
        else:
            del self.F
        torch.cuda.empty_cache()
        # get pruned_idx: load the last iteration if it doesn't exist
        if len(self.set_pruned) > 0:
            pruned_idx = self.set_pruned
        else:
            #if self.args.woodfisher_mask_path is not None:
            #    self.init_with_woodfisherblock()
            #    pruned_idx = self.set_pruned
            #else:
            #    pruned_idx = self.load_pruned()
            pruned_idx = self.load_pruned()
        # No pruning results found
        if pruned_idx is None:
            return 

        self.delta_w = torch.zeros_like(self.w).to(self.device)
        # update every layer
        
        for l in range(self.N_layer):
            fp = os.path.join(self.fisher_inv_path, f'block_fisher_inv_{l}.pkl')
            print(fp)
            timer.stop()
            l_F_inv = load_fisher_inv(fp)
            timer.stop('load fisher_inv')
            start, end = self.layer_2_w[l]
            flattened_params = self.w[start:end].to(self.device)
            l_pruned_idx = pruned_idx[self.w_2_layer[pruned_idx] == l] - start 
            _param_count = start
            
            if type(l_F_inv) == list:
                N_blk = len(l_F_inv)
                s_, e_ = 0,0
                l_delta_w = []
                for b in range(N_blk):
                    s_ = e_
                    e_ = e_ + l_F_inv[b].shape[0]
                    b_w = flattened_params[s_:e_]
                    b_pruned_idx = l_pruned_idx[
                            torch.logical_and(l_pruned_idx < e_, l_pruned_idx >= s_)] - s_
                    b_delta_w = self.cal_delta_w_not_comb_base(l_F_inv[b], b_pruned_idx, b_w)
                    self.delta_w[start+s_:start+e_] = b_delta_w
            else: 
                self.delta_w[start:end] = self.cal_delta_w_not_comb_base(l_F_inv, l_pruned_idx, flattened_params)

            del l_F_inv, flattened_params, l_pruned_idx, 
            torch.cuda.empty_cache()

        cal_delta_time = timer.stop('obtain delta_w in woodfisher way')
        self.dump_delta_w({'cal_delta_time.not_comb': cal_delta_time}) 
        return self.delta_w

    def debug(self):
        timer = Timer()
        timer.start()
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
        #hw2[:len(i1)] = self.cal_HW(i1, j1)
        timer.stop('prepare')
        hw2[:len(i1)] = self.cal_HW(i1, j1)
        hw2[len(i1):] = 0.0
        timer.stop('cal_HW')
        assert torch.isclose(hw1, hw2).all()
        timer.stop('assert')
        hw2[:len(i1)] = self.cal_HW(i1, j1)
        hw2[len(i1):] = 0.0
        timer.stop('cal_HW1')
        import pdb;pdb.set_trace()

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
