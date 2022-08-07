import torch
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import time
import logging
from utils.utils import flatten_tensor_list, get_summary_stats, dump_tensor_to_mat
from policies.pruners import GradualPruner
import math
import numpy as np
import pickle
from code_zheng.BLRSSEP import BLRSSEP
import os
from common.io import _load, _dump
from greedy_alg.greedy_io import save_idx_2_module
from greedy_alg.greedy import GreedyPruner
from greedy_alg.greedyblock import GreedyBlockPruner
from greedy_alg.greedylayer import GreedyLayerPruner

import matplotlib.pyplot as plt

class CombinatorialPruner(GradualPruner):
    def __init__(self, model, inp_args, **kwargs):
        super(CombinatorialPruner, self).__init__(model, inp_args, **kwargs)
        print("IN CombinatorialPruner")
        self._fisher_inv_diag = None
        self._prune_direction = inp_args.prune_direction
        self._zero_after_prune = inp_args.zero_after_prune
        self._inspect_inv = inp_args.inspect_inv
        self._fisher_mini_bsz = inp_args.fisher_mini_bsz
        if self._fisher_mini_bsz < 0:
            self._fisher_mini_bsz = 1
        if self.args.woodburry_joint_sparsify:
            self._param_stats = []
        if self.args.dump_fisher_inv_mat:
            self._all_grads = []
        self.set_paths()
        self.greedy_pruner = None
        if self.args.fisher_inv_path is None:
            N_samples = self.args.fisher_subsample_size * self.args.fisher_mini_bsz
            N_batches = self.args.fisher_subsample_size
            seed      = self.args.seed
            self.args.fisher_inv_path = os.path.join('./prob_regressor_data',
                f'{self.args.arch}_{self.args.dset}_{N_samples}samples_{N_batches}batches_{seed}seed.fisher_inv')

    def _compute_sample_grad_weight(self, loss):

        ys = loss
        params = []
        for module in self._modules:
            for name, param in module.named_parameters():
                # print("name is {} and shape of param is {} \n".format(name, param.shape))

                if self._weight_only and 'bias' in name:
                    continue
                else:
                    params.append(param)

        grads = torch.autograd.grad(ys, params)  # first order gradient

        # Do gradient_masking: mask out the parameters which have been pruned previously
        # to avoid rogue calculations for the hessian

        for idx, module in enumerate(self._modules):
            grads[idx].data.mul_(module.weight_mask)
            params[idx].data.mul_(module.weight_mask)

        grads = flatten_tensor_list(grads)
        params = flatten_tensor_list(params)

        #if self.args.dump_grads_mat:
        #    self._all_grads.append(grads)

        self._num_params = len(grads)

        gTw = params.T @ grads
        return grads, gTw

    def _reset_masks(self, init_past_weight_masks):
        if init_past_weight_masks is None:
            return
        for idx, module in enumerate(self._modules):
            assert args._weight_only
            module.weight_mask = init_past_weight_masks[idx] 
            assert (module.weight_mask == init_past_weight_masks[idx]).all()
    
    # Revert the weights to before pruning
    # NOTE: the weights will change after applying masks
    def revert_weight(self):
        for idx, module in enumerate(self._modules):
            module.revert_pruned()

    def _compute_sample_loss(self, dset, device, num_workers, subset_inds = None):
        #subset_inds = np.arange(self.args.batch_size*2)
        if subset_inds is None:
            dummy_loader = torch.utils.data.DataLoader(dset, batch_size=self.args.batch_size, 
                            num_workers=num_workers, shuffle=False)
        else:
            dummy_loader = torch.utils.data.DataLoader(dset, batch_size=self.args.batch_size, 
                            num_workers=num_workers,
                            sampler=SubsetRandomSampler(subset_inds), shuffle=False)
        if self.args.disable_log_soft:
            # set to true for resnet20 case
            # set to false for mlpnet as it then returns the log softmax and we go to NLL
            criterion = torch.nn.functional.cross_entropy
        else:
            criterion = F.nll_loss

        sample_loss = 0.0 
        self._model.eval()
        #import pdb;pdb.set_trace()
        #i = 0
        with torch.no_grad():
            for in_tensor, target in dummy_loader:
                #self._release_grads()
                in_tensor, target = in_tensor.to(device), target.to(device)
                output = self._model(in_tensor)
                sample_loss += criterion(output, target, reduction='sum').item()
                #i += 1
                #print(f'batch: {i}')
        #import pdb;pdb.set_trace()
        if subset_inds is None:
            sample_loss /= len(dset)
        else:
            sample_loss /= len(subset_inds)

        self.revert_weight()
        return sample_loss

    def _compute_sample_fisher(self, loss, return_outer_product=False):
        ''' Inputs:
                loss: scalar or B, Bx1 tensor
            Outputs:
                grads_batch: BxD 
                gTw: B (grads^T @ weights)
                params: D
                ff: 0.0 or DxD(sum of grads * grads^T) 
        '''
        ys = loss
        params = []
        for module in self._modules:
            for name, param in module.named_parameters():
                # print("name is {} and shape of param is {} \n".format(name, param.shape))
                if self._weight_only and 'bias' in name:
                    continue
                else:
                    params.append(param)
        
        grads = torch.autograd.grad(ys, params)  # first order gradient

        # Do gradient_masking: mask out the parameters which have been pruned previously
        # to avoid rogue calculations for the hessian

        for idx, module in enumerate(self._modules):
            grads[idx].data.mul_(module.weight_mask)
            params[idx].data.mul_(module.weight_mask)

        grads = flatten_tensor_list(grads)
        params = flatten_tensor_list(params)

        if self.args.dump_grads_mat:
            self._all_grads.append(grads)

        self._num_params = len(grads)
        self._old_weights = params

        gTw = params.T @ grads

        if not return_outer_product:
            #return grads, grads, gTw, params
            return grads, None, gTw, params
        else:
            return grads, torch.ger(grads, grads), gTw, params

    def _get_pruned_wts_scaled_basis(self, pruned_params, flattened_params):
        return -1 * torch.div(torch.mul(pruned_params, flattened_params), self._fisher_inv_diag)

    @staticmethod
    def _get_param_stat(param, param_mask, fisher_inv_diag, param_idx):
        if param is None or param_mask is None: return None
        # w_i **2 x ((F)^-1)_ii,
        inv_fisher_diag_entry = fisher_inv_diag[param_idx: param_idx + param.numel()].view_as(param)
        inv_fisher_diag_entry = inv_fisher_diag_entry.to(param.device)
        print("mean value of statistic without eps = {} is ".format(1e-10),
              torch.mean((param ** 2) / inv_fisher_diag_entry))
        print(
            "std value of statistic without eps = {} is ".format(1e-10),
            torch.std((param ** 2) / inv_fisher_diag_entry))
        return ((param ** 2) / (inv_fisher_diag_entry + 1e-10) + 1e-10) * param_mask

    def _release_grads(self):
        optim.SGD(self._model.parameters(), lr=1e-10).zero_grad()

    def _add_outer_products_efficient_v1(self, mat, vec, num_parts=2):
        piece = int(math.ceil(len(vec) / num_parts))
        vec_len = len(vec)
        for i in range(num_parts):
            for j in range(num_parts):
                mat[i * piece:min((i + 1) * piece, vec_len), j * piece:min((j + 1) * piece, vec_len)].add_(
                    torch.ger(vec[i * piece:min((i + 1) * piece, vec_len)],
                              vec[j * piece:min((j + 1) * piece, vec_len)])
                )

    def _compute_wgH(self,dset, subset_inds, device, num_workers, debug=False):
        st_time = time.perf_counter()
        
        #path_grads = self.
        
        self._model = self._model.to(device)

        print("in woodfisher: len of subset_inds is ", len(subset_inds))

        goal = self.args.fisher_subsample_size

        assert len(subset_inds) == goal * self.args.fisher_mini_bsz

        dummy_loader = torch.utils.data.DataLoader(dset, batch_size=self._fisher_mini_bsz, 
                num_workers=num_workers,
                sampler=SubsetRandomSampler(subset_inds))
        ## get g and g^T * w, denoted as XX and yy respectively
        Gs = []
        GTWs = []
        
        if self.args.aux_gpu_id != -1:
            aux_device = torch.device('cuda:{}'.format(self.args.aux_gpu_id))
        else:
            aux_device = torch.device('cpu')

        if self.args.disable_log_soft:
            # set to true for resnet20 case
            # set to false for mlpnet as it then returns the log softmax and we go to NLL
            criterion = torch.nn.functional.cross_entropy
        else:
            criterion = F.nll_loss

        self._fisher_inv = None

        num_batches = 0
        num_samples = 0

        FF = 0.0
        for in_tensor, target in dummy_loader:
            self._release_grads()

            in_tensor, target = in_tensor.to(device), target.to(device)
            output = self._model(in_tensor)
            try:
                loss = criterion(output, target, reduction='none')
            except:
                import pdb;pdb.set_trace()
            # The default reduction = 'mean' will be used, over the fisher_mini_bsz,
            # which is just a practical heuristic to utilize more datapoints

            ## compute grads, XX, yy
            g, _, gTw, w = self._compute_sample_fisher(loss, return_outer_product=False)
            Gs.append(g[None,:].detach().cpu().numpy())
            #GTWs.append(gTw[None,None].detach().cpu().numpy())
            w = w.detach().cpu().numpy()
            #FF += ff
            del g, gTw

            num_batches += 1
            num_samples += self._fisher_mini_bsz
            if num_samples == goal * self._fisher_mini_bsz:
                break

        ## save Gs and GTWs
        #grads = torch.cat(Gs, 0) * 1 / np.sqrt(self.args.fisher_subsample_size)
        #wTgs = torch.cat(GTWs, 0) * 1/np.sqrt(self.args.fisher_subsample_size)
        grads = torch.cat(Gs, 0) 
        wTgs = torch.cat(GTWs, 0)
        #FF = FF / self.args.fisher_subsample_size
        print("# of examples done {} and the goal (#outer products) is {}".format(num_samples, goal))
        print("# of batches done {}".format(num_batches))

        end_time = time.perf_counter()
        print("Time taken to compute fisher inverse with woodburry is {} seconds".format(str(end_time - st_time)))

        return grads, GTWs, w, ff
    
    def set_paths(self):
        arch_info = '' # resnet20: ['', '_allweights']
        self.wgF_path = './prob_regressor_data/' + \
                f'{self.args.arch}_{self.args.dset}_{self.args.num_samples}samples_{self.args.fisher_subsample_size}batches_{self.args.seed}seed{arch_info}.pkl'
        self.grads_path = os.path.splitext(self.wgF_path)[0] + '.fisher_inv/all_grads.pkl'

    def dump_data(self, grads, wTgs, weights, fisher_matrix):
        wgF = {'g': grads, 'gTw': wTgs, 'W': weights, 'F': fisher_matrix}
        with open(self.wgF_path, 'wb') as wgF_f:
            pickle.dump(wgF, wgF_f, pickle.HIGHEST_PROTOCOL)

    def _to_tensor(self, data, device=None):
        if type(data) == np.ndarray:
            data = torch.tensor(data)
        if type(data) == torch.Tensor and device is not None:
            data = data.to(device)
        return data

    def load_data(self, device):
        data = _load(self.wgF_path)
        g = self._to_tensor(data["g"], device)
        w = self._to_tensor(data["W"], device)
        gTw = self._to_tensor(data["gTw"], device)
        F =   self._to_tensor(data["F"],   device)
        return g, gTw, w, F
        
    
    def load_enet_w(self): # load results of elastic net
        pass

    def analyze_new_weights(self, new_w, old_w, score):
        rho = 1 - self.args.target_sparsity
        x = np.arange(new_w.shape[0])
        plt.plot(x, old_w, 'b', label='old_w')
        plt.plot(x, new_w, 'g', label='new_w')
        s = score.abs().mean()
        scaling_factor = int(1/s) if s > 0 else 1
        plt.plot(x, score * scaling_factor, label=f'score*{scaling_factor}')
        plt.legend()
        plt.savefig('weigth_diff.pdf')

    def load_greedy_mask(self, require_updated_weight=False):
        # load original model
        wgh_data = _load(self.args.wgh_path)
        if self.args.wgh_path.endswith('.npy'):
            wgh_data = wgh_data.item()
        w = torch.tensor(wgh_data['W'])

        # load pruned set from greedy algorithm
        data = np.load(self.args.greedy_path, allow_pickle=True).item()
        ## load the new remained weights by greedy
        #comb_ana_data = np.load(os.path.join(root, 'comb_remained_analysis.npy'), allow_pickle=True).item()
        #remained_sets = comb_ana_data['remained_sets']
        #diff_sets = comb_ana_data['diff_sets']

        N_weights = len(w)
        mask = torch.ones(N_weights)
        set_pruned = data['pruned_idx']
        ## DEBUG: 
        #I = [25816,28475, 25817,23510,229799,1599,125913,16046,78596,122658,5010,]
        #J = [270688,270642,122566,370,270708,1463,2023,270755,20,116658,50934,]
        #end = 8
        #I = I[end-1:end]
        #J = J[end-1:end]
        #for cnt, ii in enumerate(I):
        #    ii_idx = (set_pruned == ii).nonzero(as_tuple=True)[0]
        #    set_pruned[ii_idx] = J[cnt] 
        #
        #import pdb;pdb.set_trace()
        
        #if 'delta_w' in data.keys() and not self.args.not_update_weights:
        if require_updated_weight and not self.args.not_update_weights:
            try:
                s_key = f'delta_w.{self.args.weight_update_method}'
                #if s_key not in data and self.args.weight_update_method == 'multiple':
                #    s_key = 'delta_w'
                w = w + data[s_key].cpu() * self.args.scale_prune_update
                updated_weight = True

            except:
                import pdb;pdb.set_trace()
                raise('!!!!!No updated_weight!!!!! in {self.args.greedy_path}')
        else:
            updated_weight = False
        mask[set_pruned] = 0
        ### test efficiency of each new weights
        #mask = torch.zeros(N_weights)
        #set_remained = remained_sets[iter+1]
        #diff_set    = diff_sets[iter]
        #mask[set_remained] = 1
        #### exclude a new weight to test its efficiency
        #exclude_w   = diff_set[self.args.greedy_iter]
        #mask[exclude_w] = 0
        #assert exclude_w in set_remained, 'ecluded_w should be in set_remained!'
        #print(f'!!!Greedy!!! not include {self.args.greedy_iter}-th weights in diff_set')

        #import pdb;pdb.set_trace()
        #new_w =  w * mask
        new_w = w # Anyway, delta_w will make the pruned weight to be 0 

        #sparsity = 1 - len(set_remained)/len(w)
        # magnituded-based pruning
        #_, thresholds = torch.topk(w.abs(), descending=True)
        #mask_mag = w.abs() >= thresholds[-1]
        

        #print('!!!!sparsity in greedy!!!!!', 1-len(set_remained)/len(w))
        print('!!!!sparsity in greedy!!!!!', len(set_pruned)/len(w))
        return mask, new_w, updated_weight


    def load_ep_mask(self):
        rho = 1 - self.args.target_sparsity 
        #data_root = 'prob_regressor_results' 
        data_root = 'code_zheng'
        # 100,50,15,10,5,0.5,0.1
        scaling_factor = 1.0
        # load new weight and score
        # the default precision of decimal in python is 28, round rho and remove the most right zeros 
        s_rho = f'{rho:.6f}'.rstrip('0')
        if self.args.regressor_result_path is None:
            self.mask_w_path = os.path.join(data_root,
                    f'mlp_mnist_{self.args.fisher_subsample_size}_rho_{s_rho}_scaling_{scaling_factor:.1f}.npy')
        else:
            self.mask_w_path = self.args.regressor_result_path
        print(self.mask_w_path)
        #import pdb;pdb.set_trace()
        if not os.path.exists(self.mask_w_path):
            return None
        
        # load old weights
        num_samples = self.args.fisher_subsample_size
        self.wgF_path = './prob_regressor_data/' + \
                f'mlp_mnist_{num_samples}samples_{self.args.fisher_subsample_size}batches.pkl'
        with open(self.wgF_path, 'rb') as f:
            data = pickle.load(f)
            old_w = torch.tensor(data['W']).float()

        mask_w = np.load(self.mask_w_path, allow_pickle=True).item()
        score = torch.tensor(mask_w['score']).squeeze()
        new_w = torch.tensor(mask_w['w_mean']).squeeze()
        self.analyze_new_weights(new_w, old_w, score)
        
        #### Process Score
        # Score 1
        #score = (score - rho).abs() 
        # Score 2: magnitude-based
        #score = old_w.abs()
        # Score 3: prune weights of larges score
        score = - score

        ####  Calclulate mask
        # Play with the target network density
        #rho = 0.5 
        num_params_to_keep = int(new_w.shape[0] * rho)
        threshold,out = torch.topk(score, num_params_to_keep, sorted=True)
        acceptable_score = threshold[-1]
        # Mask1: prune the weights of the smallest scores
        mask = score >= acceptable_score
        # Mask2: random mask
        #mask = torch.rand(new_w.shape[0]) <= rho
        # Mask3: not to prune any weights
        #mask = score >= -1000
        
        return mask.float(), new_w
    
    def _get_masks(self):
        weight_masks = []
        
        for idx, module in enumerate(self._modules):
            assert self._weight_only
            weight_masks.append(module.weight_mask)
        return weight_masks
    
    def _get_weights(self):
        weights = []
        
        for idx, module in enumerate(self._modules):
            assert self._weight_only
            weights.append(module.weight.data.flatten())
        weights = torch.cat(weights).to(module.weight.device)
        return weights

    def set_idx_2_module(self):
        idx_2_module = {}
        e_idx = 0
        if not os.path.exists(self.args.idx_2_module_path):
            for idx, module in enumerate(self._modules):
                s_idx = e_idx
                e_idx = e_idx + module.weight.numel()
                idx_2_module[(s_idx, e_idx)] = self._module_names[idx]
            save_idx_2_module(self.args.idx_2_module_path, idx_2_module) 

    def update_masks(self, masks, new_weights=None):
        module_shapes_list = []
        module_param_indices_list = []
        past_weight_masks = []
        
        self._param_idx = 0
        #idx_2_module = {}
        for idx, module in enumerate(self._modules):
            assert self._weight_only
            # multiplying by the current mask makes the corresponding statistic
            # of those weights zero and keeps them removed.
            print(f'Before pruning for param {idx}: norm of weight is {torch.norm(module.weight).item()}') 
            past_weight_masks.append(module.weight_mask)
            module_param_indices_list.append(self._param_idx)
            assert self._weight_only
            module_shapes_list.append(module.weight.shape)
            s_idx = self._param_idx
            e_idx = self._param_idx + module.weight.numel()
            #idx_2_module[(s_idx, e_idx)] = self._module_names[idx]
            module.weight_mask = masks[s_idx: e_idx].view_as(module.weight_mask)
            if not self.args.not_update_weights and new_weights is not None:
                with torch.no_grad():
                    #module.weight[:] = new_weights[s_idx: e_idx].view_as(module.weight).float()
                    module.weight.copy_(new_weights[s_idx: e_idx].view_as(module.weight).float())
            self._param_idx += module.weight.numel()

            if module.bias is not None and not self._weight_only and new_weights is not None:
                print('sparsifying bias as well')
                s_idx = self._param_idx
                e_idx = self._param_idx + module.bias.numel()
                module.bias_mask =  masks[s_idx: e_idx].view_as(module.bias_mask)
                with torch.no_grad():
                    module.bias.data = new_weights[s_idx: e_idx].view_as(module.bias)
                self._param_idx += module.bias.numel()
                
            print(f'After pruning for param {idx}: norm of weight is {torch.norm(module.weight).item()}')
        #save_idx_2_module(self.args.idx_2_module_path, idx_2_module) 
        self._release_grads()
        #import pdb;pdb.set_trace()
        assert self._param_idx == masks.shape[0]

        return past_weight_masks

    ###############################################################################################
    # Monitor the training loss and decide whether to stop the online combinatorial pruning
    ###############################################################################################
    def _stat_sample_losses(self, sample_losses, t_reduce=1e-4, max_N_fluctation=5):
        N_itr = len(sample_losses)
        WILL_STOP = True
        if N_itr < 2:
            return not WILL_STOP
        sample_losses = torch.tensor(sample_losses)
        pre_min, pre_idx = torch.min(sample_losses[:-1], 0)
        if sample_losses[-1] > pre_min and len(sample_losses) - 1 - pre_idx >= max_N_fluctation:
            return WILL_STOP
        else:
            return not WILL_STOP
        #reduce = sample_losses[1:] - sample_losses[:-1]
        ## there is fluctation if the loss reduction is in (-t_reduce, t_reduce)
        #fluct  = reduce.abs() <= t_reduce
        #for i in range(N_itr - 3, 0, -1):
        #    fluct[i] = fluct[i] * fluct[i+1]
        #continuos_fluct = fluct.sum()
        ## Two cases to stop pruning: 
        ##   1. the last loss don't reduce or 
        ##   2. the last max_N_fluctation iterations become a plateau
        #if reduce[-1] < t_reduce and continuos_fluct <= max_N_fluctation:
        #    return not WILL_STOP
        #else:
        #    return WILL_STOP

    def set_greedy_pruner(self, device=None, desired_level=None):
        if self.greedy_pruner is None:
            if desired_level is None:
                desired_level = self.args.target_sparsity
            #import pdb;pdb.set_trace()
            if self.args.greedy_method == 'greedyblock':
                self.greedy_pruner = GreedyBlockPruner(
                        sparsity=desired_level, 
                        wgh_path=self.args.wgh_path, 
                        device=device, args=self.args)
            elif self.args.greedy_method == 'greedy':
                self.greedy_pruner = GreedyPruner(
                        sparsity=desired_level, 
                        wgh_path=self.args.wgh_path, 
                        device=device, args=self.args)
            elif self.args.greedy_method == 'greedylayer':
                self.greedy_pruner = GreedyLayerPruner(
                        sparsity=desired_level, 
                        wgh_path=self.args.wgh_path, 
                        device=device, args=self.args)
            else:
                import pdb;pdb.set_trace()


    def update_weights_online(self):
        DEBUG = False
        if DEBUG:
            delta_w_multiple = self.greedy_pruner.cal_delta_w()
            delta_w_single  = self.greedy_pruner.cal_delta_w_not_comb()
            scale_old = self.args.scale_prune_update
            scales = np.arange(10) * 0.1
            for s in scales:
                self.args.scale_prune_update = s
                print('\n')            
                print('scale_prune_update:', self.args.scale_prune_update)
                print('obj for selection', self.greedy_pruner.cal_objective())
                print('obj for multiple:', self.greedy_pruner.cal_objective_after_update(delta_w_multiple))
                print('obj for single:', self.greedy_pruner.cal_objective_after_update(delta_w_single))
            self.args.scale_prune_update = scale_old

            if self.args.weight_update_method == 'multiple':
                delta_w = delta_w_multiple
            else:
                delta_w = delta_w_single
        else:
            delta_w = self.greedy_pruner.load_delta_w()
            if delta_w is None:
                if self.args.weight_update_method == 'multiple':
                    delta_w = self.greedy_pruner.cal_delta_w()
                else:
                    delta_w = self.greedy_pruner.cal_delta_w_not_comb()
        w = self._get_weights() + delta_w * self.args.scale_prune_update
        return w
        
    #def on_epoch_begin(self, dset, subset_inds, device, num_workers, epoch_num, **kwargs):
    def on_epoch_begin(self, dset, subset_inds, device, num_workers, epoch_num, testset, **kwargs):
        meta = {}
        if self._pruner_not_active(epoch_num):
            print("Pruner is not ACTIVEEEE yaa!")
            return False, {}

        # ensure that the model is not in training mode, this is importance, because
        # otherwise the pruning procedure will interfere and affect the batch-norm statistics
        assert not self._model.training

        # reinit params if they were deleted during gradual pruning
        if not hasattr(self, '_all_grads'):
            self._all_grads = None
        if not hasattr(self, '_param_stats'):
            self._param_stats = []
        self.set_idx_2_module() 

        online_greedy = self.args.when_to_greedy == 'online'
        offline_fisher_matrix = False
        debug_update_weight   = self.args.ablation_study_update_weight
        level = self._required_sparsity(epoch_num)
        if not os.path.exists(self.wgF_path):
            if os.path.exists(self.grads_path):
                weights = self._get_weights()
                grads   = self.grads_path
                wTgs    = None
                fisher_matrix = None
            else:
                # TODO: make this consistent with woodfisherblock._compute_all_grads()
                grads, wTgs, weights, fisher_matrix = self._compute_wgH( 
                    dset, subset_inds, device, num_workers, debug=False)
            self.dump_data(grads, wTgs, weights, fisher_matrix)
        else:
            #import pdb;pdb.set_trace()
            grads, wTgs, weights, fisher_matrix = self.load_data(device)
        # only calculate the fisher_matrix, weights, grads
        if offline_fisher_matrix:
            return 
        #import pdb;pdb.set_trace()
        # Ablation study for the baselines: mag,woodfisher + weight update
        if debug_update_weight:
            self.set_greedy_pruner(device=device, desired_level = self.args.target_sparsity)
            self.greedy_pruner.prune_init()
            
            masks = torch.ones(len(self._get_weights()))
            masks[self.greedy_pruner.set_pruned] = 0 
            new_weights = self.update_weights_online()
            masks, new_weights = masks.to(device), new_weights.to(device)
            past_weight_masks = self.update_masks(masks, new_weights)
            
            meta['updated_weight'] = True
            meta['best_iter'] = 'magnitude'
            return True, meta
        
        #############################################################
        # we can calculate combinatorial pruning online and offline 
        if not online_greedy:
            #############################################################
            #Step2.1 Load the masks and new weights.
            #masks, new_weights = self.load_ep_mask()
            #Option1: load mask and updated weight from the saved greedy_mask file
            #         this happens when self.args.update_weight_online==False
            masks, new_weights, is_updated_weight = self.load_greedy_mask(
                    require_updated_weight = not self.args.update_weight_online)
            meta['updated_weight'] = is_updated_weight
            
            # Option2: calculate weight_update online
            if not self.args.not_update_weights and self.args.update_weight_online:
                self.set_greedy_pruner(device=device, desired_level = self.args.target_sparsity)
                new_weights = self.update_weights_online()
                meta['updated_weight'] = True

            masks, new_weights = masks.to(device), new_weights.to(device)
            # Step2.2. Update the weights and masks
            past_weight_masks = self.update_masks(masks, new_weights)
        else:
            #############################################################
            # Setp 1. Prepare Fisher matrix and weights for combinatorial prunnig 
            #self.args.range = 20
            #self.args.max_no_match = 20
            #self.args.threshold = 1e-5
            #self.args.init_method = 'mag'
            #self.args.max_iter = 100
            #self.args.swap_one_per_iter = False 
            #self.args.max_N_fluctation=5
            if self.greedy_pruner is None:
                self.set_greedy_pruner(device=device, desired_level = self.args.target_sparsity)
            #inital_weight_masks = self._get_masks()
            sample_losses = []
            sample_losses_w_update = []
            hist_masks = []
            ############################################################
            # Step2.0: A boosted intitialization for comb-pruning 
            if self.args.init_method == 'mag_perb':
                best_loss = 100000.0
                best_set_pruned = None
                best_itr = 0
                mag_perb_path = os.path.join(self.args.fisher_inv_path, 
                        f'mag_perb{self.args.max_mag_perb}_{self.args.target_sparsity}sparsity.pkl')

                if os.path.exists(mag_perb_path):
                    best_set_pruned = _load(mag_perb_path)['mag_perb_mask']
                else:    
                    for itr in range(self.args.max_mag_perb):
                        set_pruned = self.greedy_pruner.perb_online(itr)
                        masks = torch.ones(len(weights)).to(device)
                        masks[set_pruned] = 0
                        # Step2.0.2 Update the masks
                        past_weight_masks = self.update_masks(masks, new_weights = None)
                        # Step2.0.3
                        #   Note: _compute_sample_loss() has applied revert_pruned()
                        #         So the weights and masks are rolled back to the initial values
                        if self.args.dset == 'imagenet':
                            sample_loss = self._compute_sample_loss(testset, device, num_workers, 
                                    subset_inds = np.arange(10000))  
                                    #subset_inds = subset_inds[:6000])  
                                    #subset_inds = subset_inds)  
                        else:
                            sample_loss = self._compute_sample_loss(dset, device, num_workers, 
                                    subset_inds = subset_inds[:min(5000, len(subset_inds))])  
                        if sample_loss < best_loss:
                            best_loss = sample_loss
                            best_set_pruned = set_pruned
                            best_itr = itr
                        print(f'###{itr}-th iter in mag_perb')
                print(f'######mag_perb is done, {best_itr}-th sample_loss = {best_loss} ') 
                self.greedy_pruner.init_with_setting(best_set_pruned)
                _dump(mag_perb_path, {'mag_perb_mask':best_set_pruned })

                #import pdb;pdb.set_trace()
            ############################################################
            # Step2: Apply online comb-pruning 
            #       NOTE: get magnitude-based pruning results when itr = 0
            for itr in range(self.args.max_iter):
                # Step2.0 Reset the masks to original mask 
                #if itr > 0:
                #    self._reset_masks(inital_weight_masks)
                # Step2.1 Calculate the new masks from prune_online 
                to_stop, set_pruned = self.greedy_pruner.prune_online(itr)
                masks = torch.ones(len(weights)).to(device)
                masks[set_pruned] = 0
                # Step2.2 Update the masks
                hist_masks.append(masks)
                past_weight_masks = self.update_masks(masks, new_weights = None)
                # Step2.3
                #   Note: _compute_sample_loss() has applied revert_pruned()
                #         So the weights and masks are rolled back to the initial values
                if self.args.dset == 'imagenet':
                    # TODO: dset is train dataset which has transformer involves random crop 
                    #           thus make the sample_loss different in every call. 
                    #      testset is used temporately to debug. Change it later.  
                    #sample_loss = self._compute_sample_loss(dset, device, num_workers, 
                    sample_loss = self._compute_sample_loss(testset, device, num_workers, 
                            subset_inds = np.arange(10000))  
                            #subset_inds = subset_inds[:6000])  
                            #subset_inds = subset_inds)  
                else:
                    sample_loss = self._compute_sample_loss(dset, device, num_workers, 
                            subset_inds = None)  
                sample_losses.append(sample_loss)
                if to_stop:
                    break
                
                if not self.args.not_update_weights and self.args.update_weight_online:
                    new_weights = self.update_weights_online()
                    meta['updated_weight'] = True
                    past_weight_masks = self.update_masks(masks, new_weights)
                    sample_loss_w_update = self._compute_sample_loss(dset, device, num_workers, 
                                #subset_inds = subset_inds)  
                                subset_inds = None)  
                    sample_losses_w_update.append(sample_loss)
                    print('sample loss before w update', sample_loss)
                    print('sample loss after w update', sample_loss_w_update)
              
                if itr < 2:
                    to_stop = False
                else:
                    to_stop = self._stat_sample_losses(sample_losses, 
                                t_reduce=sample_losses[0]/50, max_N_fluctation = self.args.max_N_fluctation)
                if to_stop:
                    break
                #import pdb;pdb.set_trace()
                print(f'########After {itr}-th iteration, loss_reduction = {sample_loss}')
            # Step3: Choose the set_pruned of the lowest sample_loss
            #       TODO: Find the iter of lowest loss or use the last iter?
            itr =torch.tensor(sample_losses).argmin()
            meta['updated_weight'] = False
            new_weights = None

            # Step3.0. Update the weights and masks
            if not self.args.not_update_weights and self.args.update_weight_online:
                new_weights = self.update_weights_online()
                meta['updated_weight'] = True
            past_weight_masks = self.update_masks(hist_masks[itr], new_weights)

            #import pdb;pdb.se_trace()     
            meta['best_iter'] = 'magnitude' if itr == 0 else f'iter_{itr}'
            meta['sample_losses'] = sample_losses
            print(f"best_iter: {meta['best_iter']}, sample_losses={sample_losses}")
        return True, meta
