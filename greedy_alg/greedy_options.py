import os
import sys
sys.path.append('../')

import argparse

from env import ROOT, WEIGHT_ROOT, RESULT_ROOT

def get_parse():
    parser = argparse.ArgumentParser(description='Greedy combinatorial pruning')
    parser.add_argument('--greedy-dir', type=str, default=None, help='the dir of the combinatorial pruning result')
    parser.add_argument('--idx-2-module-path', type=str, default=None, help='the path of idx_2_module file')
    parser.add_argument('--wgh_path', default=None, type=str,
                        help='path where to load the weights and gradients and emp. fisher')
    parser.add_argument('--fisher_inv_path', type=str, default=None, 
                        help='the path where to load fisher_inv')    
    parser.add_argument('--greedy_method', default='greedyblock', choices = ['greedy', 'greedyblock'],
                        help='greedy or greedyblock pruning')
    parser.add_argument('--init_method', default='mag', 
                        choices = ['mag', 'wg', 'woodfisher', 'woodfisherblock'],
                        help='the method to init the pruning and remaining sets')
    parser.add_argument('--sparsity', default=0.9, type=float,
                        help='target sparsity after model compression')
    parser.add_argument('--max-iter', default=100, type=int,
                        help='the maximum iterations for greedy algorithm')
    parser.add_argument('--range', default=100, type=int,
                        help='the maximum number of weights in remaining weights to search for swaping')
    parser.add_argument('--max-no-match', default=20, type=int,
                        help='the maximum number of weights in the removed weights to miss swaping')
    parser.add_argument('--threshold', default=1e-4, type=float,
                        help='the threshold of reduction in objective loss to swap')
    #parser.add_argument('--only-calculate-delta-w', default=False, type=bool,
    #                    help='this is not related to the algorithm and is used to run experiments')

    parser.add_argument('--only-delta-w',
                        dest='only_calculate_delta_w', action='store_true',
                        help='this is not related to the algorithm and is used to run experiments')
    parser.add_argument('--not-only-delta-w',
                        dest='only_calculate_delta_w', action='store_false',
                        help='this is not related to the algorithm and is used to run experiments')
    parser.set_defaults(only_calculate_delta_w=False)

    parser.add_argument('--swap-one-per-iter', 
                        dest='swap_one_per_iter', action='store_true',
                        help='this is not related to the algorithm and is used to run experiments')
    parser.add_argument('--not-swap-one-per-iter', 
                        dest='swap_one_per_iter', action='store_false',
                        help='this is not related to the algorithm and is used to run experiments')
    parser.set_defaults(swap_one_per_iter=False)


    parser.add_argument('--woodfisher_mask_path', type=str, default=None, 
                        help='the path where to load the mask obtained from woodfisher/woodfisherblock')
    parser.add_argument('--greedy-path', type=str, default=None, help='where to load combinatorial pruning result')
    return parser.parse_args()

def debug_args(args):
    #wgh_name = 'resnet20_cifar10_1000samples_1000batches_0seed_allweights.pkl' 
    wgh_name = 'resnet20_cifar10_1000samples_1000batches_0seed.pkl' 
    arch     = os.path.splitext(wgh_name)[0]
    args.wgh_path    = os.path.join(WEIGHT_ROOT, wgh_name) 
    s_swap   = 'swap_one_pair' if args.swap_one_per_iter else 'swap_multi_pair'
    s_method = f'DEBUG-{args.greedy_method}-init_{args.init_method}-{s_swap}'
    args.greedy_dir  = os.path.join(RESULT_ROOT, s_method, arch)
    args.fisher_inv_path = os.path.join(WEIGHT_ROOT, arch + '.fisher_inv')
    args.idx_2_module_path = os.path.join(WEIGHT_ROOT, arch + '-idx_2_module.npy')
    args.sparsity    = 0.8
    args.range       = 100
    args.max_no_match= 1
    args.threshold   = 1e-3
    args.init_method = 'woodfisherblock'
    args.swap_one_per_iter = False
    args.only_calculate_delta_w = False
    args.max_iter    = 100
    args.greedy_path = os.path.join(RESULT_ROOT, "cifar10-resnet20-backbone_layers-blockwise_fisher-greedy_online_mag_all_layers/results.train_loss_all_samples.20211202.22_52/resnet20_cifar10_1000samples_1000batches_0seed/sparsity0.80_T1.00e-05_Range10_NOMATCH10-magnitude.npy")
    args.woodfisher_mask_path = os.path.join(WEIGHT_ROOT, "resnet20_cifar10_1000samples_1000batches_0seed.fisher_inv/global_mask_sparsity0.80.pkl")
