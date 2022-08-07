import os
import sys

import pathlib


ROOT = os.path.dirname(pathlib.Path(__file__).resolve())
weight_root = os.path.join(ROOT, 'prob_regressor_data')

# greedy algorithm: 
#script: python greedy_alg/schedule_job.py 0 
greedy_cls = 'greedyblock' # ['greedy', 'greedyblock']
FISHER_PATH='resnet20_cifar10_1000samples_1000batches_0seed.pkl'

# eval the greedy results
#   eval_greedy_cifar10.py: greedy_root, eval_script
# script: python eval_greedy_cifar10.py --gpu 2
greedy_root = os.path.join(ROOT, 'prob_regressor_results', greedy_cls, 
                'resnet20_cifar10_1000samples_1000batches_0seed-swap_one_False-init-mag')
eval_script = os.path.join(ROOT, 'scripts', 
                'sweep_cifar10_resnet20_oneshot.sh')

# collect evaluation: 
#   greedy_io.py: greedy_cls, eval_root
# script: python greedy_alg/greedy_io.py
eval_root = os.path.join(ROOT, 'prob_regressor_results', greedy_cls,
                'resnet20_cifar10_1000samples_1000batches_0seed-swap_one_False-init-mag.eval') 

# analyze: 
#   cmpr_pruned.py: greedy_root, weight_root
# script: python greedy_alg/cmpr_pruned.py

