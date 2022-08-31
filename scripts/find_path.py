import os
import sys
from glob import glob

seed = sys.argv[1]
paths = [f for f in glob(f'prob_regressor_results/cifar10-resnet20-backbone_layers*/*/*{seed}seed')]

for f in paths:
    print(f)
