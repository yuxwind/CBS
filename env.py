import os
import sys

import pathlib

ROOT = os.path.dirname(pathlib.Path(__file__).resolve())
WEIGHT_ROOT = os.path.join(ROOT, 'prob_regressor_data')
RESULT_ROOT = os.path.join(ROOT, 'prob_regressor_results')
PLOT_ROOT   = os.path.join(ROOT, 'analyse', 'plots')
