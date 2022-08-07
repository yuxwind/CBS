import matplotlib.pyplot as plt
import numpy as np

def analyze_new_weights(new_w=None, old_w=None, update_w=None, score=None, info=''):
    x = np.arange(new_w.shape[0])
    #import pdb;pdb.set_trace()
    if old_w is not None:
        plt.plot(x, old_w.detach().cpu().numpy(), 'b', label='old_w')
    if new_w is not None:
        plt.plot(x, new_w.detach().cpu().numpy()+0.2, 'g', label='new_w + 0.2')
    if update_w is not None:
        plt.plot(x, update_w.detach().cpu().numpy()-0.2, 'y', label='update_w - 0.2')
    if score is not None:
        s = score.abs().mean()
        scaling_factor = int(1/s) if s > 0 else 1
        scaling_factor = 10 if scaling_factor > 10 else scaling_factor
        plt.plot(x, score.detach().cpu().numpy() * scaling_factor, 'r', label=f'score*{scaling_factor}')
    plt.legend()
    if info != '':
        info = '-'+ info
    plt.savefig(f'weigth_diff{info}.pdf')
    plt.clf()
