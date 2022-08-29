import os
import sys

import pandas as pd 
import torch


def cifar10_log(compr_ratio, gt, output):
    compr_ratio=0.1
    result_dir = 'fairness'
    csv_path = os.path.join(result_dir, "CIFAR10-MG-no_finetune.csv")

    # Create general csv file
    if not os.path.exists(csv_path):
        df = pd.DataFrame(columns=["Train/Val", "Image Index", "Image Label"])
        df["Image Index"] = [i for i in range(10000)]
    else:
        df = pd.read_csv(csv_path)

    if 'GT' not in df:
        df['GT'] = gt

    # Add prediction
    with torch.no_grad():
        _, pred = output.topk(1, 1, True, True)
        df[f'CompressionRation {compr_ratio:.2f}'] = pred.numpy()
    df.to_csv(csv_path, index=False)
