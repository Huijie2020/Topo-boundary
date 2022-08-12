import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
import os
import numpy as np
from sklearn.metrics import average_precision_score
from losses.dice_loss import dice_coeff
from models.skeleton import soft_skel
from models.gabore_filter_bank import GaborFilters
import json

import torch.nn as nn

def cal_ctr_net(args, hog, loader, device):
    epochs = args.epochs
    n_val = len(loader)
    with tqdm(total=n_val, leave=False) as pbar:
        for batch in loader:
