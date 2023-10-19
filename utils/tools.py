import torch
import numpy as np
import random
from datetime import datetime


def get_cur_time():
    return datetime.now().strftime("%Y_%m_%d")


def get_cur_time_sec():
    return datetime.now().strftime("%Y_%m_%d_%H_%M_%S")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
