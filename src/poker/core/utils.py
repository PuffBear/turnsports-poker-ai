import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def normalize(val, min_val, max_val):
    return (val - min_val) / (max_val - min_val + 1e-8)
