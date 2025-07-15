import numpy as np
from torch import device, cuda, randperm


# define the hardware
def get_device(): return device('cuda' if cuda.is_available() else 'cpu')


def mix_up_data(x, y):
    lam = np.random.beta(0.2, 0.2)
    batch_size = x.size(0)
    index = randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mix_up_criterion(criterion, prediction, y_a, y_b, lam):
    return lam * criterion(prediction, y_a) + (1 - lam) * criterion(prediction, y_b)