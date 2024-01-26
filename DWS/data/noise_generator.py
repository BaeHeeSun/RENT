import os
import numpy as np
import random
from tqdm import tqdm
import copy
import torch
from math import inf
from scipy import stats

# for noise generation
# Generate symmetric noisy label
def generate_noisy_label_symmetric(args, label):
    label = torch.tensor(label)
    noisy_label = copy.deepcopy(label)
    n_ratio = float(args.noisy_ratio)*((args.n_class)/(args.n_class-1))
    n_noisy = int(len(label)*n_ratio)
    chg_idx = np.random.permutation(np.arange(len(label)))[:n_noisy]
    noisy_label[chg_idx] = torch.randint(args.n_class,(n_noisy,))

    return noisy_label

# Generate asymmetric noisy label
def generate_noisy_label_asymmetric(args, label):
    nlabel_torch = torch.tensor(args.noisy_label_list)

    label = torch.tensor(label)
    noisy_label = copy.deepcopy(label)

    n_noisy = int(len(label)*float(args.noisy_ratio))
    chg_idx = np.random.permutation(np.arange(len(label)))[:n_noisy]
    noisy_label[chg_idx] = nlabel_torch[label[chg_idx]]

    return noisy_label

# Generate asymmetric noisy label for cifar100
def generate_noisy_label_asymmetric_100(args, label):
    c_label = torch.tensor([4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                              3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                              6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                              0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                              5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                              16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                              10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                              2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                              16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                              18, 1, 2, 15, 6, 0, 17, 8, 14, 13])

    coarse_label_data = torch.zeros(20,5).long()
    for i in range(20):
        coarse_label_data[i] = torch.where(c_label == i)[0]

    label = torch.tensor(label)
    noisy_label = copy.deepcopy(label)
    n_ratio = float(args.noisy_ratio) * (5/4)
    n_noisy = int(len(label) * n_ratio)

    chg_idx = np.random.permutation(np.arange(len(label)))[:n_noisy]
    noisy_label[chg_idx] = coarse_label_data[c_label[label[chg_idx]]].gather(1, torch.randint(5, (n_noisy, 1))).squeeze()

    return noisy_label

# Generate instancewise noisy label
def generate_noisy_label_idn(args, input, label):
    input, label = torch.tensor(input), torch.tensor(label)
    n_ratio = float(args.noisy_ratio)
    flip_distribution = stats.truncnorm((0 - n_ratio) / 0.1, (1 - n_ratio) / 0.1, loc=n_ratio, scale=0.1)
    flip_rate = torch.tensor(flip_distribution.rvs(len(label)))
    W = torch.randn(args.n_class, input[0].flatten().shape[0], args.n_class)
    p = torch.sum(input.contiguous().view(len(label),-1,1)*W[label], dim=1)
    p.scatter_(1,label.unsqueeze(1),-inf)
    p = flip_rate.unsqueeze(1) * torch.softmax(p, dim=1)
    p.scatter_(1, label.unsqueeze(1), (1-flip_rate).unsqueeze(1))

    noisy_label = torch.multinomial(p,1)

    return noisy_label.squeeze(1)

# Generate openset noisy label
def generate_noisy_label_open(args, input, label):
    return







