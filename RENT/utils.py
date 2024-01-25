import csv
import numpy as np
import pandas as pd
import torch
import os

from network import ConvNet, ResNet

def define_network(args):
    if args.dataset == 'MNIST':
        net = ConvNet.LeNet(args)
    elif args.dataset == 'FMNIST':
        net = ResNet.resnet18(input_channel=args.n_channel, num_classes=args.n_class)
    elif args.dataset in ['SVHN', 'CIFAR10']:
        net = ResNet.resnet34(input_channel=args.n_channel, num_classes=args.n_class)
    elif args.dataset == 'CIFAR100':
        net = ResNet.resnet50(input_channel=args.n_channel, num_classes=args.n_class)
    elif args.dataset in ['Clothing', 'Webvision']:
        net = ResNet.resnet50(pretrained=True, input_channel=args.n_channel, num_classes=args.n_class)
    else:
        net = None

    return net

def save_csv(dir, list):
    with open(os.path.join(dir,'result.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerows(list)
    return

# result analyzer
def load_model(network, args):
    path = os.path.join(args.model_dir,'classifier.pk')
    network.load_state_dict(torch.load(path))
    network.eval()
    return network

def merge_acc(dataset, model, dim=-1):
    mean_dict, std_dict = {}, {}
    for noise in ['clean_0.0','sym_0.2','sym_0.5','asym_0.2','asym_0.4']:
        test_dir = os.path.join('result',dataset,noise,model)
        acclist = []
        for seed in range(5):
            if os.path.isfile(test_dir+'/epoch_200_seed_{}/result.csv'.format(seed)):
                data = pd.read_csv(test_dir +'/epoch_200_seed_{}/result.csv'.format(seed))
                acclist.append(np.array(data)[dim, -1]*100)
        acclist = np.array(acclist)
        mean_dict[noise], std_dict[noise] = np.mean(acclist), np.std(acclist)
    return mean_dict, std_dict


