import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    elif args.dataset in ['Food', 'FoodN', 'Clothing', 'Webvision']:
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

def merge_acc(dataset, model, epoch=200, dim=-1):
    mean_dict, std_dict = {}, {}
    alpha_list = [0.1,0.2,0.5,1.0,10.0,100.0,1000.0]
    for noise in ['sym_0.2','sym_0.5','asym_0.2','asym_0.4']:
        latex_tab = model+noise
        test_dir = os.path.join('result',dataset,noise,model)
        mean_list, std_list = [], []
        for alpha in alpha_list:
            acclist = []
            for seed in range(5):
                dir = '/epoch_{}_alpha_{}_seed_{}/result.csv'.format(epoch,alpha,seed)
                # print(test_dir+dir)
                if os.path.isfile(test_dir+dir):
                    data = pd.read_csv(test_dir +dir)
                    acclist.append(np.array(data)[dim, -1]*100)
            print(model, noise,'|', len(acclist))
            acclist = np.array(acclist)
            meanval, stdval = np.mean(acclist), np.std(acclist)
            mean_dict[noise], std_dict[noise] = meanval, stdval
            mean_list.append(meanval)
            std_list.append(stdval)
            latex_tab += '&{:.2f}'.format(meanval)+'\scriptsize'+'{$\pm$'+'{:.1f}'.format(stdval)+'}'

        print(latex_tab)
        plt.errorbar(alpha_list, mean_list,yerr=std_list)
        plt.savefig(os.path.join('result',dataset,'figure',noise+'_'+model+'.jpg'))
        plt.close()
    return mean_dict, std_dict


