import torch
from torch.utils.data import Subset, DataLoader
import time
import os
import numpy as np
import csv
import matplotlib.pyplot as plt
from collections import OrderedDict

import data
from network import ConvNet, ResNet_BLTM
from utils import save_csv
from library.method_utils import INIT

# ICML22 BLTM. baseline reproduced from https://github.com/ShuoYang-1998/BLTM

class BLTM_Sample(INIT):
    def __init__(self, args):
        super().__init__(args)

    def distill_dataset(self, threshold=1.0/2):
        if self.args.dataset == 'Food':
            distilled_data = data.Food101(root=self.args.data_dir, transform=self.args.transform, split='train', download=True)
        else:
            distilled_data = data.__dict__[self.args.dataset](args=self.args, root=self.args.data_dir, train=True, download=True)
        if type(distilled_data.targets) == list:
            distilled_data.targets = torch.tensor(distilled_data.targets)
        distilled_index_list = []

        print('Distilling')
        self.args.network.eval()
        for index, images, _, _ in self.trainloader:
            images = images.to(self.args.device)
            logits1 = self.softmax(self.args.network(images))
            logits1_max = torch.max(logits1, dim=1)
            mask = logits1_max[0] > threshold
            distilled_index_list.extend(index[mask])
            distilled_data.targets[index] = logits1_max[1].detach().cpu()
        distilled_dataset = Subset(distilled_data, np.array(distilled_index_list))
        print(len(distilled_index_list))
        self.distilled_loader = DataLoader(dataset=distilled_dataset, batch_size=self.args.batch_size, shuffle=True, pin_memory=True, num_workers=8, drop_last=True)
        return

    def define_transition_matrix(self):
        if self.args.dataset == 'MNIST':
            self.transition_net = ConvNet.LeNet_BLTM(self.args)
        elif self.args.dataset == 'FMNIST':
            self.transition_net = ResNet_BLTM.resnet18(input_channel=self.args.n_channel, num_classes=self.args.n_class)
        elif self.args.dataset in ['SVHN', 'CIFAR10']:
            self.transition_net = ResNet_BLTM.resnet34(input_channel=self.args.n_channel, num_classes=self.args.n_class)
        elif self.args.dataset == 'CIFAR100':
            self.transition_net = ResNet_BLTM.resnet50(input_channel=self.args.n_channel, num_classes=self.args.n_class)
        elif self.args.dataset in ['Food', 'FoodN', 'Clothing', 'Webvision']:
            self.transition_net = ResNet_BLTM.resnet50(pretrained=True, input_channel=self.args.n_channel, num_classes=self.args.n_class)
        else:
            self.transition_net = None

        temp = OrderedDict()
        tnet_state_dict = self.transition_net.state_dict()
        for name, parameter in self.args.network.named_parameters():
            if name in tnet_state_dict:
                temp[name] = parameter
        tnet_state_dict.update(temp)
        self.transition_net.load_state_dict(tnet_state_dict)
        self.transition_net.to(self.args.device)
        return

    def loss_bltm_forward(self, transition_batch, classes, labels):
        loss = torch.tensor(0.0).to(self.args.device)
        for j in range(len(labels)):
            loss -= torch.log(transition_batch[j][labels[j]][classes[j]]+1e-12)
        return loss.to(self.args.device)

    def train_transition_matrix(self):
        transition_optimizer = torch.optim.SGD(self.transition_net.parameters(), lr=self.args.lr, momentum=0.9)
        for epoch in range(20):
            epoch_loss = 0.0
            self.transition_net.train()
            for index, images, classes, labels in self.distilled_loader:
                images,classes,labels=images.to(self.args.device),classes.to(self.args.device),labels.to(self.args.device)
                batch_t_matrix = self.transition_net(images)
                loss = self.loss_bltm_forward(batch_t_matrix, classes, labels)
                transition_optimizer.zero_grad()
                loss.backward()
                transition_optimizer.step()
                epoch_loss += loss.item()
            print('Train_loss_epoch {}'.format(epoch), epoch_loss)
        torch.save(self.transition_net.state_dict(), os.path.join(self.args.model_dir, 'transition_matrix_network.pk'))
        return

    def sample_batch_bltm(self, transition_batch, outputs, labels):
        f_prob = self.softmax(outputs.detach())
        true_prob = torch.gather(f_prob, 1, labels.reshape(-1, 1)).squeeze()  # P(y|x)
        noisy_prob = torch.bmm(transition_batch,self.softmax(outputs).unsqueeze(2)).squeeze()  # P(y\tilde|x)=TP(y|x)
        weighting = true_prob / (torch.gather(noisy_prob, 1, labels.reshape(-1, 1)).squeeze() + 1e-12)
        smpl_idx = torch.multinomial(weighting,num_samples=int(self.args.N*len(labels)), replacement=True)

        return outputs[smpl_idx], labels[smpl_idx]

    def update_model_sir_bltm(self):
        self.args.network.train()
        epoch_loss, epoch_class_accuracy, epoch_label_accuracy = 0, 0, 0
        for index, images, classes, labels in self.trainloader:
            images = images.to(self.args.device)
            labels = labels.to(self.args.device)
            outputs = self.args.network(images)
            batch_t_matrix = self.transition_net(images)
            # loss
            s_preds,s_lbls= self.sample_batch_bltm(batch_t_matrix, outputs, labels)
            loss = self.criterion(s_preds,s_lbls)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item() * len(labels)
            # accuracy
            _, model_label = torch.max(outputs, dim=1)
            epoch_class_accuracy += (classes == model_label.cpu()).sum().item()
            epoch_label_accuracy += (labels == model_label).cpu().sum().item()

        time_elapse = time.time() - self.time
        return epoch_loss, epoch_class_accuracy, epoch_label_accuracy, time_elapse

    def run(self):
        print('\n===> Training Start: BLTM')
        # warmup
        if self.args.dataset in ['MNIST', 'FMNIST', 'SVHN', 'CIFAR10', 'CIFAR100']:
            preepoch = 20
        else:
            preepoch = 1
        for epoch in range(preepoch):
            self.update_model()
        self.distill_dataset()
        self.define_transition_matrix()
        self.train_transition_matrix()

        # train model
        for epoch in range(self.args.total_epochs):
            epoch_loss, epoch_class_acc, epoch_label_acc, time_train = self.update_model_sir_bltm()
            epoch_test_acc, time_test = self.evaluate_model()
            print('=' * 50)
            print('Epoch', epoch, 'Time', time_train, time_test)
            self.report_result(epoch_loss, epoch_class_acc, epoch_label_acc, epoch_test_acc)

        save_csv(self.args.result_dir, [self.loss_train, self.train_class_acc, self.train_label_acc, self.test_acc])
        torch.save(self.args.network.state_dict(), os.path.join(self.args.model_dir, 'classifier.pk'))

        return
