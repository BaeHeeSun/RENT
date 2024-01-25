import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import numpy as np
import csv
import time
import os

from utils import save_csv
from library.method_utils import INIT

# ICML 21 VolMinNet. baseline from https://github.com/xuefeng-li1/Provably-end-to-end-label-noise-learning-without-anchor-points
class sig_t(nn.Module):
    def __init__(self, args, init=2):
        super(sig_t, self).__init__()
        self.register_parameter(name='w', param=nn.parameter.Parameter(-init*torch.ones(args.n_class, args.n_class)))
        self.w.to(args.device)

        self.co = torch.ones(args.n_class, args.n_class).to(args.device)
        ind = np.diag_indices(args.n_class)
        self.co[ind[0], ind[1]] = torch.zeros(args.n_class).to(args.device)
        self.identity = torch.eye(args.n_class).to(args.device)

    def forward(self):
        sig = torch.sigmoid(self.w)
        T = self.identity.detach() + sig*self.co.detach()
        T = F.normalize(T, p=1, dim=1)

        return T

class VolMinNet_Sample(INIT):
    def __init__(self, args):
        super().__init__(args)
        self.trans = sig_t(args).to(args.device)

        if self.args.optimizer == 'SGD':
            self.optimizer_trans = torch.optim.SGD(self.trans.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=5e-4)
        else:
            self.optimizer_trans = torch.optim.Adam(self.trans.parameters(), lr=self.args.lr)

        print('\n===> Training Start: VolMinNet')

    def loss_learnable_trans(self, outputs, labels):
        outputs = outputs.clone().detach()
        new_proba = torch.mm(self.softmax(outputs),self.transition)
        forward_loss = self.nll(torch.log(new_proba+1e-12), labels)
        return forward_loss+0.0001*self.transition.slogdet().logabsdet

    def update_model_sir(self):
        self.args.network.train()
        epoch_loss, epoch_class_accuracy, epoch_label_accuracy = 0, 0, 0
        for index, images, classes, labels in self.trainloader:
            images = images.to(self.args.device)
            labels = labels.to(self.args.device)
            outputs = self.args.network(images)
            self.transition = self.trans()
            # loss 1. sampling loss
            s_preds, s_lbls = self.sample_batch(outputs, labels)
            loss = self.criterion(s_preds, s_lbls)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item() * len(labels)
            # loss 2. update transition matrix
            losst = self.loss_learnable_trans(outputs, labels)
            self.optimizer_trans.zero_grad()
            losst.backward()
            self.optimizer_trans.step()
            # accuracy
            _, model_label = torch.max(outputs, dim=1)
            epoch_class_accuracy += (classes == model_label.cpu()).sum().item()
            epoch_label_accuracy += (labels == model_label).cpu().sum().item()

        time_elapse = time.time() - self.time
        return epoch_loss, epoch_class_accuracy, epoch_label_accuracy, time_elapse

    def run(self):
        # train model
        for epoch in range(self.args.total_epochs):
            epoch_loss, epoch_class_acc, epoch_label_acc, time_train= self.update_model_sir()
            epoch_test_acc, time_test = self.evaluate_model()
            print('=' * 50)
            print('Epoch', epoch, 'Time', time_train, time_test)
            self.report_result(epoch_loss, epoch_class_acc, epoch_label_acc, epoch_test_acc)

        save_csv(self.args.result_dir, [self.loss_train, self.train_class_acc, self.train_label_acc, self.test_acc])
        torch.save(torch.transpose(self.transition, 0, 1), os.path.join(self.args.result_dir, 'transition_matrix.pk'))
        torch.save(self.args.network.state_dict(), os.path.join(self.args.model_dir, 'classifier.pk'))

        return

