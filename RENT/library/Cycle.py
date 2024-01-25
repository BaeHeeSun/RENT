import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import csv

from utils import save_csv
from library.method_utils import INIT

# NeurIPS 22 Cycle. No official code
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

class Cycle_Sample(INIT):
    def __init__(self, args):
        super().__init__(args)
        self.trans = sig_t(args).to(args.device) # T_Transpose
        self.trans_inv = sig_t(args).to(args.device) # T^inv_Transpose

        self.optimizer_trans = torch.optim.SGD(self.trans.parameters(), lr=self.args.lr, momentum=0.9)
        self.optimizer_t_inv = torch.optim.SGD(self.trans_inv.parameters(), lr=self.args.lr, momentum=0.9)
        print('\n===> Training Start: Cycle Consistency Regularization')

    def loss_learnable_forward(self, outputs, labels, trans):
        new_proba = torch.mm(self.softmax(outputs),trans)
        return self.nll(torch.log(new_proba+1e-12), labels)

    def loss_learnable_backward(self, outputs, labels, trans_inv):
        noisy_onehot = torch.nn.functional.one_hot(labels,num_classes=self.args.n_class).float()
        noisy_proba = torch.mm(self.softmax(noisy_onehot), trans_inv)
        return -torch.mean(self.softmax(outputs)*torch.log(noisy_proba+1e-12))*self.args.n_class

    def loss_reg(self, outputs, trans, trans_inv):
        final_proba = torch.mm(torch.mm(self.softmax(outputs), trans), trans_inv)
        return -torch.mean(self.softmax(outputs)*torch.log(final_proba+1e-12))*self.args.n_class

    def loss_func(self,outputs,labels):
        transition = self.transition.clone().detach()
        transition_inv = self.transition_inv.clone().detach()

        loss1 = self.loss_learnable_forward(outputs,labels,transition)
        loss2 = self.loss_learnable_backward(outputs,labels,transition_inv)
        loss3 = self.loss_reg(outputs,transition,transition_inv)

        return loss1+loss2+0.3*loss3

    def update_model_cycle_sir(self):
        self.args.network.train()
        epoch_loss, epoch_class_accuracy, epoch_label_accuracy = 0, 0, 0

        for index, images, classes, labels in self.trainloader:
            images = images.to(self.args.device)
            labels = labels.to(self.args.device)
            outputs = self.args.network(images)
            outputs_for_transition = outputs.clone().detach()

            self.transition = self.trans()
            self.transition_inv = self.trans_inv()

            # loss : total loss update, theta update
            s_preds, s_lbls = self.sample_batch(outputs, labels)
            loss = self.loss_func(outputs,labels)+self.criterion(s_preds, s_lbls)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # loss1 : T update
            loss1 = self.loss_learnable_forward(outputs_for_transition,labels,self.transition)
            self.optimizer_trans.zero_grad()
            loss1.backward()
            self.optimizer_trans.step()

            #loss2 : T inv update
            loss2 = self.loss_learnable_backward(outputs_for_transition,labels,self.transition_inv)
            self.optimizer_t_inv.zero_grad()
            loss2.backward()
            self.optimizer_t_inv.step()

            epoch_loss += loss.item() * len(labels)
            # accuracy
            _, model_label = torch.max(outputs, dim=1)
            epoch_class_accuracy += (classes == model_label.cpu()).sum().item()
            epoch_label_accuracy += (labels == model_label).cpu().sum().item()

        time_elapse = time.time() - self.time
        return epoch_loss, epoch_class_accuracy, epoch_label_accuracy, time_elapse

    def run(self):
        # warmup
        if self.args.dataset == 'Clothing':
            preepoch = 1
        elif self.args.dataset == 'WebVision':
            preepoch = 5
        else:
            preepoch = 10
        for epoch in range(preepoch):
            self.update_model()
        # train model
        for epoch in range(self.args.total_epochs):
            epoch_loss, epoch_class_acc, epoch_label_acc, time_train = self.update_model_cycle_sir()
            epoch_test_acc, time_test = self.evaluate_model()
            print('=' * 50)
            print('Epoch', epoch, 'Time', time_train, time_test)
            self.report_result(epoch_loss, epoch_class_acc, epoch_label_acc, epoch_test_acc)

        save_csv(self.args.result_dir, [self.loss_train, self.train_class_acc, self.train_label_acc, self.test_acc])
        torch.save(torch.transpose(self.transition, 0, 1), os.path.join(self.args.result_dir, 'transition_matrix.pk'))
        torch.save(self.args.network.state_dict(), os.path.join(self.args.model_dir, 'classifier.pk'))

        return

