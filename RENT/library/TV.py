import torch
import os
import time
import numpy as np
import csv
import matplotlib.pyplot as plt

from utils import save_csv
from library.method_utils import INIT

# total variation regularization[ICML 21] from  https://github.com/YivanZhang/lio/tree/master/ex/transition-matrix
class sig_t(torch.nn.Module):
    def __init__(self, args, init=2):
        super(sig_t, self).__init__()
        self.register_parameter(name='w', param=torch.nn.parameter.Parameter(torch.ones(args.n_class, args.n_class)))
        self.w.to(args.device)

        self.init_val = torch.log(torch.tensor(0.5/(args.n_class-1)))
        self.co = self.init_val*torch.ones(args.n_class, args.n_class).to(args.device)
        ind = np.diag_indices(args.n_class)
        self.co[ind[0], ind[1]] = torch.zeros(args.n_class).to(args.device)
        self.identity = torch.eye(args.n_class).to(args.device)

    def forward(self):
        T = -torch.log(torch.tensor(2))*self.identity.detach() + self.w*self.co.detach()
        return torch.softmax(T, dim=1)

class TV_Sample(INIT):
    def __init__(self, args):
        super().__init__(args)
        self.trans = sig_t(args).to(args.device)
        if self.args.optimizer == 'SGD':
            self.optim_trans = torch.optim.SGD(self.trans.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=5e-4)
        else:
            self.optim_trans = torch.optim.Adam(self.trans.parameters(), lr=self.args.lr)

        print('\n===> Training Start: Total Variation')

    def tv_reg(self, outputs):
        outputs = self.softmax(outputs)
        idx1, idx2 = torch.randint(0, outputs.shape[0], (2, outputs.shape[0])).to(self.args.device)
        return -0.5 * (outputs[idx1] - outputs[idx2]).abs().sum(dim=1).mean()

    def loss_learnable_trans(self, outputs, labels):
        outputs = outputs.clone().detach()
        new_proba = torch.mm(self.softmax(outputs),self.transition)
        return self.nll(torch.log(new_proba+1e-12), labels)+0.1*self.tv_reg(outputs)

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
            self.optim_trans.zero_grad()
            losst.backward()
            self.optim_trans.step()
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
        torch.save(self.transition, os.path.join(self.args.result_dir, 'transition_matrix.pk'))
        torch.save(self.args.network.state_dict(), os.path.join(self.args.model_dir, 'classifier.pk'))

        return