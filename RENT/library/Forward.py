import torch
import os
import matplotlib.pyplot as plt
import csv

from utils import save_csv
from library.method_utils import INIT

# CVPR17 Forward. baseline+Sampling

class Forward_Sample(INIT):
    def __init__(self, args):
        super().__init__(args)

    def update_transition_matrix(self):
        self.transition = torch.zeros(self.args.n_class, self.args.n_class)
        data_dict = {}
        for i in range(self.args.n_class):
            data_dict[i] = []

        for idx, _, _, labels in self.trainloader:
            for ind,lbl in zip(idx,labels):
                data_dict[lbl.item()].append(ind)

        for i in range(self.args.n_class):
            object_proba = self.proba[data_dict[i]][:, i]
            proba, index = torch.topk(object_proba, int(len(object_proba) * 0.03))
            self.transition[i] = self.proba[data_dict[i]][index[-1]]

        self.transition = torch.transpose(self.transition, 1, 0)
        self.transition = self.transition.to(self.args.device)
        return

    def run(self):
        print('\n===> Training Start: Forward Sample')
        if self.args.dataset in ['MNIST','FMNIST','SVHN','CIFAR10','CIFAR100']:
            preepoch = 10
        elif self.args.dataset in ['FoodN','Clothing']:
            preepoch = 1
        else:
            preepoch = 5
        for epoch in range(preepoch):
            self.update_model()
        self.update_transition_matrix()
        # train model
        for epoch in range(self.args.total_epochs):
            epoch_loss, epoch_class_acc, epoch_label_acc, time_train = self.update_model_sir()
            epoch_test_acc, time_test = self.evaluate_model()
            print('=' * 50)
            print('Epoch', epoch, 'Time', time_train, time_test)
            self.report_result(epoch_loss, epoch_class_acc, epoch_label_acc, epoch_test_acc)

        save_csv(self.args.result_dir, [self.loss_train, self.train_class_acc, self.train_label_acc, self.test_acc])
        torch.save(self.args.network.state_dict(), os.path.join(self.args.model_dir, 'classifier.pk'))

        return

class TForward_Sample(Forward_Sample):
    def __init__(self, args):
        super().__init__(args)
        print('\n===> Training Start: Forward-True')

    def run(self):
        if self.args.dataset in ['MNIST','FMNIST','SVHN','CIFAR10','CIFAR100']:
            preepoch = 10
        elif self.args.dataset in ['FoodN', 'Clothing']:
            preepoch = 1
        else:
            preepoch = 5

        for i in range(preepoch):
            self.update_model()
        self.update_transition_matrix()

        # train model
        self.generate_transition_matrix()
        for epoch in range(self.args.total_epochs):
            epoch_loss, epoch_class_acc, epoch_label_acc, time_train = self.update_model_sir()
            epoch_test_acc, time_test = self.evaluate_model()
            print('=' * 50)
            print('Epoch', epoch, 'Time', time_train, time_test)
            self.report_result(epoch_loss, epoch_class_acc, epoch_label_acc, epoch_test_acc)

        save_csv(self.args.result_dir, [self.loss_train, self.train_class_acc, self.train_label_acc, self.test_acc])
        torch.save(self.args.network.state_dict(), os.path.join(self.args.model_dir, 'classifier.pk'))

        return
