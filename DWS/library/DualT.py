import torch
import time
import os

from utils import save_csv
from library.method_utils import INIT

# Dual T. baseline from https://github.com/a5507203/dual-t-reducing-estimation-error-for-transition-matrix-in-label-noise-learning
class DualT_Weight(INIT):
    def __init__(self, args):
        super().__init__(args)
        print('\n===> Training Start: DualT')

    def update_transition_to_yhat(self):
        self.T_club = torch.zeros(self.args.n_class, self.args.n_class)
        data_dict = {}
        for i in range(self.args.n_class):
            data_dict[i] = []

        for idx, _, _, labels in self.trainloader:
            for ind,lbl in zip(idx,labels):
                data_dict[lbl.item()].append(ind)

        for i in range(self.args.n_class):
            object_proba = self.proba[data_dict[i]][:, i]
            proba, index = torch.topk(object_proba, int(len(object_proba) * 0.03))
            self.T_club[i] = self.proba[data_dict[i]][index[-1]]

        self.T_club = torch.transpose(self.T_club, 1, 0)
        self.T_club = self.T_club.to(self.args.device)
        return

    def update_transition_to_ytilde(self):
        self.args.network.eval()
        self.T_spade = torch.zeros(self.args.n_class, self.args.n_class)
        for index, images, _, labels in self.trainloader:
            images = images.to(self.args.device)
            outputs = self.args.network(images)
            _, max_label = torch.max(outputs,dim=1)
            for lbl, pml in zip(labels, max_label):
                self.T_spade[lbl][pml]+=1.0

        self.T_spade/=torch.sum(self.T_spade,dim=0)
        self.T_spade = torch.nan_to_num(self.T_spade).to(self.args.device)
        return

    def run(self):
        if self.args.dataset in ['MNIST', 'FMNIST', 'SVHN', 'CIFAR10', 'CIFAR100']:
            preepoch = 20
        else:
            preepoch = 1
        for epoch in range(preepoch):
            self.update_model()
        # update transition matrix
        self.update_transition_to_yhat()
        self.update_transition_to_ytilde()

        self.transition = torch.matmul(self.T_spade, self.T_club)

        for epoch in range(self.args.total_epochs):
            epoch_loss, epoch_class_acc, epoch_label_acc, time_train = self.update_model_reweight()
            epoch_test_acc, time_test = self.evaluate_model()
            print('=' * 50)
            print('Epoch', epoch, 'Time', time_train, time_test)
            self.report_result(epoch_loss, epoch_class_acc, epoch_label_acc, epoch_test_acc)

        save_csv(self.args.result_dir, [self.loss_train, self.train_class_acc, self.train_label_acc, self.test_acc])
        torch.save(self.args.network.state_dict(), os.path.join(self.args.model_dir, 'classifier.pk'))

        return
