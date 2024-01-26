import torch
import torch.nn as nn
import time
import os
import torch.distributions.dirichlet as dir

from utils import save_csv
from library.method_utils import INIT

class reweight_loss(nn.Module):
    def __init__(self, args):
        super(reweight_loss, self).__init__()
        self.softmax = nn.Softmax(dim=1).to(args.device)
        self.crit = nn.CrossEntropyLoss(reduction='none').to(args.device)
        self.args = args

    def forward(self, outputs, T, labels):
        outputs_clone = outputs.clone().detach()
        f_prob = self.softmax(outputs_clone)
        true_prob = torch.gather(f_prob, 1, labels.reshape(-1, 1)).squeeze()+1e-12
        noisy_prob = torch.sum(T[labels] * f_prob, dim=1)+1e-12
        w = true_prob/noisy_prob

        m = dir.Dirichlet(self.args.alpha*w)
        return torch.mean(torch.matmul(m.sample((1,self.args.batch_size))[0],self.crit(outputs,labels)))

class Reweight(INIT):
    def __init__(self, args):
        super().__init__(args)

    def run(self):
        return

class TReweight(Reweight):
    def __init__(self, args):
        super().__init__(args)
        print('\n===> Training Start: Importance Dirichlet')

    def run(self):
        # train model
        if self.args.dataset in ['MNIST','FMNIST','SVHN','CIFAR10','CIFAR100']:
            preepoch = 10
        else:
            preepoch = 2
        for i in range(preepoch):
            self.update_model()

        self.generate_transition_matrix()
        for epoch in range(self.args.total_epochs):
            epoch_loss, epoch_class_acc, epoch_label_acc, time_train = self.update_model_reweight()
            epoch_test_acc, time_test = self.evaluate_model()
            print('=' * 50)
            print('Epoch', epoch, 'Time', time_train, time_test)
            self.report_result(epoch_loss, epoch_class_acc, epoch_label_acc, epoch_test_acc)

        save_csv(self.args.result_dir, [self.loss_train, self.train_class_acc, self.train_label_acc, self.test_acc])
        torch.save(self.args.network.state_dict(), os.path.join(self.args.model_dir, 'classifier.pk'))

        return
