import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np

import time
import os

from utils import save_csv
from library.method_utils import INIT

# NIPS20 https://github.com/xiaoboxia/Part-dependent-label-noise
def norm(T):
    row_abs = torch.abs(T)
    row_sum = torch.sum(row_abs, 1)
    T_norm = row_abs / row_sum
    return T_norm

class Matrix_optimize(nn.Module):
    def __init__(self, args, basis_num, num_classes):
        super(Matrix_optimize, self).__init__()
        self.args = args
        self.basis_matrix = self._make_layer(basis_num, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.normal_(m.weight, std=1e-1)

    def _make_layer(self, basis_num, num_classes):
        layers = []
        for i in range(0, basis_num):
            layers.append(nn.Linear(num_classes, 1, False))
        return nn.Sequential(*layers)

    def forward(self, W, num_classes):
        results = torch.zeros(num_classes, 1).to(self.args.device)
        for i in range(len(W)):
            coefficient_matrix = float(W[i]) * torch.eye(num_classes, num_classes).to(self.args.device)
            self.basis_matrix[i].weight.data = norm(self.basis_matrix[i].weight.data)  # s.t.
            anchor_vector = self.basis_matrix[i](coefficient_matrix)
            results += anchor_vector
        return results

class PDN_Sample(INIT):
    def __init__(self, args):
        super().__init__(args)

    def respresentations_extract(self, dim=512):
        self.args.network.eval()
        self.rep = torch.zeros(self.lentrain, dim)
        with torch.no_grad():
            for index, images, classes, labels in self.trainloader:
                feature, _ = self.args.network(images.to(self.args.device), True)
                self.rep[index] = feature.detach().cpu()

        self.rep = self.rep.numpy()
        return

    def probability_extract(self):
        self.args.network.eval()
        self.prob = torch.zeros(self.lentrain, self.args.n_class)
        with torch.no_grad():
            for index, images, _, _ in self.trainloader:
                self.prob[index] = F.softmax(self.args.network(images.to(self.args.device)), dim=1).detach().cpu()
        self.prob = self.prob.numpy()
        return

    def train_m(self):
        m, n = np.shape(self.rep)
        W = np.mat(np.random.random((m, 20)))
        H = np.mat(np.random.random((20, n)))

        for _ in range(10):
            if np.sum(np.square(self.rep-np.dot(W,H))) < 1e-5:  # threshold
                break

            a = np.dot(W.T, self.rep)  # Hkj
            b = np.dot(np.dot(W.T, W), H)
            for i_1 in range(20):
                for j_1 in range(n):
                    if b[i_1, j_1] != 0:
                        H[i_1, j_1] = H[i_1, j_1] * a[i_1, j_1] / b[i_1, j_1]

            c = np.dot(self.rep, H.T)
            d = np.dot(np.dot(W, H), H.T)
            for i_2 in range(m):
                for j_2 in range(20):
                    if d[i_2, j_2] != 0:
                        W[i_2, j_2] = W[i_2, j_2] * c[i_2, j_2] / d[i_2, j_2]

        W /= np.sum(W, 1)

        return W

    def estimate_matrix(self):
        transition_matrix_group = np.empty((20, self.args.n_class, self.args.n_class))
        idx_matrix_group = np.empty((self.args.n_class, 20))
        a = np.linspace(97, 99, 20)
        a = list(a)
        for index in range(len(a)):
            T = np.empty((self.args.n_class, self.args.n_class))  # +1 -> index
            ind = []
            for i in np.arange(self.args.n_class):
                robust_eta = self.prob[:, i]
                robust_eta[robust_eta >= np.percentile(robust_eta,a[index],interpolation='higher')] = 0.0
                idx_best = np.argmax(robust_eta)
                ind.append(idx_best)
                T[i] = self.prob[idx_best]

            transition_matrix_group[index] = T/(np.sum(T,1))
            idx_matrix_group[:, index] = np.array(ind)

        return idx_matrix_group, transition_matrix_group

    def init_params(self, net):
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')

            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=1e-1)

        return net.to(self.args.device)

    def basis_matrix_optimize(self):
        self.basis_matrix_group = np.empty((20, self.args.n_class, self.args.n_class))
        model = Matrix_optimize(self.args, 20, self.args.n_class)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        idx_matrix_group, transition_matrix_group = self.estimate_matrix()
        func = nn.MSELoss()

        for i in range(self.args.n_class):
            model = self.init_params(model)
            for epoch in range(1500):
                loss_total = 0.
                for j in range(20):
                    prediction = model(list(np.array(self.w_group[int(idx_matrix_group[i, j]),:]))[0], self.args.n_class)
                    optimizer.zero_grad()
                    loss = func(prediction, torch.from_numpy(transition_matrix_group[j, i, :][:, np.newaxis]).float().to(self.args.device))
                    loss.backward()
                    optimizer.step()
                    loss_total += loss.item()
                if loss_total < 0.02:
                    print(time.time() - self.time)
                    break

            for x in range(20):
                parameters = model.basis_matrix[x].weight.data.cpu().numpy()
                self.basis_matrix_group[x, i, :] = parameters

        for i in range(self.basis_matrix_group.shape[0]):
            for j in range(self.basis_matrix_group.shape[1]):
                for k in range(self.basis_matrix_group.shape[2]):
                    if self.basis_matrix_group[i, j, k] < 1e-6:
                        self.basis_matrix_group[i, j, k] = 0.
        return

    def update_transition_matrix(self):
        print('Transition Matrix is estimating...Wait...')
        if self.args.dataset == 'CIFAR10':
            dim = 512
        else:
            dim = 2048
        self.respresentations_extract(dim)
        self.probability_extract()
        self.w_group = self.train_m()
        self.basis_matrix_optimize()
        return

    def update_model_sample(self):
        self.args.network.train()
        epoch_loss, epoch_class_accuracy, epoch_label_accuracy = 0, 0, 0
        for index, images, classes, labels in self.trainloader:
            images = images.to(self.args.device)
            labels = labels.to(self.args.device)
            outputs = self.args.network(images)

            # loss
            f_prob = F.softmax(outputs.detach())
            true_prob = torch.gather(f_prob, 1, labels.reshape(-1, 1)).squeeze()  # P(y|x)

            tmplen = len(labels)
            noisy_prob = torch.zeros(tmplen, self.args.n_class).to(self.args.device)
            for i, idx, lbls, prb in zip(range(tmplen), index, labels, f_prob):
                w = np.expand_dims(np.transpose(self.w_group[idx]), axis=2)  # 20*1*1
                t = (w * self.basis_matrix_group).sum(0)  # c*c
                t = torch.from_numpy(t[:, lbls]).to(self.args.device)
                noisy_prob[i] = t*prb

            weighting = true_prob / (torch.gather(noisy_prob, 1, labels.reshape(-1, 1)).squeeze() + 1e-12)
            smpl_idx = torch.multinomial(weighting, num_samples=int(self.args.N * len(labels)), replacement=True)

            loss = self.criterion(outputs[smpl_idx],labels[smpl_idx])
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
        print('\n===> Training Start: PDN')
        # pretrain
        if self.args.dataset in ['MNIST','FMNIST','SVHN','CIFAR10','CIFAR100']:
            preepoch = 5
        else:
            preepoch = 1

        for i in range(preepoch):
            self.update_model()

        self.update_transition_matrix()

        # train model
        for epoch in range(self.args.total_epochs):
            epoch_loss, epoch_class_acc, epoch_label_acc, time_train = self.update_model_sample()
            epoch_test_acc, time_test = self.evaluate_model()
            print('=' * 50)
            print('Epoch', epoch, 'Time', time_train, time_test)
            self.report_result(epoch_loss, epoch_class_acc, epoch_label_acc, epoch_test_acc)

        save_csv(self.args.result_dir, [self.loss_train, self.train_class_acc, self.train_label_acc, self.test_acc])
        torch.save(self.args.network.state_dict(), os.path.join(self.args.model_dir, 'classifier.pk'))

        return
