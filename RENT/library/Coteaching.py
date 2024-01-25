import torch
import torch.optim as optim
import os
import time

from utils import save_csv
from library.method_utils import INIT

# coteaching

class Coteaching(INIT):
    def __init__(self, args):
        super().__init__(args)

        if self.args.optimizer == 'SGD':
            self.optimizer1 = optim.SGD(self.args.network1.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=5e-4)
            self.optimizer2 = optim.SGD(self.args.network2.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=5e-4)
        else:
            self.optimizer1 = optim.Adam(self.args.network1.parameters(), lr=self.args.lr)
            self.optimizer2 = optim.Adam(self.args.network2.parameters(), lr=self.args.lr)

        self.metric = torch.nn.CrossEntropyLoss(reduction='none').to(self.args.device)

    def update_model(self):
        self.args.network1.train()
        self.args.network2.train()
        for index, images, classes, labels in self.trainloader:
            images = images.to(self.args.device)
            labels = labels.to(self.args.device)
            outputs1 = self.args.network1(images)
            outputs2 = self.args.network2(images)
            # loss
            loss1 = self.criterion(outputs1, labels)
            self.optimizer1.zero_grad()
            loss1.backward()
            self.optimizer1.step()
            loss2 = self.criterion(outputs2, labels)
            self.optimizer2.zero_grad()
            loss2.backward()
            self.optimizer2.step()

        return

    def update_model_coteaching(self):
        self.args.network1.train()
        self.args.network2.train()
        epoch_loss, epoch_class_accuracy, epoch_label_accuracy = 0, 0, 0
        for index, images, classes, labels in self.trainloader:
            images = images.to(self.args.device)
            labels = labels.to(self.args.device)
            outputs1 = self.args.network1(images)
            outputs2 = self.args.network2(images)

            # select samples
            _, net1_index = torch.sort(self.metric(outputs1, labels))
            _, net2_index = torch.sort(self.metric(outputs2, labels))
            num_remember = int((1-float(self.args.noisy_ratio)) * len(labels))
            net1_index, net2_index = net1_index[:num_remember], net2_index[:num_remember]

            # loss
            loss1 = self.criterion(outputs1[net2_index], labels[net2_index])
            self.optimizer1.zero_grad()
            loss1.backward()
            self.optimizer1.step()
            loss2 = self.criterion(outputs2[net1_index], labels[net1_index])
            self.optimizer2.zero_grad()
            loss2.backward()
            self.optimizer2.step()
            epoch_loss += loss1.item() * len(labels)

            # accuracy
            _, model_label = torch.max(outputs1.detach(), dim=1)
            epoch_class_accuracy += (classes == model_label.cpu()).sum().item()
            epoch_label_accuracy += (labels == model_label).cpu().sum().item()

        time_elapse = time.time() - self.time
        return epoch_loss, epoch_class_accuracy, epoch_label_accuracy, time_elapse

    def evaluate_model(self):
        # calculate test accuracy
        self.args.network1.eval()
        epoch_accuracy = 0
        for _, images, classes, _ in self.testloader:
            images = images.to(self.args.device)
            classes = classes.to(self.args.device)
            outputs = self.args.network1(images)
            # accuracy
            _, model_label = torch.max(outputs.detach(), dim=1)
            epoch_accuracy += (classes == model_label).cpu().sum().item()

        time_elapse = time.time() - self.time
        return epoch_accuracy, time_elapse

    def run(self):
        print('\n===> Training Start: Coteaching')
        if self.args.dataset in ['MNIST','FMNIST','SVHN','CIFAR10','CIFAR100']:
            preepoch = 10
        else:
            preepoch = 2
        for epoch in range(preepoch):
            self.update_model()
        # train model
        for epoch in range(self.args.total_epochs):
            epoch_loss, epoch_class_acc, epoch_label_acc, time_train = self.update_model_coteaching()
            epoch_test_acc, time_test = self.evaluate_model()
            print('=' * 50)
            print('Epoch', epoch, 'Time', time_train, time_test)
            self.report_result(epoch_loss, epoch_class_acc, epoch_label_acc, epoch_test_acc)

        save_csv(self.args.result_dir, [self.loss_train, self.train_class_acc, self.train_label_acc, self.test_acc])
        torch.save(self.args.network.state_dict(), os.path.join(self.args.model_dir, 'classifier.pk'))

        return
