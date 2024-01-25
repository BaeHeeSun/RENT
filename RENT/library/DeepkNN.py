import torch
import os
import time
from sklearn.neighbors import KNeighborsClassifier

from utils import save_csv
from library.method_utils import INIT

# http://proceedings.mlr.press/v119/bahri20a/bahri20a.pdf no official code...

class DKNN(INIT):
    def __init__(self, args):
        super().__init__(args)
        print('\n===> Training Start: Deep KNN')

        self.neigh = KNeighborsClassifier(n_neighbors=500, weights='distance')

    def update_model_knn(self):
        self.args.network.train()
        epoch_loss, epoch_class_accuracy, epoch_label_accuracy = 0, 0, 0

        for index, images, classes, labels in self.trainloader:
            selection_index = (self.knn_label[index]==labels).to(self.args.device)

            images = images.to(self.args.device)
            labels = labels.to(self.args.device)
            outputs = self.args.network(images)

            # loss
            loss = self.criterion(outputs[selection_index], labels[selection_index])
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item() * len(labels)
            # accuracy
            _, model_label = torch.max(outputs.detach(), dim=1)
            epoch_class_accuracy += (classes == model_label.cpu()).sum().item()
            epoch_label_accuracy += (labels == model_label).cpu().sum().item()

            self.proba[index] = self.softmax(outputs).detach().cpu()

        time_elapse = time.time() - self.time
        return epoch_loss, epoch_class_accuracy, epoch_label_accuracy, time_elapse

    def run(self):
        if self.args.dataset in ['MNIST','FMNIST','SVHN','CIFAR10','CIFAR100']:
            preepoch = 10
        else:
            preepoch = 2
        for epoch in range(preepoch):
            self.update_model()

        softmax_output = []
        labels_list = []
        for index, images, _, labels in self.trainloader:
            images = images.to(self.args.device)
            outputs = self.args.network(images)
            softmax_output+=self.softmax(outputs).detach().cpu().tolist()
            labels_list+=labels.tolist()

        self.neigh.fit(softmax_output, labels_list)
        self.knn_label = torch.tensor(self.neigh.predict(softmax_output)).long()

        # train model
        for epoch in range(self.args.total_epochs):
            epoch_loss, epoch_class_acc, epoch_label_acc, time_train = self.update_model_knn()
            epoch_test_acc, time_test = self.evaluate_model()
            print('=' * 50)
            print('Epoch', epoch, 'Time', time_train, time_test)
            self.report_result(epoch_loss, epoch_class_acc, epoch_label_acc, epoch_test_acc)

        save_csv(self.args.result_dir, [self.loss_train, self.train_class_acc, self.train_label_acc, self.test_acc])
        torch.save(self.args.network.state_dict(), os.path.join(self.args.model_dir, 'classifier.pk'))

        return