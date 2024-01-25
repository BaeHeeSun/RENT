import torch
import torch.nn.functional as F
import os
import time

from utils import save_csv
from library.method_utils import INIT

# https://openreview.net/pdf?id=2VXyy9mIyU3
# https://github.com/haochenglouis/cores

class CORES(INIT):
    def __init__(self, args):
        super().__init__(args)
        print('\n===> Training Start: CORES')

    def loss_cores(self, output, label):
        loss = F.cross_entropy(output, label, reduce=False)
        reg = -torch.log(self.softmax(output)+1e-12)

        # sel metric
        loss_v = torch.ones(len(label)).to(self.args.device)
        loss_v[loss>torch.mean(reg, 1)] = 0
        loss_ = (loss-2*torch.mean(reg, 1))[loss_v==1]

        return torch.sum(loss_) / sum(loss_v+1e-12)

    def update_model_cores(self):
        self.args.network.train()
        epoch_loss, epoch_class_accuracy, epoch_label_accuracy = 0, 0, 0
        for index, images, classes, labels in self.trainloader:
            images = images.to(self.args.device)
            labels = labels.to(self.args.device)
            outputs = self.args.network(images)
            # loss
            loss = self.loss_cores(outputs, labels)
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
        if self.args.dataset == 'Clothing':
            preepoch = 1
        else:
            preepoch = 10

        for i in range(preepoch):
            self.update_model()

        # train model
        for epoch in range(self.args.total_epochs):
            epoch_loss, epoch_class_acc, epoch_label_acc, time_train = self.update_model_cores()
            epoch_test_acc, time_test = self.evaluate_model()
            print('=' * 50)
            print('Epoch', epoch, 'Time', time_train, time_test)
            self.report_result(epoch_loss, epoch_class_acc, epoch_label_acc, epoch_test_acc)

        save_csv(self.args.result_dir, [self.loss_train, self.train_class_acc, self.train_label_acc, self.test_acc])
        torch.save(self.args.network.state_dict(), os.path.join(self.args.model_dir, 'classifier.pk'))

        return