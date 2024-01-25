import torch
import os

from utils import save_csv
from library.method_utils import INIT

# cross entropy loss. baseline

class CE(INIT):
    def __init__(self, args):
        super().__init__(args)
        print('\n===> Training Start: Baseline')

    def run(self):
        # train model
        for epoch in range(self.args.total_epochs):
            epoch_loss, epoch_class_acc, epoch_label_acc, time_train = self.update_model()
            epoch_test_acc, time_test = self.evaluate_model()
            print('=' * 50)
            print('Epoch', epoch, 'Time', time_train, time_test)
            self.report_result(epoch_loss, epoch_class_acc, epoch_label_acc, epoch_test_acc)

        save_csv(self.args.result_dir, [self.loss_train, self.train_class_acc, self.train_label_acc, self.test_acc])
        torch.save(self.args.network.state_dict(), os.path.join(self.args.model_dir, 'classifier.pk'))
        
        return

