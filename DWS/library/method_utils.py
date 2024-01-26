import torch
import torch.nn as nn
import torch.optim as optim
import time
import torch.distributions.dirichlet as dir

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
        norm_w = w / torch.sum(w)

        m = dir.Dirichlet(self.args.alpha*norm_w)
        return torch.mean(torch.matmul(m.sample((1,self.args.batch_size))[0],self.crit(outputs,labels)))

class INIT(object):
    def __init__(self, args):
        print('\n===> Initialization')
        self.args = args
        self.trainloader, self.testloader = self.args.Loader.trainloader, self.args.Loader.testloader
        self.lentrain, self.lentest = self.args.Loader.lentrain, self.args.Loader.lentest
        self.criterion = nn.CrossEntropyLoss().to(self.args.device)  # mean
        self.nll = nn.NLLLoss().to(self.args.device)
        self.loss_func_reweight = reweight_loss(self.args)
        self.softmax = nn.Softmax(dim=1).to(self.args.device)

        if self.args.optimizer == 'SGD':
            self.optimizer = optim.SGD(self.args.network.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=5e-4)
        else:
            self.optimizer = optim.Adam(self.args.network.parameters(), lr=self.args.lr)

        self.proba = torch.zeros(self.lentrain, self.args.n_class)
        self.loss_train, self.train_class_acc, self.train_label_acc, self.test_acc = [], [], [], []

        self.time = time.time()

    def update_model(self):
        self.args.network.train()
        epoch_loss, epoch_class_accuracy, epoch_label_accuracy = 0, 0, 0
        for index, images, classes, labels in self.trainloader:
            images = images.to(self.args.device)
            labels = labels.to(self.args.device)
            outputs = self.args.network(images)
            # loss
            loss = self.criterion(outputs, labels)
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

    def update_model_reweight(self):
        self.args.network.train()
        epoch_loss, epoch_class_accuracy, epoch_label_accuracy = 0, 0, 0
        for index, images, classes, labels in self.trainloader:
            images = images.to(self.args.device)
            labels = labels.to(self.args.device)
            outputs = self.args.network(images)
            # loss
            loss = self.loss_func_reweight(outputs, self.transition, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
            # accuracy
            _, model_label = torch.max(outputs.detach(), dim=1)
            epoch_class_accuracy += (classes == model_label.cpu()).sum().item()
            epoch_label_accuracy += (labels == model_label).cpu().sum().item()

        time_elapse = time.time() - self.time
        return epoch_loss, epoch_class_accuracy, epoch_label_accuracy, time_elapse

    def evaluate_model(self):
        # calculate test accuracy
        self.args.network.eval()
        epoch_accuracy = 0
        for _, images, classes, _ in self.testloader:
            images = images.to(self.args.device)
            classes = classes.to(self.args.device)
            outputs = self.args.network(images)
            # accuracy
            _, model_label = torch.max(outputs.detach(), dim=1)
            epoch_accuracy += (classes == model_label).cpu().sum().item()

        time_elapse = time.time() - self.time
        return epoch_accuracy, time_elapse

    def report_result(self, epoch_loss, epoch_class_acc, epoch_label_acc, epoch_test_acc):
        epoch_loss /= self.lentrain
        epoch_class_acc /= self.lentrain
        epoch_label_acc /= self.lentrain
        epoch_test_acc /= self.lentest

        self.loss_train.append(epoch_loss)
        self.train_class_acc.append(epoch_class_acc)
        self.train_label_acc.append(epoch_label_acc)
        self.test_acc.append(epoch_test_acc)

        print('Train', epoch_loss, epoch_class_acc, epoch_label_acc)
        print('Test', epoch_test_acc)

        return

    def generate_transition_matrix(self):
        if self.args.noise_type == 'sym':
            n_rate = 1-float(self.args.noisy_ratio)*(self.args.n_class/(self.args.n_class-1))
            self.transition = n_rate*torch.eye(self.args.n_class)+\
                              (float(self.args.noisy_ratio)/(self.args.n_class-1))*torch.ones(self.args.n_class, self.args.n_class)
        elif self.args.noise_type == 'asym':
            if self.args.dataset != 'CIFAR100':
                n_rate = 1-float(self.args.noisy_ratio)
                self.transition = n_rate*torch.eye(self.args.n_class)+\
                                  (1-n_rate)*torch.zeros(self.args.n_class, self.args.n_class).scatter_(0,torch.tensor([self.args.noisy_label_list]),1)
            else:
                # alternative matrix preparation
                c_label = torch.tensor([4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                                        3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                                        6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                                        0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                                        5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                                        16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                                        10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                                        2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                                        16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                                        18, 1, 2, 15, 6, 0, 17, 8, 14, 13])

                alternative_mat = torch.zeros(self.args.n_class, self.args.n_class)
                for i in range(20):
                    for row in torch.where(c_label == i)[0]:
                        for col in torch.where(c_label == i)[0]:
                            alternative_mat[row.item()][col.item()] = 1.0
                n_rate = 1 - float(self.args.noisy_ratio) * (5/4)
                self.transition = n_rate*torch.eye(self.args.n_class)+(float(self.args.noisy_ratio)/4)*alternative_mat
        elif self.args.noise_type == 'idn': # 얘는 뭘 할 수가 없잖아
            self.transition = None
        else: # clean
            self.transition = torch.eye(self.args.n_class)

        self.transition = self.transition.to(self.args.device)
        return
