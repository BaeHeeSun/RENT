import argparse
import torchvision.transforms as transforms

####################################################################################################################
parser = argparse.ArgumentParser(description="main")

# data condition
parser.add_argument('--dataset', type=str, default=None, help = 'MNIST, FMNIST, SVHN, CIFAR10, CIFAR100, Food, Clothing, Webvision')
parser.add_argument('--noise_type', type=str, default='clean', help='clean, sym, asym')
parser.add_argument('--noisy_ratio', type=str, default=None, help='between 0 and 1')

# classifier condition
parser.add_argument('--class_method', type=str, default=None, help='classifier method')

# experiment condition
parser.add_argument('--optimizer', type=str, default='Adam', help='SGD, Adam')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--lr', type=float, default=0.001, help = "Learning rate (Default : 1e-3)")
parser.add_argument('--total_epochs', type=int, default=200, help='total training epoch')
parser.add_argument('--N', type=float, default=1.0, help='how much samples to choose?')

# etc
parser.add_argument('--set_gpu', type=str, default='0', help='gpu setting')
parser.add_argument('--data_dir', type=str, default='./data/')

args = parser.parse_args()
####################################################################################################################

# Dataset Information
if args.dataset == 'MNIST':
    args.n_channel = 1
    args.noisy_label_list = [0, 1, 7, 8, 4, 6, 5, 7, 8, 9]
    args.n_class = 10
    args.transform = transforms.Compose([transforms.ToTensor()])
    args.batch_size = 128

elif args.dataset == 'FMNIST':
    args.n_channel = 1
    args.noisy_label_list = [6, 1, 4, 3, 4, 7, 6, 7, 8, 9]
    args.n_class = 10
    args.transform = transforms.Compose([transforms.ToTensor()])
    args.batch_size = 128

elif args.dataset == 'SVHN':
    args.n_channel = 3
    args.noisy_label_list = [0, 1, 7, 8, 4, 6, 5, 7, 8, 9]
    args.n_class = 10
    args.transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor()
    ])
    args.test_transform = transforms.Compose([transforms.ToTensor()])
    args.batch_size = 128

elif args.dataset == 'CIFAR10':
    args.n_channel = 3
    args.noisy_label_list = [0,1,0,5,7,3,6,7,8,1]
    args.n_class = 10
    args.transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor()
    ])
    args.test_transform = transforms.Compose([transforms.ToTensor()])
    args.batch_size = 128

elif args.dataset == 'CIFAR100':
    args.n_channel = 3
    args.n_class = 100
    args.transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor()
    ])
    args.test_transform = transforms.Compose([transforms.ToTensor()])
    args.batch_size = 128

#############################################################################################
elif args.dataset in ['Food', 'FoodN']:
    args.n_channel = 3
    args.n_class = 101
    args.noise_type = 'nat'
    args.noisy_ratio = '0.2'
    args.transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    args.test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    args.batch_size = 100

elif args.dataset == 'Clothing':
    args.n_channel = 3
    args.n_class = 14
    args.noise_type = 'nat'
    args.noisy_ratio = '0.385'
    args.transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)),
    ])
    args.test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)),
    ])
    args.batch_size = 100

# # Possible Future Dataset
# elif args.dataset == 'Webvision':
#     args.n_channel = 3
#     args.n_class = 50
#     args.noisy_ratio = '0.42'
#     args.train_transform = transforms.Compose([
#         transforms.Resize(256),
#         transforms.RandomCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#
#     args.test_transform = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#     ])
#     args.batch_size = 64

else:
    args.n_channel = -1
    args.n_class = -1
    args.train_transform = None
    args.test_transform = None
    args.batch_size = None


# Dummy setting for code implementations
if args.noise_type == 'clean':
    args.noisy_ratio = '0.0'