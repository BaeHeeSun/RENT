from torch.utils.data import DataLoader

from data.MNIST import MNIST,FashionMNIST
from data.SVHN import SVHN
from data.CIFAR import CIFAR10, CIFAR100
# from data.Food import Food101
from data.Real import FoodN, Clothing

class dataloader(): # MNIST/FMNIST/SVHN/CIFAR10/CIFAR100
    def __init__(self, args):
        # define data
        base_data_dir = args.data_dir
        if args.dataset == 'MNIST':
            trainset = MNIST(args=args, root=base_data_dir, train=True, download=True)
            testset = MNIST(args=args, root=base_data_dir, train=False)
        elif args.dataset == 'FMNIST':
            trainset = FashionMNIST(args=args, root=base_data_dir, train=True, download=True)
            testset = FashionMNIST(args=args, root=base_data_dir, train=False)
        elif args.dataset == 'SVHN':
            trainset = SVHN(args=args, root=base_data_dir+'SVHN/', split='train', download=True)
            testset = SVHN(args=args, root=base_data_dir+'SVHN/', split='test', download=True)
        elif args.dataset == 'CIFAR10':
            trainset = CIFAR10(args=args, root=base_data_dir, train=True, download=True)
            testset = CIFAR10(args=args, root=base_data_dir, train=False)
        elif args.dataset == 'CIFAR100':
            trainset = CIFAR100(args=args, root=base_data_dir, train=True, download=True)
            testset = CIFAR100(args=args, root=base_data_dir, train=False)
        elif args.dataset == 'Food':
            trainset = Food101(root=base_data_dir, transform=args.transform, split='train', download=True)
            testset = Food101(root=base_data_dir, transform=args.test_transform, split='test')
        elif args.dataset == 'FoodN':
            trainset = FoodN(args=args, root=base_data_dir, train=True)
            testset = Food101(root=base_data_dir, transform=args.test_transform, split='test', download=True)
        elif args.dataset == 'Clothing':
            trainset = Clothing(args=args, root=base_data_dir, train=True)
            testset = Clothing(args=args, root=base_data_dir, train=False)
        else:
            trainset = None
            testset = None

        self.lentrain, self.lentest = len(trainset), len(testset)

        self.trainloader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=8)
        self.testloader = DataLoader(dataset=testset, batch_size=200, shuffle=False, num_workers=8)
