from torch.utils.data import Dataset
import os
from PIL import Image

class FoodN(Dataset): #food101N
    def __init__(self, args, root, train=True):
        self.train = train
        self.class_txt = root+'/Food101N/meta/classes.txt'

        # class to cat
        self.name2cat = dict()
        with open(self.class_txt) as fp:
            for i, line in enumerate(fp):
                row = line.strip()
                self.name2cat[row] = i-1

        self.images, self.targets = [], []

        if self.train:
            self.root = root + '/Food101N'
            self.data_tsv = self.root + '/meta/imagelist.tsv'
            self.transform = args.transform
            with open(self.data_tsv) as f:
                f.readline()  # skip first line
                for line in f:
                    row = line.strip().split('/')
                    class_name = row[0]
                    self.images.append(os.path.join('images', line.strip()))
                    self.targets.append(self.name2cat[class_name])

        # else:
        #     self.root = root+'/food-101'
        #     self.transform = args.test_transform
        #     with open(self.root+'/meta/test.txt') as f:
        #         for line in f:
        #             row = line.strip().split('/')
        #             class_name = row[0]
        #             self.images.append(os.path.join('images', line.strip() + '.jpg'))
        #             self.targets.append(self.name2cat[class_name])

    def __getitem__(self, item):
        img_path = self.images[item]
        label = self.targets[item]
        image = Image.open(self.root + '/' + img_path)
        img = self.transform(image)

        return item, img, label, label

    def __len__(self):
        return len(self.targets)

class Clothing(Dataset): # clothing1m
    def __init__(self, args, root, train=True):
        self.root = root+'/Clothing'
        self.train=train
        self.images, self.targets = [], []

        if self.train:
            self.transform = args.transform
            with open('%s/annotations/noisy_label_kv.txt' % self.root, 'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    entry = l.split()
                    img_path = entry[0][7:]
                    self.images.append(img_path)
                    self.targets.append(int(entry[1]))
        else:
            self.transform = args.test_transform
            with open('%s/annotations/clean_label_kv.txt' % self.root, 'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    entry = l.split()
                    img_path = entry[0][7:]
                    self.images.append(img_path)
                    self.targets.append(int(entry[1]))

    def __getitem__(self, item):
        img_path = self.images[item]
        label = self.targets[item]
        image = Image.open(self.root + '/' + img_path).convert('RGB')
        img = self.transform(image)

        return item, img, label, label

    def __len__(self):
        return len(self.targets)
