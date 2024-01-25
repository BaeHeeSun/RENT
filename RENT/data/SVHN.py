import os.path
from typing import Any, Callable, Optional, Tuple

import pickle
import numpy as np
from PIL import Image

from torchvision.datasets.utils import download_url, check_integrity, verify_str_arg
from torchvision.datasets.vision import VisionDataset

import data.noise_generator as nlgen


class SVHN(VisionDataset):
    """`SVHN <http://ufldl.stanford.edu/housenumbers/>`_ Dataset.
    Note: The SVHN dataset assigns the label `10` to the digit `0`. However, in this Dataset,
    we assign the label `0` to the digit `0` to be compatible with PyTorch loss functions which
    expect the class labels to be in the range `[0, C-1]`

    .. warning::

        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load data from `.mat` format.

    Args:
        root (string): Root directory of dataset where directory
            ``SVHN`` exists.
        split (string): One of {'train', 'test', 'extra'}.
            Accordingly dataset is selected. 'extra' is Extra training set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    split_list = {
        "train": [
            "http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
            "train_32x32.mat",
            "e26dedcc434d2e4c54c9b2d4a06d8373",
        ],
        "test": [
            "http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
            "test_32x32.mat",
            "eb5a983be6a315427106f1b164d9cef3",
        ],
        "extra": [
            "http://ufldl.stanford.edu/housenumbers/extra_32x32.mat",
            "extra_32x32.mat",
            "a93ce644f1a588dc4d68dda5feec44a7",
        ],
    }

    def __init__(
        self,
        args,
        root: str,
        split: str = "train",
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, target_transform=target_transform)

        self.args = args  # arguments
        self.root = root
        self.split = verify_str_arg(split, "split", tuple(self.split_list.keys()))

        self.url = self.split_list[split][0]
        self.filename = self.split_list[split][1]
        self.file_md5 = self.split_list[split][2]

        self.transform = args.transform
        self.test_transform = args.test_transform

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        # import here rather than at top of file because this is
        # an optional dependency for torchvision
        import scipy.io as sio

        # reading(loading) mat file as array
        loaded_mat = sio.loadmat(os.path.join(self.root, self.filename))

        self.data = loaded_mat["X"]
        # loading from the .mat file gives an np array of type np.uint8
        # converting to np.int64, so that we have a LongTensor after
        # the conversion from the numpy array
        # the squeeze is needed to obtain a 1D tensor
        self.targets = loaded_mat["y"].astype(np.int64).squeeze()

        # the svhn dataset assigns the class label "10" to the digit 0
        # this makes it inconsistent with several loss functions
        # which expect the class labels to be in the range [0, C-1]
        np.place(self.targets, self.targets == 10, 0)
        self.data = np.transpose(self.data, (3, 2, 0, 1))

        self.noisy_pickle_data_dir = os.path.join('./data', 'NOISY', self.args.dataset, self.args.noise_type + '_' + self.args.noisy_ratio + '.pk')

        if self.split == 'train':
            if os.path.exists(self.noisy_pickle_data_dir):
                self.noisy_data_load()
            else:
                print('No data found. Generate noisy data...')
                os.makedirs(os.path.join('./data', 'NOISY', self.args.dataset), exist_ok=True)
                self.generate_noisy_data(args)
        else:
            self.noisy_labels = self.targets

    def generate_noisy_data(self, args):
        if self.args.noise_type == 'clean':
            self.noisy_labels = self.targets
        elif self.args.noise_type == 'sym':
            self.noisy_labels = nlgen.generate_noisy_label_symmetric(args, self.targets)
        elif self.args.noise_type == 'asym':
            self.noisy_labels = nlgen.generate_noisy_label_asymmetric(args, self.targets)
        elif self.args.noise_type == 'idn':
            self.noisy_labels = nlgen.generate_noisy_label_idn(args, self.data, self.targets)
        elif self.args.noise_type == 'open': ##### really?
            self.noisy_labels = None
        else:
            self.noisy_labels = None

        with open(self.noisy_pickle_data_dir, 'wb') as f:
            pickle.dump(self.noisy_labels,f)

    def noisy_data_load(self):
        with open(self.noisy_pickle_data_dir, 'rb') as f:
            self.noisy_labels = pickle.load(f)

        print('Saved data loaded...')

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, label = self.data[index], int(self.targets[index]), self.noisy_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            if self.split == 'train':
                img = self.transform(img)
            else:
                img = self.test_transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
            label = self.target_transform(label)

        return index, img, target, label

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        root = self.root
        md5 = self.split_list[self.split][2]
        fpath = os.path.join(root, self.filename)
        return check_integrity(fpath, md5)

    def download(self) -> None:
        md5 = self.split_list[self.split][2]
        download_url(self.url, self.root, self.filename, md5)

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)
