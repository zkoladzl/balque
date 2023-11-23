from torch.utils.data.dataset import Dataset
from torchvision import datasets
class CIFAR10_train(Dataset):
    def __init__(self, path, transforms):
        self.dataset = datasets.CIFAR10(path, train = True, transform= transforms, download= True)
    def __getitem__(self, index):
        img, label = self.dataset[index]
        return img, label, index
    def __len__(self) -> int:
        return len(self.dataset)
class CIFAR10_test(Dataset):
    def __init__(self, path, transforms):
        self.dataset = datasets.CIFAR10(path, train = False, transform= transforms)
    def __getitem__(self, index):
        img, label = self.dataset[index]
        return img, label, index
    def __len__(self) -> int:
        return len(self.dataset)
class CIFAR100_train(Dataset):
    def __init__(self, path, transforms):
        self.dataset = datasets.CIFAR100(path, train = True, transform= transforms, download=True)
    def __getitem__(self, index):
        img, label = self.dataset[index]
        return img, label, index
    def __len__(self) -> int:
        return len(self.dataset)
class CIFAR100_test(Dataset):
    def __init__(self, path, transforms):
        self.dataset = datasets.CIFAR100(path, train = False, transform= transforms, download= True)
    def __getitem__(self, index):
        img, label = self.dataset[index]
        return img, label, index
    def __len__(self) -> int:
        return len(self.dataset)
class SVHN_train(Dataset):
    def __init__(self, path, transforms):
        self.dataset = datasets.SVHN(path, split = 'train', transform= transforms)
    def __getitem__(self, index):
        img, label = self.dataset[index]
        return img, label, index
    def __len__(self) -> int:
        return len(self.dataset)
class SVHN_test(Dataset):
    def __init__(self, path, transforms):
        self.dataset = datasets.SVHN(path, split = 'test', transform=transforms)
    def __getitem__(self, index):
        img, label = self.dataset[index]
        return img, label, index
    def __len__(self) -> int:
        return len(self.dataset)