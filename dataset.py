import os
from PIL import Image
# import pickle
# from collections import namedtuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms


# from torchvision import datasets
# import lmdb


class LaionDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.data = []
        for f in os.listdir(self.path):
            t = os.path.splitext(f)[0]
            img_path = os.path.join(self.path, t + '.jpg')
            text_path = os.path.join(self.path, t + '.txt')
            if not os.path.exists(img_path) or not os.path.exists(text_path):
                continue
            self.data.append(t)
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        img_path = os.path.join(self.path, self.data[index] + '.jpg')
        text_path = os.path.join(self.path, self.data[index] + '.txt')

        img_ = Image.open(img_path).convert('RGB')
        img = self.transform(img_)

        # print(text_path)
        with open(text_path, 'r') as f:
            text = f.read().strip()

        # print(type(text))
        return img, text

# CodeRow = namedtuple('CodeRow', ['top', 'bottom', 'filename'])
#
#
# class ImageFileDataset(datasets.ImageFolder):
#     def __getitem__(self, index):
#         sample, target = super().__getitem__(index)
#         path, _ = self.samples[index]
#         dirs, filename = os.path.split(path)
#         _, class_name = os.path.split(dirs)
#         filename = os.path.join(class_name, filename)
#
#         return sample, target, filename
#
#
# class LMDBDataset(Dataset):
#     def __init__(self, path):
#         self.env = lmdb.open(
#             path,
#             max_readers=32,
#             readonly=True,
#             lock=False,
#             readahead=False,
#             meminit=False,
#         )
#
#         if not self.env:
#             raise IOError('Cannot open lmdb dataset', path)
#
#         with self.env.begin(write=False) as txn:
#             self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))
#
#     def __len__(self):
#         return self.length
#
#     def __getitem__(self, index):
#         with self.env.begin(write=False) as txn:
#             key = str(index).encode('utf-8')
#
#             row = pickle.loads(txn.get(key))
#
#         return torch.from_numpy(row.top), torch.from_numpy(row.bottom), row.filename
