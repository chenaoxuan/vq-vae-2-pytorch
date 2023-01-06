import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from dataset import LaionDataset

data_path = './dataset/0/'

# transform = transforms.Compose(
#     [
#         transforms.Resize(32),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
#     ]
# )
#
# dataset = datasets.ImageFolder(data_path, transform=transform)
# loader = DataLoader(dataset, batch_size=128, num_workers=0)
#
# for i, (img, label) in enumerate(loader):
#     print(i)
#     print("img", img.shape)

dataset = LaionDataset(path=data_path)
loader = DataLoader(dataset, batch_size=2, num_workers=0)
for i, (img, text) in enumerate(loader):
    print(i, ": ", img.shape)
    print(i, ": ", text)
    if i==2:
        break
