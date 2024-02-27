import torchvision.datasets as datasets
from torch.utils.data import Dataset
import os
import random
import PIL.Image as Image
import pdb
import torch
import torch.nn as nn
# from torchvision import transforms
from torchvision.transforms import v2
from torchvision import tv_tensors

class My_dataset(Dataset):
    def __init__(self, root, train=True, size=1024):
        super().__init__()
        
        if train:
            self.trans = v2.Compose([
                v2.Resize(size=(size, size)),
                v2.RandomPhotometricDistort(p=0.5),
                # v2.RandomRotation(10),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0]),
            ])
            self.img_folder = os.path.join(root, 'images', 'train')
            self.label_folder = os.path.join(root, 'labels', 'train')
            
        else:
            self.trans = v2.Compose([
                v2.Resize(size=(size, size)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0]),
            ])
            self.img_folder = os.path.join(root, 'images', 'val')
            self.label_folder = os.path.join(root, 'labels', 'val')


        self.All_name = [i.split('.')[0] for i in  os.listdir(self.img_folder)]

    def __getitem__(self, index):
        
        img_name = os.path.join(self.img_folder, self.All_name[index] + '.png')
        label_name = os.path.join(self.label_folder, self.All_name[index] + '.txt')

        with Image.open(img_name) as I:
            with open(label_name) as f:
                img = tv_tensors.Image(I)
                h, w = img.shape[1], img.shape[2]
                line = f.readline()
                bb = [float(i) for i in line.strip().split(' ')[1:]]
                bb = [w* bb[0], h * bb[1], w* bb[2], h * bb[3]]
                bb = tv_tensors.BoundingBoxes([bb], canvas_size=(h, w), format='CXCYWH')

        image, label = self.trans(img, bb)
        vec = torch.tensor([label[0][0] / image.shape[2] - 0.5, 1 - label[0][1] / image.shape[1] - 0.5])
        return image, vec * 2, self.All_name[index]

    def __len__(self):
        return len(self.All_name)