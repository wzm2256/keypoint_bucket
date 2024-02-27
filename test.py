import os.path

from torch.utils.data import DataLoader
from torchvision import transforms
# from torchvision.models import resnet50, ResNet50_Weights
import argparse
import torch
import pdb
# import net
import torch.optim as optim
# import net1
# import network
import torchvision.models as models
import numpy as np
import dataset
import torchvision
from torchvision import tv_tensors
from unet.unet_model import UNet
import util
from tqdm import tqdm
from torchvision.transforms import v2
import PIL.Image as Image
import cv2 as cv
from pathlib import Path
import train


def main(args):
	y_size = args.size
	x_size = args.size
	x = np.linspace(-1, 1, x_size)
	y = np.linspace(1, -1, y_size)
	xx, yy = np.meshgrid(x, y, indexing='xy')
	pixel_xy = torch.tensor(np.stack([xx, yy], 0), device='cuda')

	D_train = dataset.My_dataset(args.data_root, train=True, size=args.size)
	D_test = dataset.My_dataset(args.data_root, train=False, size=args.size)

	trainloader = DataLoader(D_train, batch_size=args.train_bs, shuffle=True, num_workers=0, drop_last=True)
	testloader = DataLoader(D_test, batch_size=args.train_bs, shuffle=False, num_workers=0, drop_last=False)
	#############Define Network

	my_network = UNet(3, 3)
	my_network.load_state_dict(torch.load(args.model_weight, map_location='cpu'))
	my_network.to('cuda')
	my_network.eval()

	print('Report test performance...')
	All_L_loss, All_variance_loss, All_name, Loss = train.Eval(my_network, testloader, pixel_xy, args)
	with open('all_test.txt', 'a') as f:
		for r in range(len(All_name)):
			f.write(f'{All_name[r]}\t{All_L_loss[r]}\t{All_variance_loss[r]}\n')

	print('Report train performance...')
	All_L_loss, All_variance_loss, All_name, Loss = train.Eval(my_network, trainloader, pixel_xy, args)
	with open('all_train.txt', 'a') as f:
		for r in range(len(All_name)):
			f.write(f'{All_name[r]}\t{All_L_loss[r]}\t{All_variance_loss[r]}\n')



if __name__ == '__main__':

    agsps = argparse.ArgumentParser()
    agsps.add_argument('--size', type=int, default=227)
    agsps.add_argument('--model_weight', type=str, default='best.pt')
    agsps.add_argument('--data_root', type=str, default=r'/home/wzm/MyCode/yolo/YOLODataset/')
    agsps.add_argument('--train_bs', type=int, default=4)
    agsps.add_argument('--lambdav', type=float, default=0.1)

    args = agsps.parse_args()

    main(args)