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

def predict_one(trans, image, model, save_folder, pixel_xy):
	with Image.open(image) as I:
		img = tv_tensors.Image(I)
		image_tensor = trans(img).unsqueeze(0)

	with torch.no_grad():
		feature = model(image_tensor)

	direction, _, _ = util.my_loss(feature, None, pixel_xy)

	# pdb.set_trace()
	direction_x, direction_y = direction[0][0], direction[0][1]

	p = Path(image)
	txt_file_name = os.path.join(save_folder, 'labels', f"keypoint_{p.name.split('.')[0]}.txt")
	img_file_name = os.path.join(save_folder, f"keypoint_{p.name}")

	src = cv.imread(cv.samples.findFile(image), cv.IMREAD_COLOR)
	src = cv.resize(src, (512, 512))
	# pdb.set_trace()
	center = ((direction_x + 1) / 2, (1 - direction_y) / 2)

	cv.circle(src, (int(center[0] * 512), int(center[1] * 512)), 25, (0, 255, 255), 3)

	print(f'File {p.name}; \t key point ({center[0].item():.3f}, {center[1].item():.3f})')
	with open(txt_file_name, 'w') as f:
		f.write(f'{center[0]} {center[1]}')
	cv.imwrite(img_file_name, src)
	print(f'Keypoint saved to {txt_file_name}')
	print(f'Image with keypoint saved to {img_file_name}')



def main(args):

	y_size = args.size
	x_size = args.size
	x = np.linspace(-1, 1, x_size)
	y = np.linspace(1, -1, y_size)
	xx, yy = np.meshgrid(x, y, indexing='xy')
	pixel_xy = torch.tensor(np.stack([xx, yy], 0),)

	if os.path.isdir(args.data):
		All = [os.path.join(args.data, i) for i in os.listdir(args.data)]
	else:
		All = [args.data]

	my_network = UNet(3, 3)
	my_network.load_state_dict(torch.load(args.model_weight, map_location='cpu'))
	my_network.eval()

	trans = v2.Compose([
		v2.Resize(size=(args.size, args.size)),
		v2.ToDtype(torch.float32, scale=True),
		v2.Normalize(mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0]),
	])

	for img in All:
		predict_one(trans, img, my_network, args.save_path, pixel_xy)


if __name__ == '__main__':

    agsps = argparse.ArgumentParser()
    agsps.add_argument('--size', type=int, default=227)
    agsps.add_argument('--model_weight', type=str, default='best.pt')
    agsps.add_argument('--data', type=str, default='')
    agsps.add_argument('--save_path', type=str, default='Result')


    args = agsps.parse_args()

    main(args)