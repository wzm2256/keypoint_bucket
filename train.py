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

def Eval(my_network, testloader, pixel_xy, args):
    All_L_loss = torch.zeros((0))
    All_variance_loss = torch.zeros((0))
    All_name = []
    with torch.no_grad():
        for i, Data  in enumerate(tqdm(testloader)):
            I, label, name_list = Data
            I_cuda, label_cuda = I.cuda(), label.cuda()
            
            
            feature = my_network(I_cuda)

            _, variance, L_regression = util.my_loss(feature, label_cuda, pixel_xy)
            
            All_variance_loss = torch.cat([All_variance_loss, variance.to('cpu').detach()])
            All_L_loss = torch.cat([All_L_loss, L_regression.to('cpu').detach()])
            All_name += name_list

    Loss = All_L_loss.mean() + args.lambdav * All_variance_loss.mean()
    # pdb.set_trace()
    return All_L_loss, All_variance_loss, All_name, Loss


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
    
    my_network = UNet(3, 3).to('cuda')

    optimizer = optim.Adam(my_network.parameters(), lr=args.lr_D)

    Max_test_acc = 100.
    for e in range(args.epoch):

        for i, Data  in enumerate(tqdm(trainloader)):
            I, label, _ = Data
            I_cuda, label_cuda = I.cuda(), label.cuda()
            feature = my_network(I_cuda)

            # direction, variance = util.direction(feature, pixel_xy)
            # L_regression = util.my_loss(direction, label_cuda)
            
            direction, variance, L_regression = util.my_loss(feature, label_cuda, pixel_xy)
            L = (L_regression.mean() + args.lambdav * variance.mean()) * 100

            # print(f'Epoch {e} batch{i} Var: {variance.mean().item():.4f} Regression: {L_regression.mean().item():.4f} ')

            optimizer.zero_grad()
            L.backward()
            optimizer.step()

        my_network.eval()
        All_L_loss, All_variance_loss, All_name, Loss = Eval(my_network, testloader, pixel_xy, args)
        print('Test------Epoch {}: \t Loss{}'.format(e, Loss))
        if Loss <= Max_test_acc:
            print('Saving the best model')
            torch.save(my_network.state_dict(), 'best.pt')
            Max_test_acc = Loss

            with open('Record.txt', 'w') as f:
                for r in range(len(All_name)):
                    f.write(f'{All_name[r]}\t{All_L_loss[r]}\t{All_variance_loss[r]}\n')
        
        my_network.train()


if __name__ == '__main__':

    agsps = argparse.ArgumentParser()
    agsps.add_argument('--size', type=int, default=227)
    agsps.add_argument('--data_root', type=str, default=r'raw_image/YOLODataset')
    agsps.add_argument('--train_bs', type=int, default=4)
    agsps.add_argument('--lr_D', type=float, default=1e-4)
    agsps.add_argument('--lambdav', type=float, default=0.1)
    agsps.add_argument('--epoch', type=int, default=30)

    args = agsps.parse_args()

    main(args)