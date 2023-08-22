from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import copy
from torchvision import transforms
from torch.utils.data import DataLoader
from dataloader.driverloader import StereoDataset  
from models import *  


data_root = '/content/drive/MyDrive/drivingstereo/foggy'


batch_size_train = 1
max_epochs = 150
learning_rate = 0.001


transform = transforms.Compose([
    transforms.ToTensor(),  
])


train_dataset = StereoDataset(data_root=data_root, transform=transform)



train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, num_workers=8)

model = stackhourglass(maxdisp=192)  


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))


def train(imgL, imgR, disp_L):
    model.train()
    imgL = Variable(imgL.to(device))
    imgR = Variable(imgR.to(device))
    disp_L = Variable(disp_L.to(device))

    mask = (disp_L > 0)
    mask.detach_()

    optimizer.zero_grad()

    output1, output2, output3 = model(imgL, imgR)

    
    disp_L_resized = F.interpolate(disp_L, size=output1.shape[2:], mode='bilinear', align_corners=True)

    loss = 0.5 * F.smooth_l1_loss(output1[mask], disp_L_resized[mask], reduction='mean') + \
           0.7 * F.smooth_l1_loss(output2[mask], disp_L_resized[mask], reduction='mean') + \
           F.smooth_l1_loss(output3[mask], disp_L_resized[mask], reduction='mean')

    loss.backward()
    optimizer.step()

    return loss.item()

def main():
    start_full_time = time.time()

    try:
        for epoch in range(1, max_epochs + 1):
            total_train_loss = 0

            for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(train_loader):
                start_time = time.time()

                loss = train(imgL_crop, imgR_crop, disp_crop_L)
                print('Epoch %d Iter %d training loss = %.3f, time = %.2f' % (epoch, batch_idx, loss, time.time() - start_time))
                total_train_loss += loss

            print('Epoch %d total training loss = %.3f' % (epoch, total_train_loss / len(train_loader)))

            # SAVE
            savefilename = f'./finetune_{epoch}.tar'
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'train_loss': total_train_loss / len(train_loader),
            }, savefilename)

    except KeyboardInterrupt:
        print("Training interrupted. Saving current model state.")
        savefilename = './finetune_interrupted.tar'
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'train_loss': total_train_loss / len(train_loader),
        }, savefilename)

    print('Full finetune time = %.2f HR' % ((time.time() - start_full_time) / 3600))

if __name__ == '__main__':
    main()
