"""
This file needs to contain the main training loop. The training code should be encapsulated in a main() function to
avoid any global variables.
"""
from torchvision.datasets import Cityscapes
import torchvision.transforms.v2 as transforms
from argparse import ArgumentParser
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch
from model import Model, CNN_autoencoder
import os
import numpy as np
import matplotlib.pyplot as plt


def get_arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default=".", help="Path to the data")
    """add more arguments here and change the default values to your needs in the run_container.sh file"""
    return parser

class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        # Ignore index 255
        targets[targets == 255] = torch.nan

        # Flatten label and prediction tensors
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # Determine Dice loss
        intersection = (predictions * targets).sum()                       
        dice = (2.*intersection + self.smooth)/(predictions.sum() + targets.sum() + self.smooth)  
        return 1 - dice


def main(args):
    """define your model, trainingsloop optimitzer etc. here"""

    # define transform
    regular_transform = transforms.Compose([transforms.Resize((1024, 512)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # data loading
    #path_local = "C:\\Users\\20192326\\Documents\\YEAR 1 AIES\\Neural networks for computer vision\\Assignment\\data"
    dataset = Cityscapes(args.data_path, split='train', mode='fine', target_type='semantic', transforms=regular_transform) #args.data_path
    validation_ratio = 0.1
    val_size = int(validation_ratio*len(dataset))
    train_size = len(dataset)-val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    batch_size = 25
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # visualize example images

    # define model
    model = Model()#.cuda()

    # define optimizer and loss function (don't forget to ignore class index 255)
    criterion = DiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 0.9)

    # training/validation loop
    epochs = 25

    train_loss = []
    val_loss = []
    for i in range(epochs):
        train_loss_epoch = 0
        val_loss_epoch = 0
        for X, Y in train_loader:
            optimizer.zero_grad()
            predictions = model(X)
            loss = criterion(predictions, Y)
            loss.backward()
            optimizer.step()
            train_loss_epoch += loss
            print("Loss of batch at epoch " + str(i+1) + " : " + str(float(loss)))

        # Make a scheduler step at end of each epoch
        scheduler.step()

        # Determine validation loss
        for X, Y in val_loader:
            predictions = model(X)
            loss_val = criterion(predictions, Y)
            val_loss_epoch += loss_val

        train_loss.append(float(train_loss_epoch)/train_size)
        val_loss.append(float(val_loss_epoch)/val_size)
        print("Average train loss of epoch " + str(i+1) + ": " + str(float(train_loss_epoch)/train_size))
        print("Average validation loss of epoch " + str(i+1) + ": " + str(float(val_loss_epoch)/val_size))

    # save model
    torch.save(model.state_dict(), 'basic_model')

    # visualize some results
    plt.plot(range(1, epochs+1), train_loss, color='r', label='train loss')
    plt.plot(range(1, epochs+1), val_loss, color='b', label='validation loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss of neural network")
    plt.legend()
    plt.savefig('Train performance basic model')

    pass


if __name__ == "__main__":
    # Get the arguments
    parser = get_arg_parser()
    args = parser.parse_args()
    main(args)