"""
This file needs to contain the main training loop. The training code should be encapsulated in a main() function to
avoid any global variables.
"""
from model import Model
from torchvision.datasets import Cityscapes
import torchvision.transforms as transforms
from argparse import ArgumentParser
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch as torch


def get_arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default=".", help="Path to the data")
    """add more arguments here and change the default values to your needs in the run_container.sh file"""
    return parser


def main(args):
    """define your model, trainingsloop optimitzer etc. here"""

    # define transform
    regular_transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
                                            ])

    # data loading
    dataset = Cityscapes(args.data_path, split='train', mode='fine', target_type='semantic', transform=regular_transform)
    validation_ratio = 0.2
    val_size = int(validation_ratio*len(dataset))
    train_size = len(dataset)-val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    batch_size = 10
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # visualize example images

    # define model
    model = Model().cuda()

    # define optimizer and loss function (don't forget to ignore class index 255)
    # Still need to ignore index 255 and make own Dice or Jacard loss
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 0.9)

    # training/validation loop
    epochs = 5

    for i in range(epochs):
        train_loss_epoch = 0
        val_loss_epoch = 0
        for X, Y in train_loader:
            optimizer.zero_grad()
            predictions = model(X)
            loss = criterion(predictions, X)
            loss.backward()
            optimizer.step()
            train_loss_epoch += loss

        # Make a scheduler step at end of each epoch
        scheduler.step()

        # Determine validation loss
        for X, Y in val_loader:
            predictions = model(X)
            loss_val = criterion(predictions, X)
            val_loss_epoch += loss_val

        print("Average train loss of epoch" + str(i) + ": " + str(train_loss_epoch/train_size))
        print("Average validation loss of epoch" + str(i) + ": " + str(val_loss_epoch/train_size))

    # save model


    # visualize some results

    pass


if __name__ == "__main__":
    # Get the arguments
    #parser = get_arg_parser()
    #args = parser.parse_args()
    #main(args)

