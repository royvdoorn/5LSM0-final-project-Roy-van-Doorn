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
from model import SegNet, Unet, Efficiency_model
import os
import numpy as np
import matplotlib.pyplot as plt
import utils
import torch.nn.utils.prune as prune

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default=".", help="Path to the data")
    """add more arguments here and change the default values to your needs in the run_container.sh file"""
    return parser

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        """
        m = torch.nn.Softmax(dim=1)
        prediction_soft = m(predictions)
        prediction_max = torch.nn.functional.gumbel_softmax(prediction_soft, tau=1, hard=True, dim=1)
        print(prediction_max.unique())
        print(np.shape(prediction_max))
        print(targets.unique())
        """
        # Ignore index 255
        mask = targets != 255
        targets = targets[mask]
        predictions = predictions.squeeze(1)
        predictions = predictions[mask]

        # Flatten label and prediction tensors
        predictions = predictions.view(-1)
        targets = targets/18
        targets = targets.view(-1)
        
        # Determine Dice loss
        intersection = (predictions * targets).sum()                       
        dice = (2.*intersection + self.smooth)/(predictions.sum() + targets.sum() + self.smooth)  
        return 1 - dice


def main(args):
    """define your model, trainingsloop optimitzer etc. here"""

    # define transform
    regular_transform = transforms.Compose([transforms.Resize((256, 256)),
                                            #transforms.RandomVerticalFlip(p=0.25),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # data loading
    path_local = "C:\\Users\\20192326\\Documents\\YEAR 1 AIES\\Neural networks for computer vision\\Assignment\\data"
    dataset = Cityscapes(args.data_path, split='train', mode='fine', target_type='semantic', transforms=regular_transform) #args.data_path
    validation_ratio = 0.1
    val_size = int(validation_ratio*len(dataset))
    train_size = len(dataset)-val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    batch_size = 25
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)#, num_worker=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)#, num_worker=8)

    # visualize example images
    '''
    for X, Y in train_dataset:
        transforms.ToPILImage()((X).byte()).save("C:/Users/20192326/Documents/YEAR 1 AIES/Neural networks for computer vision/Assignment/FinalAssignment/Example_input_image.png")
        transforms.ToPILImage()((Y*255).byte()).save("C:/Users/20192326/Documents/YEAR 1 AIES/Neural networks for computer vision/Assignment/FinalAssignment/Example_segmented_image.png")
        break
    '''

    # define model
    model = Unet()#.cuda()

    # define optimizer and loss function (don't forget to ignore class index 255)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)#, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 0.9)

    # training/validation loop
    epochs = 5

    train_loss = []
    val_loss = []
    for i in range(epochs):
        train_loss_epoch = 0
        val_loss_epoch = 0
        for X, Y in train_loader:
            target = (Y*255).long().squeeze(1)
            target = utils.map_id_to_train_id(target).to(device)
            optimizer.zero_grad()
            predictions = model(X).to(device)
            loss = criterion(predictions, target)
            loss.backward()
            optimizer.step()
            train_loss_epoch += loss
            print("Loss of batch at epoch " + str(i+1) + " : " + str(float(loss)))

        # Make a scheduler step at end of each epoch
        scheduler.step()

        # Determine validation loss
        for X, Y in val_loader:
            target = (Y*255).long().squeeze(1)
            target = utils.map_id_to_train_id(target)
            predictions = model(X)
            loss_val = criterion(predictions, target)
            val_loss_epoch += loss_val

        train_loss.append(float(train_loss_epoch)/train_size)
        val_loss.append(float(val_loss_epoch)/val_size)
        print("Average train loss of epoch " + str(i+1) + ": " + str(float(train_loss_epoch)/train_size))
        print("Average validation loss of epoch " + str(i+1) + ": " + str(float(val_loss_epoch)/val_size))

    # save model
    torch.save(model.state_dict(), 'Unet test submission')

    # visualize training data
    plt.plot(range(1, epochs+1), train_loss, color='r', label='train loss')
    plt.plot(range(1, epochs+1), val_loss, color='b', label='validation loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss of neural network")
    plt.legend()
    plt.savefig('Train performance of Unet single batch anti overfit')

    pass

def postprocess_dice(prediction, shape):
    prediction = prediction*18
    prediction = prediction.int()
    return prediction

def postprocess(prediction, shape):
    """Post process prediction to mask:
    Input is the prediction tensor provided by your model, the original image size.
    Output should be numpy array with size [x,y,n], where x,y are the original size of the image and n is the class label per pixel.
    We expect n to return the training id as class labels. training id 255 will be ignored during evaluation."""
    m = torch.nn.Softmax(dim=1)
    prediction_soft = m(prediction)
    prediction_max = torch.argmax(prediction_soft, axis=1)
    prediction = transforms.functional.resize(prediction_max, size=shape, interpolation=transforms.InterpolationMode.NEAREST)
    processed = prediction.cpu().detach().numpy()
    processed = processed.squeeze()
    return processed

def preprocess(img):
    """preproces image:
    input is a PIL image.
    Output image should be pytorch tensor that is compatible with your model"""

    regular_transform = transforms.Compose([transforms.Resize((256, 256)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    img = regular_transform(img)
    img = img.unsqueeze(0)
    return img

def visualize():
    model = SegNet()
    model.load_state_dict(torch.load("models\\Segnet model"))

    # define transform
    regular_transform = transforms.Compose([transforms.Resize((256, 256)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    path_local = "C:\\Users\\20192326\\Documents\\YEAR 1 AIES\\Neural networks for computer vision\\Assignment\\data"
    dataset = Cityscapes(path_local, split='train', mode='fine', target_type='semantic', transforms=regular_transform) #args.data_path

    batch_size = 1
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for X, Y in train_loader:
        #X = preprocess(X)
        prediction = model(X)
        processed = postprocess(prediction, shape=(1024, 2048))
        print(np.unique(processed))
        print(processed)
        plt.imshow(processed, cmap='tab20c')  # You can choose any colormap you prefer
        plt.title('Segmentation')
        plt.savefig("Images\\segmented image SegNet model.png")
        break

def prune_model():
    model = Model()
    model.load_state_dict(torch.load("models\\extended_u_net"))

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(params)

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, 'weight', 0.5)
            prune.remove(module, 'weight')

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(params)

    # save model
    torch.save(model.state_dict(), 'pruned_extended_u_net')    

    pass

if __name__ == "__main__":
    # Get the arguments
    parser = get_arg_parser()
    args = parser.parse_args()
    main(args)

    #visualize()
    
    #prune_model()

    #model = Model()
    #params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #print(params)
    
