import torch
import torch.nn as nn
import torch.nn.functional as function

""" segmentation model example
"""

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        """ Encoder """
        self.e1 = encoder_block(3, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)

        """ Bottleneck """
        self.b = conv_block(512, 1024)

        """ Decoder """
        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)

        """ Classifier """
        self.outputs = nn.Conv2d(64, 34, kernel_size=1, padding=0)

    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        """ Bottleneck """
        b = self.b(p4)

        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        """ Segmentation output """
        outputs = self.outputs(d4)

        return outputs


class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

""" Encoder block:
    It consists of an conv_block followed by a max pooling.
    Here the number of filters doubles and the height and width half after every block.
"""
class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p

""" Decoder block:
    The decoder block begins with a transpose convolution, followed by a concatenation with the skip
    connection from the encoder block. Next comes the conv_block.
    Here the number filters decreases by half and the height and width doubles.
"""
class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)

        return x


class CNN_autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1a = nn.Conv2d(3, 64, 3, 1, padding='same')
        self.norm_1 = nn.BatchNorm2d(64)
        self.conv_1b = nn.Conv2d(64, 64, 3, 1, padding='same')
        self.conv_2a = nn.Conv2d(64, 128, 3, 1, padding='same')
        self.norm_2 = nn.BatchNorm2d(128)
        self.conv_2b = nn.Conv2d(128, 128, 3, 1, padding='same')
        self.pool = nn.MaxPool2d(2, 2)
        self.conv_3 = nn.Conv2d(128, 256, 3, 1, padding='same')

        self.conv_3a = nn.ConvTranspose2d(256, 128, 3, 1, padding=1)
        self.conv_3b = nn.ConvTranspose2d(128, 128, 5, 1, padding=1) #1
        self.conv_3c = nn.ConvTranspose2d(128, 64, 3, 2, padding=1, output_padding=1) #2
        self.conv_4a = nn.ConvTranspose2d(64, 64, 5, 1, padding=3) #3
        self.conv_4b = nn.ConvTranspose2d(64, 34, 3, 2, padding=3, output_padding=1) #1
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.norm_1(function.relu(self.conv_1a(x)))
        x1 = self.dropout(self.pool(self.norm_1(function.relu(self.conv_1b(x)))))

        x = self.norm_2(function.relu(self.conv_2a(x1)))
        x2 = self.dropout(self.pool(self.norm_2(function.relu(self.conv_2b(x)))))

        x = function.relu(self.conv_3(x2))
        x = function.relu(self.conv_3a(x))

        x = torch.cat([x, x2], axis=1)
        x = self.norm_2(function.relu(self.conv_3b(x)))
        x = self.dropout(self.norm_1(function.relu(self.conv_3c(x))))

        x = torch.cat([x, x1], axis=1)
        x = self.norm_1(function.relu(self.conv_4a(x)))
        x = self.dropout(function.sigmoid(self.conv_4b(x)))
        return x