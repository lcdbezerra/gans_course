import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    # this generator is a simplified version of DCGAN to speed up the training

    def __init__(self):
        super(Generator, self).__init__()
        # todo implement your architecture here
        NotImplementedError


    def forward(self, z, y_class):
        # z shape is batchsize x 100
        y_class = F.one_hot(y_class, num_classes=10).float()  # convert the input number to one-hot tensor. shape: batchsize x 10

        # todo implement the forward function. It should generate a batchsize x 1 x 32 x 32 image
        NotImplementedError




class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # todo implement your architecture here

    def forward(self, img, y_class):
        # img shape is batchsize x 1 x 32 x 32
        y_class = F.one_hot(y_class, num_classes=10).float()  # convert the input number to one-hot tensor. shape: batchsize x 10

        # todo implement the forward function. It should generate a batchsize x 1 tensor
        NotImplementedError

