import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    # this generator is a simplified version of DCGAN to speed up the training

    def __init__(self):
        super(Generator, self).__init__()
        # todo implement your architecture here
        self.net = nn.Sequential(
            # input size 110 x 1 x 1
#             nn.ConvTranspose2d(110, 1024, 4, 1, 0, bias=False),
#             nn.BatchNorm2d(1024),
#             nn.ReLU(),
            
#             nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
            
            nn.ConvTranspose2d(110, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            # size 512 x 4 x 4
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # state size. 256 x 8 x 8
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # size 128 x 16 x 16
            nn.ConvTranspose2d(128, 1, 4, 2, 1, bias=False),
            nn.Tanh(),
            # output size 1 x 32 x 32
        )


    def forward(self, z, y_class):
        # z shape is batchsize x 100
        y_class = F.one_hot(y_class, num_classes=10).float()  # convert the input number to one-hot tensor. shape: batchsize x 10
        z = torch.cat((z, y_class), dim=1)
        # todo implement the forward function. It should generate a batchsize x 1 x 32 x 32 image
        if z.shape[-1] != 1:
            # change the shape from (batch_size, 100) to (batch_size, 100, 1, 1)
            z = z[..., None, None]
            
        output = self.net(z)
        return output




class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # todo implement your architecture here
        sizes = [128,256,512]
        
        self.net1 = nn.Sequential(   
            nn.Conv2d(1,sizes[0], 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(sizes[0], sizes[1], 4, 2, 1, bias=False),
            nn.BatchNorm2d(sizes[1]),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(sizes[1], sizes[2], 4, 2, 1, bias=False),
            nn.BatchNorm2d(sizes[2]),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Flatten(),
        )
        
        self.net2 = nn.Sequential(
            nn.Linear(sizes[2]*16+10,512),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(512,64),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(64,1),
            nn.Sigmoid(),
        )
        
#         self.net = nn.Sequential(
            
#             nn.Conv2d(1,sizes[0], 4, 2, 1, bias=False),
#             nn.LeakyReLU(0.2,inplace=True),
#             nn.Conv2d(sizes[0], sizes[1], 4, 2, 1, bias=False),
#             nn.BatchNorm2d(sizes[1]),
#             nn.LeakyReLU(0.2,inplace=True),
#             nn.Conv2d(sizes[1], sizes[2], 4, 2, 1, bias=False),
#             nn.BatchNorm2d(sizes[2]),
#             nn.LeakyReLU(0.2,inplace=True),
#             nn.Flatten(),
#             nn.Linear(sizes[2]*16,1),
#             nn.Sigmoid(),
#         )

    def forward(self, img, y_class):
        # img shape is batchsize x 1 x 32 x 32
        y_class = F.one_hot(y_class, num_classes=10).float()  # convert the input number to one-hot tensor. shape: batchsize x 10
        # bsz = y_class.size(0)
        # y_pad = F.pad(y_class, pad=(0, 32*32-10), mode='constant', value=0).view(bsz,1,32,32)
        # z = torch.cat((img,y_pad), dim=-1)
        # z = self.net(z).squeeze(-1)
        
        
        y = self.net1(img)
        # print(y.size())
        y = torch.cat((y, y_class), dim=1)
        
        z = self.net2(y).squeeze(-1)
        
        return z

        
        