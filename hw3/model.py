import torch.nn as nn


class Generator(nn.Module):
    # this generator is a simplified version of DCGAN to speed up the training

    def __init__(self):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # input size 100 x 1 x 1
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
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

    def forward(self, z):
        if z.shape[-1] != 1:
            # change the shape from (batch_size, 100) to (batch_size, 100, 1, 1)
            z = z[..., None, None]

        output = self.net(z)
        return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        sizes = [128,256,512]
        
        self.net = nn.Sequential(
            
            nn.Conv2d(1,sizes[0], 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(sizes[0], sizes[1], 4, 2, 1, bias=False),
            nn.BatchNorm2d(sizes[1]),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(sizes[1], sizes[2], 4, 2, 1, bias=False),
            nn.BatchNorm2d(sizes[2]),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Flatten(),
            nn.Linear(sizes[2]*16,1),
            nn.Sigmoid(),
            
            
            # nn.Conv2d(1,128,4,2,1,bias=False),
            # nn.LeakyReLU(0.2,inplace=True),
            # nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(256),
            # nn.LeakyReLU(0.2,inplace=True),
            # nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(512),
            # nn.LeakyReLU(0.2,inplace=True),
            # #Shape (batch_sizex 512 x 4 x 4)
            # nn.Flatten(),
            # nn.Linear(512*4*4,1),
            # nn.Sigmoid(),
        )

    def forward(self, img):
        return self.net(img).squeeze(-1)