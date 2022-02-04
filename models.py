
from cmath import tanh
import torch 
import torch.nn as nn 
import torch.nn.functional as F 


#  img_size must be at the same value in the Encoder and in the Discriminator !!
class Encoder(nn.Module):
    def __init__(self, img_size=32, input_channel=3, latent_dim=200, training=True):
        super().__init__()
        self.input_channel = input_channel
        self.latent_dim = latent_dim
        self.training = training
        self.img_size = img_size

        self.encConvStack = nn.Sequential(
            nn.Conv2d(self.input_channel, 64, 5, padding=2, stride=2),
            nn.BatchNorm2d(64,momentum=0.9),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv2d(64, 128, 5, padding=2, stride=2),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv2d(128, 256,5 , padding=2,stride=2),
            nn.BatchNorm2d(256,momentum=0.9),
            nn.LeakyReLU(0.2, inplace=False),
        )

        x = torch.rand(self.input_channel, self.img_size, self.img_size).view(-1, self.input_channel, self.img_size, self.img_size)
        self._after_bottlneck = None
        self._before_bottlneck = None
        self.before_bottlneck(x)

        self.fcStack = nn.Sequential(
            nn.Linear(self._after_bottlneck, 2048),
            nn.BatchNorm1d(2048,momentum=0.9),
            nn.LeakyReLU(0.2, inplace=False),
        )

        self.mean = nn.Linear(2048, self.latent_dim)
        self.log_var = nn.Linear(2048, self.latent_dim)

    def before_bottlneck(self, x):

        x = self.encConvStack(x) 
        if self._before_bottlneck is None:
            self._before_bottlneck = x[0].shape
        if self._after_bottlneck is None:
            self._after_bottlneck = x[0].flatten().shape[0] 

        return x 

    def reparameterize(self, mu, logvar):
        '''
        compute z = mu + std * epsilon
        '''
        if self.training:
            # do this only while training
            # compute the standard deviation from logvar
            std = torch.exp(0.5 * logvar)
            # sample epsilon from a normal distribution with mean 0 and
            # variance 1
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x): 
        x = self.before_bottlneck(x)
        x = x.view(-1, self._after_bottlneck)
        x = self.fcStack(x)

        mu_z = self.mean(x)
        logvar_z = self.log_var(x)
        z_sample = self.reparameterize(mu_z, logvar_z)

        return mu_z, logvar_z, z_sample 

class Generator(nn.Module):
    def __init__(self, z_dim=200, output_channel=3):
        super().__init__()
        self.z_dim = z_dim
        self.output_channel = output_channel

       
        # self.fcStack = nn.Sequential(
        #     nn.Linear(self.z_dim, 256*8*8),
        #     nn.BatchNorm1d(256*8*8),
        #     nn.ReLU(inplace=True),


        # )

        self.convs = nn.Sequential(
            nn.ConvTranspose2d(self.z_dim, 512, 4, 1, 0, bias=False),
            nn.ReLU(inplace=False),
            # nn.ConvTranspose2d(512, 384, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(384),
            # nn.ReLU(inplace=True),
            # nn.ConvTranspose2d(384, 192, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(192),
            # nn.ReLU(inplace=True),
            # nn.ConvTranspose2d(192, 96, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(96),
            # nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=False),
            nn.ConvTranspose2d(64, 3, 4, 2, 0, bias=False),
            nn.Tanh()
        )

        # self.convs = nn.Sequential(
        #     # nn.ConvTranspose2d(256, 256, 5, 1, 0, bias=False),
        #     nn.ConvTranspose2d(self.z_dim, 256, 5, 1, 0, bias=False),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     nn.ConvTranspose2d(256, 128, 5, 2, 1, bias=False),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
            
        #     nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.ConvTranspose2d(64, self.output_channel, 4, 2, 1, bias=False),
          
        #     nn.Tanh()
        # )

        
    
    def forward(self,z):
        # x = self.fcStack(x)
        # x = x.view(-1, 256, 8, 8)
        z = z.view(z.size(0), z.size(1), 1, 1)
        # x_gen = self.deconvStack(z)
        x_gen = self.convs(z)
        return x_gen


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(32, 128, 5, 2, 2, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(128, 256, 5, 2, 2, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(256, 256, 5, 2, 2, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv2d(256, 512, 5, 2, 2, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=False),
            
        )

        self.fcStack = nn.Sequential(
            nn.Linear(8*8*256,512),
            nn.BatchNorm1d(512,momentum=0.9),
            nn.Linear(512,1),
            nn.Sigmoid()

        )

    def forward(self, x):

        x = self.convs(x)

        x = x.view(-1, 256 * 8 * 8)

        out_2 = x
        # print(f"---out_2 size : {out_2.size()}")

        x = self.fcStack(x)
        
        return x.squeeze(), out_2.squeeze()



