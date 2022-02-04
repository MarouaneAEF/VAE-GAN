
from functools import total_ordering
import torch 
from torch import binary_cross_entropy_with_logits, device, optim 
from torch.autograd import Variable
import torch.nn as nn 
import torch.nn.functional as F
import torchvision.utils as vutils 
import torch 


import numpy as np 

from models import Generator, Encoder, Discriminator
from utils.utils import weights_init_normal, weights_init


class Trainer:
    def __init__(self,
                data,
                generator = Generator,
                encoder = Encoder,
                discriminator=Discriminator,
                batch_size=8,
                epochs=301,
                lr_e=2e-4,
                lr_g=2e-4,
                lr_d=2e-4,
                n_samples=64,
                z_dim=200,
                img_size=64,
                device="cuda"
                ):

        self.generator = generator
        self.encoder = encoder
        self.discriminator = discriminator
        self.train_loader = data

        self.batch_size = batch_size
        self.epochs = epochs
        self.lr_e = lr_e
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.n_samples = n_samples
        self.z_dim = z_dim
        self.img_size = img_size
        self.device = device


    def train(self, epoch):

        E = self.encoder().to(self.device)
        G= self.generator().to(self.device)
        D = self.discriminator().to(self.device)

        E.train()
        G.train()
        D.train()

        E.apply(weights_init)
        G.apply(weights_init)
        D.apply(weights_init)

        optimizer_e = optim.Adam(E.parameters(), self.lr_e)
        optimizer_g = optim.Adam(G.parameters(), self.lr_g)
        optimizer_d = optim.Adam(D.parameters(), self.lr_d)
        
        i=0
        enc_loss = 0
        dec_loss = 0
        dis_loss = 0
        prior_loss = 0 

        for x, _ in self.train_loader:
            
            #  true and fake labels 
            y_true = Variable(torch.ones(self.batch_size)).to(self.device)
            y_fake = Variable(torch.zeros(self.batch_size)).to(self.device)

            # 1)  real data : random mini-batch from dataset
            x_true = x.float().to(self.device)
            # print(f"x_true size : {x_true.size()}")
            # sampling latent space corresponding to real data
            mu_z, logvar_z ,z_true,  = E(x_true)
            # print(f"z_true : {z_true.size()}")
            # KL divergence of the prio
            loss_prior =  -0.5 * torch.sum(1 + logvar_z - mu_z.pow(2) - logvar_z.exp())
            #  sampling image corresponding to real latent variable 
            x_gen = G(z_true)
            # print(f"x_gen size : {x_gen.size()}")
            # compute D(x) for real and fake image 
            # print(f"x_true size: {x_true.size()}")
            # print(f"x_gen size: {x_gen.size()}")
            # print(f"x_prior size: {x_prior.size()}")
            ld_r, fd_r = D(x_true) # ----------------Disl(x_real)
            ld_f, fd_f = D(x_gen)  # ----------------Discl(D(z)) 
            
            
            # 5 line of algo 
            loss_Disl = F.mse_loss(fd_r , fd_f) 
            # loss_pixel = F.mse_loss(x_true, x_gen )
            
            # print(f"ld_r:{ld_r.size()}")
            # print(f"ld_f:{ld_f.size()}")
            # print(f"ld_p:{ld_p.size()}")
            # print(f"fd_r:{fd_r.size()}")
            # print(f"fd_f:{fd_f.size()}")
            # print(f"fd_p:{fd_p.size()}")

            # ------------------------------------
            # sampling latent z_p 
            z_prior = Variable(torch.randn(256, self.z_dim)).to(self.device)
            # sampling image corresponding to z_p 
            # print(f"z_prior size : {z_prior.size()}")
            x_prior = G(z_prior)
            ld_p, fd_p = D(x_prior)
            # loss_dis_prior = F.binary_cross_entropy(ld_p, y_fake)
            # GAN loss (8th line of the algo) 
            loss_gan = F.binary_cross_entropy(ld_r, y_true) + 0.5*(F.binary_cross_entropy(ld_f, y_fake) + F.binary_cross_entropy(ld_p, y_fake))
            
            

            loss_enc = loss_Disl + loss_prior#loss_prior + loss_Disl
            loss_dec = 0.01*F.binary_cross_entropy(ld_p, y_true) + loss_gan 
            # loss_dis = loss_gan 
            total_loss = loss_enc + loss_dec +loss_gan

            optimizer_d.zero_grad()
            loss_gan.backward(retain_graph=True)
            optimizer_e.zero_grad()
            loss_enc.backward(retain_graph=True)
            optimizer_g.zero_grad()
            loss_dec.backward()
            # total_loss.backward()
            optimizer_d.step()
            optimizer_e.step()
            optimizer_g.step()
            

            
        

            enc_loss += loss_enc.item()
            dec_loss += loss_dec.item()
            dis_loss += loss_gan.item()
            prior_loss += loss_prior.item()
            
            if i%50 == 0:
                status = f"Epoch:{epoch}/Iter:{i}, Total Loss: {total_loss.item()/len(self.train_loader):>5f}, enc_loss : {enc_loss/len(self.train_loader):>5f}, dec_loss: {dec_loss/len(self.train_loader):>5f}, dis_loss: {loss_gan/len(self.train_loader):>5f}, LK_D: {loss_prior/len(self.train_loader):>5f}" 
                print(status)
            if i%100 == 0:
                vutils.save_image(x_true.data[:16], f"./reconstruction/true/{i}_{epoch}.png")
                vutils.save_image(x_gen.data[:16], f"./reconstruction/fake/{i}_{epoch}.png")
              
                
            if epoch % 100 == 0:
                vutils.save_image(torch.cat([x_true.data[:16], x_gen.data[:16]]), f'./reconstruction/comparison/{epoch}.png')
                torch.save(E.state_dict(), f"./models/encoder/encoder_{epoch}.pth")
                torch.save(G.state_dict(), f"./models/generator/generator_{epoch}.pth")
                torch.save(D.state_dict(), f"./models/discriminator/discriminator_{epoch}.pth")
            i+=1 
            if i == 389:
                break



