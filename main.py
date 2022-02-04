import torch 
from train import Trainer
from models import Discriminator, Encoder, Generator
from dataset import get_data_loader, cifar10loader

device = torch.device("cuda")

# dataloader= get_data_loader()
trainloader, testloader  = cifar10loader(batch_size=64)
vae_gan = Trainer(trainloader, Generator, Encoder, Discriminator)

epochs = 2
for epoch in range(1, epochs + 1):

    vae_gan.train(epoch)
    # vae_gan.test(epoch)
    