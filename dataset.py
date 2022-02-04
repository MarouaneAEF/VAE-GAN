import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import random
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import torch 


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

def get_data_loader(batch_size=8, img_size=16):
    my_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize
    ])
    train_dataset = datasets.ImageFolder(root = 'data/images', transform=my_transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader 


def cifar10loader(batch_size=64):
    transform = transforms.Compose([transforms.Resize(64), #3*32*32
                                    transforms.ToTensor()
                        ])

    train_loader = torch.utils.data.DataLoader(
			datasets.CIFAR10(
				root="./data", train=True, download=True, transform=transform),
				batch_size=batch_size,
				shuffle=True)

    test_loader  = torch.utils.data.DataLoader(
					datasets.CIFAR10(
				root="./data", train=False, download=True, transform=transform),
				batch_size=batch_size,
				shuffle=True)
	
    return train_loader, test_loader 


def mnistloader(batch_size):
	train_loader = torch.utils.data.DataLoader(
		datasets.MNIST(
			'./data', train=True, download=True,
			transform=transforms.ToTensor()
		),
		batch_size=batch_size,
		shuffle=True
	)

	test_loader = torch.utils.data.DataLoader(
		datasets.MNIST(
			'./data', train=False, download=True,
			transform=transforms.ToTensor()
		),
		batch_size=batch_size,
		shuffle=True
	)

	return train_loader, test_loader