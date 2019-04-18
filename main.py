# -*- coding: utf-8 -*-
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.autograd import Variable
# import model_resnet
from model import dcgan

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Hyper parameters
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_epochs', type=int, default=200)
parser.add_argument('--loss', type=str, default='hinge')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
parser.add_argument('--model', type=str, default='resnet')

args = parser.parse_args()

# CIFAR-10 dataset
dataset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                       download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.Resize(32),
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))


# Data loader
loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=2)

# Setting hyper parameter according to the paper
Z_dim = 128
adam_alpha = 0.0002
adam_beta1 = 0.0
adam_beta2 = 0.9
n_dis = 5 # Number of updates to discriminator for every update to generator
# step = 100000

# Model
# if args.model == 'resnet':
#     discriminator = model_resnet.Discriminator().to(device)
#     generator = model_resnet.Generator(Z_dim).to(device)
# else:
discriminator = dcgan.Discriminator().to(device)
generator = dcgan.Generator(Z_dim).to(device)

# Optimizer
optim_disc = optim.Adam(discriminator.parameters(), lr=adam_alpha, betas=(adam_beta1,adam_beta2))
optim_gen  = optim.Adam(generator.parameters(), lr=adam_alpha, betas=(adam_beta1,adam_beta2))

# Loss function
def discriminator_loss(d_real, d_fake):
    if args.loss == 'hinge':
        real_loss = nn.ReLU()(1.0 - d_real).mean()
        fake_loss = nn.ReLU()(1.0 + d_fake).mean()

    elif args.loss == 'wasserstein':
        real_loss = -d_real.mean()
        fake_loss = d_fake.mean()

    else:
        real_label = Variable(torch.ones(args.batch_size, 1).to(device))
        fake_label = Variable(torch.zeros(args.batch_size, 1).to(device))

        real_loss = nn.BCEWithLogitsLoss()(d_real, real_label)
        fake_loss = nn.BCEWithLogitsLoss()(d_fake, fake_label)

    return real_loss + fake_loss

def generator_loss(d_fake):
    if args.loss == 'hinge' or args.loss == 'wasserstein':
        return -d_fake.mean()
    else:
        real_label = Variable(torch.ones(args.batch_size, 1).to(device))
        return nn.BCEWithLogitsLoss()(d_fake, real_label)

# Training function
total_step = len(loader)
fixed_z = Variable(torch.randn(args.batch_size, Z_dim)).to(device)
def train(epoch):

    if not os.path.exists('out/'):
        os.makedirs('out/')

    for batch_idx, (data, target) in enumerate(loader):
        if data.size()[0] != args.batch_size:
            continue
        data = data.to(device)
        target = target.to(device)

        # Update discriminator
        for _ in range(n_dis):
            z = Variable(torch.randn(args.batch_size, Z_dim)).to(device)
            optim_disc.zero_grad()

            d_real = discriminator(data)
            d_fake = discriminator(generator(z))

            loss_disc = discriminator_loss(d_real, d_fake)
            loss_disc.backward()

            optim_disc.step()

        z = Variable(torch.randn(args.batch_size, Z_dim)).to(device)

        # Update generator
        optim_gen.zero_grad()

        d_fake = discriminator(generator(z))

        loss_gen = generator_loss(d_fake)
        loss_gen.backward()

        optim_gen.step()

        if (batch_idx+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Disc Loss: {:.4f}, Gen Loss: {:.4f}'
                  .format(epoch+1, args.num_epochs, batch_idx+1, total_step, loss_disc.item(), loss_gen.item()))

        if batch_idx == 0:
            # Save the picture
            torchvision.utils.save_image(data, 'out/real_epoch_{}.png'.format(str(epoch).zfill(3)), normalize=True)

            samples = generator(fixed_z).cpu().data
            torchvision.utils.save_image(samples, 'out/fake_epoch_{}.png'.format(str(epoch).zfill(3)), normalize=True)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

# Train and evaluate in every epoch
for epoch in range(args.num_epochs):
    train(epoch)
    torch.save(discriminator.state_dict(), os.path.join(args.checkpoint_dir, 'disc_{}'.format(epoch)))
    torch.save(generator.state_dict(), os.path.join(args.checkpoint_dir, 'gen_{}'.format(epoch)))
