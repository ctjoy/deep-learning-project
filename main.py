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
from evaluation.inception_score import inception_score
from evaluation.fid_score import calculate_fid_given_paths

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

FID_SCORE_BLOCK_INDEX_BY_DIM = {
    64: 0,   # First max pooling features
    192: 1,  # Second max pooling featurs
    768: 2,  # Pre-aux classifier features
    2048: 3  # Final average pooling features
}

# Hyper parameters
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_epochs', type=int, default=200)
parser.add_argument('--loss', type=str, default='hinge')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
parser.add_argument('--model', type=str, default='resnet')
parser.add_argument('--fid_sroce_feature_dims', type=int, default=2048,
                    choices=list(FID_SCORE_BLOCK_INDEX_BY_DIM),
                    help=('Dimensionality of Inception features to use. '
                          'By default, uses pool3 features'))

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
d_iters = 5 # Number of updates to discriminator for every update to generator

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

    if not os.path.exists('out/real/') or os.path.exists('out/fake/'):
        os.makedirs('out/real/')
        os.makedirs('out/fake/')

    for batch_idx, (data, target) in enumerate(loader):
        if data.size()[0] != args.batch_size:
            continue
        data = data.to(device)
        target = target.to(device)

        # Update discriminator
        for _ in range(d_iters):
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

            # Save the picture
            torchvision.utils.save_image(data, 'out/real/epoch_{}_batch_{}.png'.format(str(epoch).zfill(3), batch_idx+1), normalize=True)

            samples = generator(fixed_z).cpu().data
            torchvision.utils.save_image(samples, 'out/fake/epoch_{}_batch_{}.png'.format(str(epoch).zfill(3), batch_idx+1), normalize=True)

            # Calulate the inception score and FID score
            use_cuda = True if torch.cuda.is_available() else False
            inception_score_mean, inception_score_std = inception_score(samples, cuda=use_cuda, batch_size=32, resize=True, splits=10)
            print('Inception Score: {:.2f}±{:.2f}'.format(inception_score_mean, inception_score_std))

            fid_score = calculate_fid_given_paths(('out/real', 'out/fake'), args.batch_size, device, args.fid_sroce_feature_dims)
            print('FID Score: {:.2f}'.format(fid_score))

os.makedirs(args.checkpoint_dir, exist_ok=True)

# Train and evaluate in every epoch
for epoch in range(args.num_epochs):
    train(epoch)
    torch.save(discriminator.state_dict(), os.path.join(args.checkpoint_dir, 'disc_{}'.format(epoch)))
    torch.save(generator.state_dict(), os.path.join(args.checkpoint_dir, 'gen_{}'.format(epoch)))
