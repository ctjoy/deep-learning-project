# -*- coding: utf-8 -*-
import argparse
import os
import subprocess
import numpy as np
import torch
from torch.autograd import Variable
import torchvision
from model import dcgan
from score.inception_score.inception_score import inception_score

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Get model setting
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epoch', type=int, default=0)
parser.add_argument('--model', type=str, default='dcgan')
parser.add_argument('--loss', type=str, default='hinge')
parser.add_argument('--dataset', type=str, default='cifar10')

args = parser.parse_args()

# Setting hyper parameter according to the paper
Z_dim = 128
num_samples = 5000

if args.model == 'cifar10':
    # CIFAR-10 dataset
    dataset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                           download=True,
                                           transform=torchvision.transforms.Compose([
                                               torchvision.transforms.Resize(32),
                                               torchvision.transforms.ToTensor(),
                                               torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
else:
    # STL-10 dataset
    dataset = torchvision.datasets.STL10(root='../data', train=True,
                                           download=True,
                                           transform=torchvision.transforms.Compose([
                                               torchvision.transforms.Resize(32),
                                               torchvision.transforms.ToTensor(),
                                               torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

# Data loader
loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=2)

# Get the trained model
# if args.model == 'resnet':
#     generator = model_resnet.Generator(Z_dim).to(device)
# else:
generator = dcgan.Generator(Z_dim).to(device)
generator.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, '{}_{}_{}_gen_{}'.format(args.model, args.loss, args.dataset, args.epoch))))

# Generate fake and real images
print("Sampling {} images...".format(num_samples))
num_batches = num_samples // args.batch_size + 1

eval_images = []
for batch_idx, (data, target) in enumerate(loader):

    if (batch_idx+1) == num_batches:
        break

    z = Variable(torch.randn(args.batch_size, Z_dim)).to(device)
    samples = generator(z).cpu().data
    eval_images.append(samples)

    if not os.path.exists('out/{}_{}_{}/fake/'.format(args.model, args.loss, args.dataset)):
        os.makedirs('out/{}_{}_{}/fake/'.format(args.model, args.loss, args.dataset))

    if not os.path.exists('out/{}_{}_{}/real/'.format(args.model, args.loss, args.dataset)):
        os.makedirs('out/{}_{}_{}/real/'.format(args.model, args.loss, args.dataset))

    torchvision.utils.save_image(samples, 'out/{}_{}_{}/fake/{}.png'.format(args.model, args.loss, args.dataset, str(batch_idx).zfill(5)), normalize=True)
    torchvision.utils.save_image(data, 'out/{}_{}_{}/real/{}.png'.format(args.model, args.loss, args.dataset, str(batch_idx).zfill(3)), normalize=True)

eval_images = np.vstack(eval_images)
eval_images = eval_images[:num_samples]
eval_images = list(eval_images)

# Calc Inception score
print("Calculating Inception Score...")
use_cuda = True if torch.cuda.is_available() else False
inception_score_mean, inception_score_std = inception_score(eval_images, cuda=use_cuda, batch_size=args.batch_size, resize=True)
print('Inception Score: Mean = {:.2f} \tStd = {:.2f}'.format(inception_score_mean, inception_score_std))

# Calc FID score
print("Calculating FID Score...")
fid_score_file = 'score/fid_score/fid_score.py'
real_image_path = 'out/{}_{}_{}/real/'.format(args.model, args.loss, args.dataset)
fake_image_path = 'out/{}_{}_{}/fake/'.format(args.model, args.loss, args.dataset)
subprocess.run([fid_score_file, real_image_path, fake_image_path])
