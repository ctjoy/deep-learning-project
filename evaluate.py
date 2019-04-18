# -*- coding: utf-8 -*-
import argparse
import os
import numpy as np
import torch
from torch.autograd import Variable
import torchvision
from model import dcgan
from score.inception_score import inception_score

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# FID_SCORE_BLOCK_INDEX_BY_DIM = {
#     64: 0,   # First max pooling features
#     192: 1,  # Second max pooling featurs
#     768: 2,  # Pre-aux classifier features
#     2048: 3  # Final average pooling features
# }

# Get model setting
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epoch', type=int, default=0)
parser.add_argument('--model', type=str, default='dcgan')
# parser.add_argument('--fid_sroce_feature_dims', type=int, default=2048,
#                     choices=list(FID_SCORE_BLOCK_INDEX_BY_DIM),
#                     help=('Dimensionality of Inception features to use. '
#                           'By default, uses pool3 features'))

args = parser.parse_args()

# Setting hyper parameter according to the paper
Z_dim = 128
num_samples = 5000

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

# Get the trained model
# if args.model == 'resnet':
#     generator = model_resnet.Generator(Z_dim).to(device)
# else:
generator = dcgan.Generator(Z_dim).to(device)
generator.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, 'gen_{}'.format(args.epoch))))

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

    if not os.path.exists('fake/'):
        os.makedirs('fake/')

    if not os.path.exists('real/'):
        os.makedirs('real/')

    torchvision.utils.save_image(samples, 'fake/{}.png'.format(str(args.batch_size).zfill(5)), normalize=True)
    torchvision.utils.save_image(data, 'real/{}.png'.format(str(0).zfill(3)), normalize=True)

eval_images = np.vstack(eval_images)
eval_images = eval_images[:num_samples]
eval_images = np.clip((eval_images + 1.0) * 127.5, 0.0, 255.0).astype(np.uint8)
eval_images = list(eval_images)

# Calc Inception score
print("Calculating Inception Score...")
use_cuda = True if torch.cuda.is_available() else False
inception_score_mean, inception_score_std = inception_score(samples, cuda=use_cuda, batch_size=args.batch_size, resize=True)
print('Inception Score: Mean = {:.2f} \tStd = {:.2f}'.format(inception_score_mean, inception_score_std))

print("Calculating FID Score...")
command = './score/pytorch_fid/fid_score.py real/ fake/'
os.system(command)
# fid_score = calculate_fid_given_paths(('real/', 'fake/'), args.batch_size, device, args.fid_sroce_feature_dims)
# print('FID Score: {:.2f}'.format(fid_score))
