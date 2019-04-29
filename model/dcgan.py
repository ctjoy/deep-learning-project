# DCGAN-like generator and discriminator
from torch import nn
from torch.nn.utils import spectral_norm

channels = 3
leak = 0.1
w_g = 4

class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.z_dim = z_dim

        self.model = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(z_dim, 512, 4, stride=1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(leak, inplace=True),
            spectral_norm(nn.ConvTranspose2d(512, 256, 4, stride=2, padding=(1,1))),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(leak, inplace=True),
            spectral_norm(nn.ConvTranspose2d(256, 128, 4, stride=2, padding=(1,1))),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(leak, inplace=True),
            spectral_norm(nn.ConvTranspose2d(128, 64, 4, stride=2, padding=(1,1))),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(leak, inplace=True),
            spectral_norm(nn.ConvTranspose2d(64, channels, 3, stride=1, padding=(1,1))),
            nn.Tanh())

    def forward(self, z):
        return self.model(z.view(-1, self.z_dim, 1, 1))

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(

            spectral_norm(nn.Conv2d(channels, 64, 3, stride=1, padding=(1,1))),
            nn.LeakyReLU(leak, inplace=True),
            spectral_norm(nn.Conv2d(64, 64, 4, stride=2, padding=(1,1))),
            nn.LeakyReLU(leak, inplace=True),

            spectral_norm(nn.Conv2d(64, 128, 3, stride=1, padding=(1,1))),
            nn.LeakyReLU(leak, inplace=True),
            spectral_norm(nn.Conv2d(128, 128, 4, stride=2, padding=(1,1))),
            nn.LeakyReLU(leak, inplace=True),

            spectral_norm(nn.Conv2d(128, 256, 3, stride=1, padding=(1,1))),
            nn.LeakyReLU(leak, inplace=True),
            spectral_norm(nn.Conv2d(256, 256, 4, stride=2, padding=(1,1))),
            nn.LeakyReLU(leak, inplace=True),

            spectral_norm(nn.Conv2d(256, 512, 3, stride=1, padding=(1,1))),
            nn.LeakyReLU(leak, inplace=True),
        )

        self.fc = spectral_norm(nn.Linear(w_g * w_g * 512, 1))

    def forward(self, x):
        m = self.model(x)
        return self.fc(m.view(-1, w_g * w_g * 512))
