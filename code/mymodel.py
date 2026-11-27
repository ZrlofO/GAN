"""Generative adversarial network (GAN): generator and discriminator.

TODO: complete the classes `Generator` and `Discriminator`.

You are free to design the architectures (type and number of layers, their
sizes, the latent dimension), but keep the interface described in the
docstrings: the notebook and the other files rely on it.

Notation
    B      : number of images in a batch
    z      : latent variables,             shape (B, latent_dim), z ~ N(0, I)
    x      : real images,                  shape (B, 1, 32, 32), values in [0, 1]
    G(z)   : generated images,             shape (B, 1, 32, 32), values in [0, 1]
    logits : outputs of the discriminator, shape (B,), one real number per image
    D(x)   : sigmoid(logit), the probability that the image x is real
"""

import torch
import torch.nn as nn


def init_weights(module):
    """Weight initialization of DCGAN: N(0, 0.02) for the convolutions."""
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)


class Generator(nn.Module):
    def __init__(self):
        """Define the layers of the generator.

        The notebook creates the generator with `Generator()`, so every
        argument you add needs a default value.

        Must set `self.latent_dim` (int, at least 2): the dimension of z.
        """
        super().__init__()
        self.latent_dim = 100
        # DCGAN: (B, 100, 1, 1) -> 4x4 -> 8x8 -> 16x16 -> 32x32
        self.net = nn.Sequential(
            nn.ConvTranspose2d(self.latent_dim, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1),
            nn.Sigmoid(),
        )
        self.apply(init_weights)

    def forward(self, z):
        """Generator: images G(z) generated from the latent variables z.

        Args:
            z: latent variables, shape (B, latent_dim).

        Returns:
            generated images, shape (B, 1, 32, 32), values in [0, 1].
        """
        return self.net(z.view(z.size(0), self.latent_dim, 1, 1))


class Discriminator(nn.Module):
    def __init__(self):
        """Define the layers of the discriminator.

        The notebook creates the discriminator with `Discriminator()`, so
        every argument you add needs a default value.
        """
        super().__init__()
        # DCGAN: 32x32 -> 16x16 -> 8x8 -> 4x4 -> 1x1
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, 1, 0),
        )
        self.apply(init_weights)

    def forward(self, x):
        """Discriminator: one logit per image.

        The logit is a real number: do not apply a sigmoid at the end. The
        probability that the image x is real is D(x) = sigmoid(logit).

        Args:
            x: real or generated images, shape (B, 1, 32, 32), values in [0, 1].

        Returns:
            logits, shape (B,).
        """
        return self.net(x).view(-1)
