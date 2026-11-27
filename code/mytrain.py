"""Training of the GAN: loss functions and one epoch of training.

TODO: complete `discriminator_loss`, `generator_loss` and `train_one_epoch`.

Notation
    x           : real images
    G(z)        : images generated from latent variables z ~ N(0, I)
    real_logits : outputs of the discriminator for x,    shape (B,)
    fake_logits : outputs of the discriminator for G(z), shape (B,)
    D(.)        : sigmoid(logit), the probability that an image is real
"""

import torch
import torch.nn.functional as F

# Number of updates of the generator for one update of the discriminator
GENERATOR_STEPS = 3

# The images given to the discriminator, real and generated alike, are shifted
# by up to MAX_SHIFT pixels in each direction. With 500 training images the
# discriminator otherwise memorizes them and stops guiding the generator.
MAX_SHIFT = 2


def random_shift(images, max_shift=MAX_SHIFT):
    """Shift every image by its own random offset, filling with zeros.

    The output is differentiable with respect to `images`, so the generator
    still receives gradients through the shifted images.

    Args:
        images: shape (B, C, H, W).

    Returns:
        shifted images, shape (B, C, H, W).
    """
    batch_size, _, height, width = images.shape
    device = images.device
    padded = F.pad(images, [max_shift] * 4)
    top = torch.randint(0, 2 * max_shift + 1, (batch_size, 1, 1), device=device)
    left = torch.randint(0, 2 * max_shift + 1, (batch_size, 1, 1), device=device)
    index = torch.arange(batch_size, device=device).view(-1, 1, 1)
    rows = top + torch.arange(height, device=device).view(1, -1, 1)
    cols = left + torch.arange(width, device=device).view(1, 1, -1)
    return padded[index, :, rows, cols].permute(0, 3, 1, 2)


def discriminator_loss(real_logits, fake_logits):
    """Loss of the discriminator for a batch.

        loss_d = - mean of log D(x)  -  mean of log(1 - D(G(z)))

    It is the binary cross-entropy of the real images with the label 1
    (real) plus the binary cross-entropy of the generated images with the
    label 0 (generated). Each term is averaged over the B images of the
    batch.

    Compute the loss from the logits, not from the probabilities: it is
    numerically stable (see F.binary_cross_entropy_with_logits).

    Args:
        real_logits: outputs of the discriminator for real images,      shape (B,).
        fake_logits: outputs of the discriminator for generated images, shape (B,).

    Returns:
        loss_d: a scalar tensor.
    """
    loss_real = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
    loss_fake = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
    return loss_real + loss_fake


def generator_loss(fake_logits):
    """Loss of the generator for a batch (non-saturating loss).

        loss_g = - mean of log D(G(z))

    It is the binary cross-entropy of the generated images with the label 1
    (real), averaged over the B images of the batch: the generator is
    trained to have its images classified as real.

    Args:
        fake_logits: outputs of the discriminator for generated images, shape (B,).

    Returns:
        loss_g: a scalar tensor.
    """
    return F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))


def train_one_epoch(generator, discriminator, optimizer_g, optimizer_d, dataloader, device):
    """Train the GAN for one epoch, i.e. one pass over `dataloader`.

    Put the two networks in training mode first. Then, for every mini-batch
    of B real images x:

        1. Move x to `device`, draw B latent variables z ~ N(0, I) and
           generate the images G(z).
        2. Update the discriminator: compute `discriminator_loss` from its
           outputs for x and for G(z), back-propagate, and update its
           parameters with `optimizer_d`. This loss must not send gradients
           to the generator: detach G(z).
        3. Update the generator: compute `generator_loss` from the outputs
           of the updated discriminator for G(z), back-propagate, and update
           its parameters with `optimizer_g`.

    Args:
        generator:     the generator, already on `device`.
        discriminator: the discriminator, already on `device`.
        optimizer_g:   the optimizer of the generator, from myoptim.get_optimizers.
        optimizer_d:   the optimizer of the discriminator, from myoptim.get_optimizers.
        dataloader:    yields batches of real images, shape (B, 1, 32, 32),
                       values in [0, 1] (there are no labels).
        device:        torch.device to compute on.

    Returns:
        dict {"loss_d": ..., "loss_g": ..., "d_real": ..., "d_fake": ...} of
        Python floats, the average of each term per image over the whole
        epoch:
            loss_d : loss of the discriminator (step 2)
            loss_g : loss of the generator (step 3)
            d_real : D(x), probability given to the real images (step 2)
            d_fake : D(G(z)), probability given to the generated images (step 2)
    """
    generator.train()
    discriminator.train()

    totals = {"loss_d": 0.0, "loss_g": 0.0, "d_real": 0.0, "d_fake": 0.0}
    num_images = 0

    for x in dataloader:
        x = x.to(device)
        batch_size = x.size(0)
        z = torch.randn(batch_size, generator.latent_dim, device=device)
        fake = generator(z)

        real_logits = discriminator(random_shift(x))
        fake_logits = discriminator(random_shift(fake.detach()))
        loss_d = discriminator_loss(real_logits, fake_logits)
        optimizer_d.zero_grad()
        loss_d.backward()
        optimizer_d.step()

        # The first update uses the images generated above, the next ones new images
        loss_g_sum = 0.0
        for step in range(GENERATOR_STEPS):
            if step > 0:
                z = torch.randn(batch_size, generator.latent_dim, device=device)
                fake = generator(z)
            loss_g = generator_loss(discriminator(random_shift(fake)))
            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()
            loss_g_sum += loss_g.item()

        totals["loss_d"] += loss_d.item() * batch_size
        totals["loss_g"] += loss_g_sum / GENERATOR_STEPS * batch_size
        totals["d_real"] += torch.sigmoid(real_logits).sum().item()
        totals["d_fake"] += torch.sigmoid(fake_logits).sum().item()
        num_images += batch_size

    return {term: total / num_images for term, total in totals.items()}
