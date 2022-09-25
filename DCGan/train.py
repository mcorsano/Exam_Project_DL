import torch
import torch.nn as nn
from generator import *
from discriminator import *
import utilities
import torchvision
from torchvision.utils import save_image
from tqdm import tqdm


def initialize_weights(model):
    # weights are initialized according to DCGan paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)



def train_model(dataloader, generator, discriminator, generator_optimizer, discriminator_optimizer, lossCriteria):
    fixed_noise = torch.randn(32, utilities.NOISE_DIM, 1, 1).to(utilities.DEVICE)   # you can see progression as it trains
    loop = tqdm(dataloader, leave=True)  # for the progression bar

    for epoch in range(utilities.NUM_EPOCHS):
        for batch_idx, (real, _) in enumerate(loop):   # no labels needed; unsupervised!
            real = real.to(utilities.DEVICE)
            noise = torch.randn(utilities.BATCH_SIZE, utilities.NOISE_DIM, 1, 1).to(utilities.DEVICE)
            fake = generator(noise)

            ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
            D_real = discriminator(real).reshape(-1)   # single value for each example
            loss_D_real = lossCriteria(D_real, torch.ones_like(D_real))

            D_fake = discriminator(fake).reshape(-1)
            loss_D_fake = lossCriteria(D_fake, torch.zeros_like(D_fake))

            discriminator_loss = (loss_D_real + loss_D_fake) / 2

            discriminator.zero_grad()
            discriminator_loss.backward(retain_graph=True)
            discriminator_optimizer.step()

            ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
            output = discriminator(fake).reshape(-1)
            generator_loss = lossCriteria(output, torch.ones_like(output))

            generator.zero_grad()
            generator_loss.backward()
            generator_optimizer.step()

            if batch_idx % 50 == 0:
                print(
                    f"Epoch [{epoch}/{utilities.NUM_EPOCHS}] Batch {batch_idx}/{len(dataloader)} \
                        Loss D: {discriminator_loss:.4f}, loss G: {generator_loss:.4f}"
                )

                with torch.no_grad():
                    fake = generator(fixed_noise)
                    save_image(fake, f"saved_images/fake_{epoch}_{batch_idx}.png", normalize=True)
