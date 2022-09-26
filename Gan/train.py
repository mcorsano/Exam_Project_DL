import torch
import torch.nn as nn
import utilities
import torchvision
from torchvision.utils import save_image



def train_model(dataLoader, generator, discriminator, generator_optimizer, discriminator_optimizer, lossCriteria):
    fixed_noise = torch.randn((utilities.BATCH_SIZE, utilities.Z_DIM)).to(utilities.DEVICE)

    for epoch in range(utilities.NUM_EPOCHS):
        for batch_idx, (real_img, _) in enumerate(dataLoader):          # batch_idx, (real_images, labels)
            real_img = real_img.view(-1, 784).to(utilities.DEVICE)      # linearization

            ### Train Discriminator: max log(D(real)) + log(1 - D(G(noise)))
            noise = torch.randn(utilities.BATCH_SIZE, utilities.Z_DIM).to(utilities.DEVICE)  # from gaussian(0,1), shape(32x64)
            fake = generator(noise)

            D_real = discriminator(real_img).view(-1)   # flattened. it has shape torch.Size([32])
            loss_D_real = lossCriteria(D_real, torch.ones_like(D_real))

            D_fake = discriminator(fake).view(-1)
            loss_D_fake = lossCriteria(D_fake, torch.zeros_like(D_fake))

            discriminator_loss = (loss_D_real + loss_D_fake) / 2

            discriminator.zero_grad()
            discriminator_loss.backward(retain_graph=True)   # bc we need "fake" for the training of the generator
            discriminator_optimizer.step()

            ### Train Generator: min log(1 - D(G(noise))) <-> max log(D(G(noise))
            # we decide to go for the maximization instead of the minimization process
            # since performs better in terms of saturating gradient
            output = discriminator(fake).view(-1)
            generator_loss = lossCriteria(output, torch.ones_like(output))

            generator.zero_grad()
            generator_loss.backward()
            generator_optimizer.step()

            if batch_idx == 0:
                print(
                    f"Epoch [{epoch}/{utilities.NUM_EPOCHS}] Batch {batch_idx}/{len(dataLoader)} \
                        Loss D: {discriminator_loss:.4f}, loss G: {generator_loss:.4f}"
                )

                with torch.no_grad():
                    fake = generator(fixed_noise).reshape(-1, 1, 28, 28)
                    real = real_img.reshape(-1, 1, 28, 28)

                    save_image(fake, f"saved_images/fake_{epoch}.png", normalize=True)
                    save_image(real, f"saved_images/real_{epoch}.png", normalize=True)


