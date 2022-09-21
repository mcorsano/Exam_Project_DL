import torch
import torch.nn as nn
import utilities
import torchvision
from torchvision.utils import save_image




def train_model(dataLoader, generator, discriminator, generator_optimizer, discriminator_optimizer, lossCriteria):
    fixed_noise = torch.randn((utilities.BATCH_SIZE, utilities.Z_DIM)).to(utilities.DEVICE)

    step = 0

    for epoch in range(utilities.NUM_EPOCHS):
        for batch_idx, (real_img, label) in enumerate(dataLoader):          # batch_idx, (real_images, labels)
            real_img = real_img.view(-1, 784).to(utilities.DEVICE)      # linearization

            ### Train Discriminator: max log(D(real)) + log(1 - D(G(noise)))
            noise = torch.randn(utilities.BATCH_SIZE, utilities.Z_DIM).to(utilities.DEVICE)  # from gaussian(0,1), shape(32x64)
            fake = generator(noise)

            disc_real = discriminator(real_img).view(-1)   # flattened
            lossD_real = lossCriteria(disc_real, torch.ones_like(disc_real))

            disc_fake = discriminator(fake).view(-1)
            lossD_fake = lossCriteria(disc_fake, torch.zeros_like(disc_fake))

            lossD = (lossD_real + lossD_fake) / 2

            discriminator.zero_grad()
            lossD.backward(retain_graph=True)
            discriminator_optimizer.step()

            ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
            # we decide to go for the maximization instead of the minimization process
            # since performs better in terms of saturating gradient
            output = discriminator(fake).view(-1)
            lossG = lossCriteria(output, torch.ones_like(output))
            generator.zero_grad()
            lossG.backward()
            generator_optimizer.step()

            if batch_idx == 0:
                print(
                    f"Epoch [{epoch}/{utilities.NUM_EPOCHS}] Batch {batch_idx}/{len(dataLoader)} \
                        Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
                )

                with torch.no_grad():
                    fake = generator(fixed_noise).reshape(-1, 1, 28, 28)
                    data = real_img.reshape(-1, 1, 28, 28)

                    save_image(fake*0.5+0.5, f"saved_images/fake_{epoch}.png")
                    save_image(data*0.5+0.5, f"saved_images/real_{epoch}.png")

                    step += 1



