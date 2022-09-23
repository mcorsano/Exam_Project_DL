import torch
import torch.nn as nn
from generator import *
from discriminator import *
import utilities
import torchvision
from torchvision.utils import save_image



def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)



def train_model(dataloader, gen, disc, opt_gen, opt_disc, criterion):
    fixed_noise = torch.randn(32, utilities.NOISE_DIM, 1, 1).to(utilities.DEVICE)   # you can see progression as it trains
    step = 0

    for epoch in range(utilities.NUM_EPOCHS):
        # Target labels not needed! <3 unsupervised
        for batch_idx, (real, _) in enumerate(dataloader):
            real = real.to(utilities.DEVICE)
            noise = torch.randn(utilities.BATCH_SIZE, utilities.NOISE_DIM, 1, 1).to(utilities.DEVICE)
            fake = gen(noise)

            ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
            disc_real = disc(real).reshape(-1)
            loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = disc(fake.detach()).reshape(-1)
            loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            loss_disc = (loss_disc_real + loss_disc_fake) / 2
            disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()

            ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
            output = disc(fake).reshape(-1)
            loss_gen = criterion(output, torch.ones_like(output))
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            # Print losses occasionally and print to tensorboard
            if batch_idx % 2 == 0:
                print(
                    f"Epoch [{epoch}/{utilities.NUM_EPOCHS}] Batch {batch_idx}/{len(dataloader)} \
                        Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
                )

                with torch.no_grad():
                    fake = gen(fixed_noise)
                    save_image(fake, f"saved_images/fake_{epoch}_{batch_idx}.png", normalize=True)
                    #save_image(data*0.5+0.5, f"saved_images/real_{epoch}.png")

                    # take out (up to) 32 examples
                    # img_grid_real = torchvision.utils.make_grid(
                    #     real[:32], normalize=True
                    # )
                    # img_grid_fake = torchvision.utils.make_grid(
                    #     fake[:32], normalize=True
                    # )

                #     writer_real.add_image("Real", img_grid_real, global_step=step)
                #     writer_fake.add_image("Fake", img_grid_fake, global_step=step)

                # step += 1