import torch
import torch.nn as nn
from generator import *
from discriminator import *
import utilities
import torchvision
from torchvision.utils import save_image
from tqdm import tqdm



def weights_init(model):
    for mod in model.modules():
        if isinstance(mod, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(mod.weight.data, 0.0, 0.02)



def train_model(dataloader, generator, discriminator, generator_optimizer, discriminator_optimizer, lossCriteria):
    fixed_z = torch.randn(32, utilities.Z_DIM, 1, 1).to(utilities.DEVICE)   # you can see progression as it trains
    loop = tqdm(dataloader, leave=True)  # for the progression bar

    for epoch in range(utilities.NUM_EPOCHS):
        for batch_index, (real_img, _) in enumerate(loop):   # no labels needed; unsupervised!
            real_img = real_img.to(utilities.DEVICE)
            z = torch.randn(utilities.BATCH_SIZE, utilities.Z_DIM, 1, 1).to(utilities.DEVICE)
            fake_img = generator(z)

            ### Train Discriminator
            ### loss function: max log(D(x)) + log(1 - D(G(z)))
            D_real = discriminator(real_img).reshape(-1)   # single value for each example
            loss_D_real = lossCriteria(D_real, torch.ones_like(D_real))

            D_fake = discriminator(fake_img).reshape(-1)
            loss_D_fake = lossCriteria(D_fake, torch.zeros_like(D_fake))

            discriminator_loss = (loss_D_real + loss_D_fake) / 2

            discriminator.zero_grad()
            discriminator_loss.backward(retain_graph=True)
            discriminator_optimizer.step()

            ### Train Generator
            ### loss function: max log(D(G(z))
            D_fake_new = discriminator(fake_img).reshape(-1)
            generator_loss = lossCriteria(D_fake_new, torch.ones_like(D_fake_new))

            generator.zero_grad()
            generator_loss.backward()
            generator_optimizer.step()

            if batch_index % 50 == 0:
                print(
                    f"Epoch [{epoch}/{utilities.NUM_EPOCHS}] Batch {batch_index}/{len(dataloader)} \
                        Loss D: {discriminator_loss:.4f}, loss G: {generator_loss:.4f}"
                )

                with torch.no_grad():
                    fake_img = generator(fixed_z)
                    save_image(fake_img, f"saved_images/fake_{epoch}_{batch_index}.png", normalize=True)
