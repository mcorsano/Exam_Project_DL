import torch
import utilities
import torch.nn as nn
from torchvision.utils import save_image



def test_model(x_generator, y_generator, testLoader):
    x, y = next(iter(testLoader))

    x_generator.eval()
    y_generator.eval()

    with torch.no_grad():
        it = 0
        for y, x in testLoader:
            x = x.to(utilities.DEVICE)
            y = y.to(utilities.DEVICE)

            y_fake = y_generator(x)
            x_fake = x_generator(y)

            save_image(x*0.5+0.5, "tests" + f"/x_{it}.png")
            save_image(x_fake*0.5+0.5, "tests" + f"/x_fake_{it}.png")
            save_image(y*0.5+0.5, "tests" + f"/y_{it}.png")
            save_image(y_fake*0.5+0.5, "tests" + f"/y_fake_{it}.png")
            it += 1

