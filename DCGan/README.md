# DCGan
A DCGAN is a direct extension of the GAN, except that it explicitly uses convolutional and convolutional-transpose layers in the discriminator and generator, respectively.  
It was first described by Radford et. al. in the paper "Unsupervised Representation Learning With Deep Convolutional Generative Adversarial Networks" - 2015.

## Dataset
Our DCGan implementation allows for the training on three different datasets, all present in the `torchvision.datasets` pytorch module . When one dataset is selected, it is automatically downloaded, if not already present, and put in the correct folder.
The three datasets are:  
* MNIST dataset: handwritten digits.
* CelebA dataset: large-scale face attributes dataset with more than 200K celebrity images.
* Flower102 dataset: category dataset, consisting of 102 flower categories occuring in the United Kingdom. Each class consists of between 40 and 258 images.

## Results
![](saved_images/MNIST1.png)   (MNIST dataset)
![](saved_images/CelebA1.png)  (CelebA dataset)
![](saved_images/Flower1.png)  (Flower102 dataset)