# Pix2Pix
Pix2Pix is a Generative Adversarial Network, designed for general purpose image-to-image translation.

This approach was first presented by Phillip Isola, et al. in 2016, in the paper titled “Image-to-Image Translation with Conditional Adversarial Networks” and presented at CVPR in 2017.

## Dataset
The network has been trained on the "maps" dataset, downloaded from Kaggle.  
The folders containing the dataset must follow the following hierarchy:

<pre>
├───data  
│   └───maps  
│       └───test
|       └───train
│       └───val
</pre>

## Results
Real:![](validations/real.png)
Generated image:![](validations/generated.png)
Real target:![](validations/label.png)