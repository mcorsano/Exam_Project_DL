# Exam project - Deep Learning Course

## Authors 
Giulia Bernardi and Marianna Corsano

Introduction
=============
This is the project for the Deep Learning Exam.
It consists in the python implementation of four different GANs.

Launch
======
To run each one of the four networks:
```
python ./main.py
```

Technical details
=================

## Datasets
The datasets are loaded at runtime. In order to run correctly all the GANs, a directory hierarchy for the storage of the datasets must be guaranteed. In particular:

#### Gan
The simple Gan implementation relies on the MNIST dataset which is automatically downloaded and put in the correct folder. It is a database of handwritten digits which has a training set of 60,000 examples, and a test set of 10,000 examples.

#### DCGan
Our DCGan implementation allows for the training on three different datasets, all present in the `torchvision.datasets` pytorch module . When one dataset is selected, it is automatically downloaded, if not already present, and put in the correct folder.
The three datasets are 
* MNIST dataset: handwritten digits.
* CelebA dataset: large-scale face attributes dataset with more than 200K celebrity images, each with 40 attribute annotations.
* Flower102 dataset: category dataset, consisting of 102 flower categories occuring in the United Kingdom. Each class consists of between 40 and 258 images.


### CycleGan
The network has been trained on the "Horse2Zebra" dataset, downloaded from Kaggle. The folder hierarchy of the dataset must be the following:  
data  
<pre>
├───data  
│   └───train  
│       └───horses
|       └───zebras
│   └───val
│       └───horses
|       └───zebras
</pre>