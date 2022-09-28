# Exam project - Deep Learning Course

## Authors 
Giulia Bernardi and Marianna Corsano

Introduction
=============
This is the project for the Deep Learning Exam.  
It consists in the python implementation of four different Generative Adversarial Networks.

Launch
======
To run each one of the four networks:
```
python ./main.py
```

Technical details
=================
## Datasets  
The networks are trained on a given set of specified datasets that are either part of `torchvision.datasets` or downloaded from Kaggle.  
For every model, the correct folder hierarchy for dataset directories is specified.  

## Required external Packages   
* **tqdm** - for progression bar
```
pip install tqdm
```

* **PIL** - for image processing
```
pip install PIL
```

* **albumentations** - for image augmentation
```
pip install albumentations
```

* **PyTorch** - for machine learning framework  
*system-dependent installation: see https://pytorch.org/ for further details*