## Fourier Sensitivity and Regularization of Computer Vision

Official code repository for the paper "Fourier Sensitivity and Regularization of Computer Vision".

Fourier Sensitivity of computer vision models is based on a rigorously defined measure of sensitivity to input frequencies.
Please see the paper for details.

<!-- ![Fourier-sensitivity](github-image.png) -->
<img src="github-image.png" width="750px">

## Installing libraries
The code was run with Python3.8

```
pip install -r requirements.txt
```

## Generating Fourier Sensitivity Plots
Please see the jupyter notebook "Fourier-Sensitivity.ipynb" for examples of plotting the Fourier Sensitivity of pre-trained models. 
Run all cells to re-generate the plots (be sure to change the path to the dataset, i.e., PATH\_TO\_IMAGENET)

## Fourier Regularization
We have provided a reference implementation of Fourier-regularizer training on CIFAR10 (train.py). Please use the commands below. 
```
# standard training (CIFAR10)
python train.py

# Fourier-regularized training (CIFAR10)
python train.py --regularizer {LSF,MSF,ASF} --regularier_lambda 0.5

```

