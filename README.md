# Q-MRS
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18717924.svg)](https://doi.org/10.5281/zenodo.18717924)

This repository provides the code and associated resources for **“Q-MRS: Quantitative Magnetic Resonance Spectral Analysis Using Deep Learning”**.

It includes:
* pre-trained model weights for the [Convolutional Vision Transformer (CvT)](https://arxiv.org/abs/2103.15808)
* the [BIG GABA](https://www.nitrc.org/projects/biggaba/) MEGA-PRESS dataset processed using the [Osprey software package](https://github.com/schorschinho/osprey/tree/develop)
* an example notebook demonstrating model inference on *in vivo* data

The simulated dataset used to train the CvT is publicly available on Zenodo (see DOI above).

## Optimization Loop
![](figures/invivo_optimization.svg)

## Network Architecture
![](figures/cvt_architecture.svg)

## Contact
Please feel free to contact cw3360@columbia.edu with any questions, comments, or suggestions.