# Bayesian Multi-View Clustering

This repository contains code and parameter details supporting F1000 submission Bayesian Multi-View Clustering Given Complex Inter-view Structure.

## Tutorial

We suggest that new users begin by stepping through the tutorial Jupyter notebook located in eval/simulations. It demonstrates the use of BMVC on simulated data has described in the original paper. 
 
## Repository description  

The bmvc/ directory contains:
- BaseBMVCModel.py: the base class for BMVC. 
- FullBMVCModel.py: class implementing the full likelihood of the BMVC model. 
- PseudoBMVCModel.py: class implementing the pseudolikelihood of the BMVC model. 
- gd_base.py: the wrapper framework for a gradient descent inference procedure.
- vi_base.py: the wrapper framework for a black box variational inference  inference procedure.  
- helpers.py: supporting helper functions for computation.
- runners.py: user-friendly functions which wrap the learning procedure for varying inference procedures and likelihoods. 

The eval/ directory contains:
- a simulations/ directory with:  a notebook demonstrating a sample use of BMVC on simulated data; supporting code for generating simulated data; a runner script run_simulation.py which runs the simulation analyses detailed in the paper. 
- pr_params.json: the parameters used for running the public health analyses detailed in the paper. 
- tcga_params.json:  the parameters used for running the biological analyses detailed in the paper. 


## Dependencies 

You will need the following packages:
- autograd
- numpy
- scipy
- matplotlib
- sklearn
- pandas
- seaborn

## Inference framework 

The inference framework used here is based on an extension of the work done by David Duvenaud and Ryan Adams in "Black-Box Stochastic Variational Inference in Five Lines of Python". (See https://www.cs.toronto.edu/~duvenaud/papers/blackbox.pdf). It also inspired in part by James Vucovic's (@jamesvuc) fork, in which he implemented a variety of really useful gradient estimates. 

We have further extended this framework into an easier-to-use lightweight framework for quick model iteration using black box variational inference in python. Please see https://github.com/bshapiro/bbvi-framework for more details.
