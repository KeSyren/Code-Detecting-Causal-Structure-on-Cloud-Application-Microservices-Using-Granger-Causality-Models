# Neural Granger Causality for Temporal Dependency Discovery

The coder is based on the `Neural-GC` repository (containing code for a neural network based approach to discovering Granger causality in nonlinear time series). The methods are described in [this paper](https://arxiv.org/abs/1802.05842).

## Installation

To install the code, please clone the repository. All you need is `Python 3`, `PyTorch (>= 0.4.0)`, `numpy` and `scipy`.

## Demos

See examples in the Jupyter notebooks `cmlp_lagged_var_demo.ipynb`, `clstm_lorenz_demo.ipynb`, and `crnn_lorenz_demo.ipynb`.

## Data
Simulation data : linear simulation data and nonlinear simulation data.

## Run the code

- download the code to any folder: [yourpath]/paper-GC-master/
- cd [yourpath]/paper-GC-master/
- source venv/bin/activate
- nohup python -u filename.py>log.out 2>&1 &
- INPUT: simulation time series data (e.g., 5 time series data)
- OUTPUT: learned weights (5*5 weights matrix)


## How it works

The models implemented in this repository, termed cMLP, cLSTM, and cRNN, are neural networks that model multivariate time series by forecasting each sequence separately. During training, sparse penalties on the first hidden layer's weight matrix set groups of parameters to zero, which can be interpreted as discovering Granger causality.

The cMLP model can be trained with three different penalties: group lasso, group sparse group lasso, and hierarchical. The cLSTM and cRNN models both use a group lasso penalty, and differ from one another only in the type of RNN cell they use.

Training models with non-convex loss functions and non-smooth penalties requires a specialized optimization strategy, and we use the generalized iterative shrinkage and thresholding algorithm ([GISTA](https://arxiv.org/abs/1303.4434)), which differs from the better known iterative shrinkage and thresholding algorithm (ISTA) only in its use of a line search criterion. Our implementation begins by performing ISTA steps without checking the line search criterion, and switches to a line search when the objective function fails to decrease sufficiently between loss checks.

## References
- Qing Wang, et. al "Detecting Causal Structure on Cloud Application Microservices Using Granger Causality Models", IEEE CLOUD, 2021.
- Alex Tank, Ian Covert, Nicholas Foti, Ali Shojaie, Emily Fox. Neural Granger Causality for Nonlinear Time Series. *arXiv preprint arXiv:1802.05842*, 2018.
- Chunqiu Zeng, Qing Wang, et. al, Online Inference for Time-varying Temporal Dependency Discovery from Time Series. IEEE Big Data, 2016.
