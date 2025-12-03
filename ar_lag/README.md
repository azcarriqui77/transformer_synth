# Autoregressive model's optimal parameters study for transformer temperature novelties prediction

This directory contains the scripts for studying the optimal parameters of the autoregressive-based prediction model used in the thermal anomaly prediction algorithm for transformer temperature data. The AR model is employed to forecast temperature values and detect thermal novelties in various regions of a transformer. Apart from the AR lag study, the prediction window size is also analyzed to enhance the model's performance. The best pair of parameters (AR order and prediction window size) is determined based on the lowest values for AIC criteria.

## Directory structure

```
ar_lag/
├── modules — Different scripts of the implementation of the novelty detection algorithm.
├── README.md — This file.
├── simulation.py — Script that runs the prediction algorithm for different AR orders and a selected prediction window size in config.py file
├── notebook_lag.ipynb - Jupyter Notebook for analyzing and visualizing the results of the AR order study.
└──config.py — Configuration file for setting paths and parameters.
```