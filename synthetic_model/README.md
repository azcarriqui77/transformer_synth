# Synthetic model for annual temperature profile of transformers

This directory contains the scripts for generating a synthetic annual temperature profile of transformers. The synthetic data simulates temperature variations in different regions of a transformer over the course of a year, allowing for the testing and evaluation of anomaly detection methods.

## Directory structure

```
synthetic_model/
├── python_code — Scripts for generating the synthetic temperature profile.
├── matlab_code — Old MATLAB scripts for synthetic data generation.
├── temperature_ambience - Contains the ambient temperature model used for generating a ambient temperature profile, which is necessary for the synthetic temperature generation. Its implementation is in a Jupyter Notebook.
└── README.md — This file.
```

Inside the `python_code` directory, you will find the main scripts for generating the synthetic temperature profile. For running the simulation and generation process, just execute the `main.py` script after configuring the path in `configuration.py`. This path will be used to save the output results in the `output/` directory.