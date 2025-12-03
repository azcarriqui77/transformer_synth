# Thermal model for a power transformer - Synthetic temperature profile generation and AR model order study

This repository contains the Python scripts for the **generation of a synthetic temperature profile** for the previously defined regions of a transformer, simulating its thermal behavior over a full year based on its thermal data collected over a single day. Additionally, the **optimal parameters of the AR model** for the proposed prediction algorithm is studied.

<p style="text-align:center"><img src="termico.gif" alt="Thermal model animation" width="600" center="true"></p>

---

## Overview

This project implements a set of **Python** scripts (based on an previous **MATLAB** implementation of the project) to generate and analyze synthetic annual temperature values in various zones of an electrical transformer, starting from its temperature values for a specific day. This is done by modeling the transformer as a thermal system with random anomalies and malfunctions. The goal is not to create a model that perfectly matches the real-world system, but one that exhibits behavior **qualitatively similar** to that of an electrical transformer.

Subsequently, using this synthetic data, the performance of an **autoregressive (AR) algorithm** based on Artificial Intelligence and designed for **detecting thermal novelties** in the transformer's components is tested. This is the **primary objective** for using this data. Beforehand, the **optimal order** of this algorithm is studied for its best performance.

Associated article link for more detail:

---

## Repository Structure

In the main directory, we find:

```markdown
├── README.md — Main project documentation.
├── ar_lag — Scripts for studying the optimal order of the AR prediction model.
├── environment.yml
├── predictions — Implementation of the anomaly prediction algorithm.
└── synthetic_model — Creation of the synthetic annual temperature profile.
```

The directory `synthetic_model` contains the scripts for creating the synthetic annual temperature profile. The directory `ar_lag` contains the scripts for studying the optimal order of the AR prediction model. The directory `predictions` contains the scripts to run the anomaly prediction algorithm. Inside each directory, there is a `README.md` file with more detailed information about its contents and usage.

---

## Core Features

  * **Synthetic generation** of temperature data for different regions of a transformer over the course of a year.
  * **Export** of results in formats compatible with the Python language.
  * **Configurable parameters**, such as the system's thermodynamic parameters.
  * **Reproducible environment** using `environment.yml` to ensure compatibility across different machines.
  * **Validation of thermal novelty detection algorithms** that alert of potential faults in the transformer.

-----

## Usage

### Create Conda Environment

To replicate the environment used for synthetic data generation:

```bash
conda env create -f synthetic_model/environment.yml
conda activate resisto_syn
```

(Replace `resisto_env` with the name defined in your `environment.yml` if it is different.)

### Execution

In the file **`synthetic_model/python_code/configuration.py`**, replace the variable **`ruta`** with the path to the directory where the repository is located.

From the **`python_code/`** directory:

```bash
python main.py
```

The results will be automatically saved in the **`output/`** directory.

To execute the anomaly detection algorithm:

```bash
conda env create -f predictions/environment.yml
conda activate resisto_pred
```

```bash
python predictions/simulation.py
```

---

## Acknowledgements

This work is funded by Universidad de Granada and Endesa Distribución under the Endesa-UGR chair in Artificial Intelligence. Besides, this research is part of the PID2022-137451OB-I00 and PID2022-137629OA-I00 projects, funded by the CIN/AEI/10.13039/501100011033 and by FSE+, and the C-ING-183-UGR23 project, cofunded by Consejería de Universidad, Investigación e Innovación and by the European Union under the Andalusia ERDF Program 2021-2027. 

---

