# Bayesian Neural Networks for Survival Data
#### Probabilistic and Machine Learning course at University of Trieste (prof. Luca Bortolussi)
by Alice Conighi, Emanuele Valea, Giovanni Zedda 

A.Y. 2025-2026

## Setup
The execution of the cells of the Jupyter notebooks and the side code requires `torch`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `lifelines` and `ucimlrepo`.

## Files Description
- `EDA_and_CoxModel.ipynb` is a notebook containing a description of the dataset and a preliminary analysis based on the classical proportional hazards
  (Cox) model;
- `cox-bnn.ipynb` is the **core notebook**, which contains more advanced methods and techniques for the analysis of censored data. `cox-bnn.py` is just
  code to carry out experiments;
- `cox-classic-mlp.py` is a script to compare Bayesian neural networks with classical neural networks;
- `helpers` is a mini-library for custom functionalities.

## Dataset
The *AIDS Clinical Trials Group Study 175* data were retrieved
through the [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/dataset/890/aids+clinical+trials+group+study+175),
but the data is referred as external, from the [NIH Website](https://clinicaltrials.gov/study/NCT00000625).

## AI Policy
0-AI Policy. No AI was used in the development of the code or the annexed slides and material, nor was used as a blind reference for the understanding of the topic and the derivation of the methods.

## License and Credits
The code is under [MIT License](LICENSE).

Data hereby provided are under their respective licenses and ownerships.

---
Alice Conighi, MSc student in Data Science and Artificial Intelligence 

Giovanni Zedda, MSc student in Data Science and Artificial Intelligence 

Emanuele Valea, MSc student in Stastical and Actuarial Sciences

University of Trieste, July 2026

---
Copyright (c) 2026 A. Conighi, E. Valea, G. Zedda
