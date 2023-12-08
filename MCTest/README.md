# CS1671 Final Project (Machine Comprehension)
### Jacob Emmerson
DUE: 12/08/23 @ 11:59pm

## About

This is the repository for my final project in CS1671 (Human Language Technologies). For this project, we were tasked with creating a question answering system which would be evaluated on the Machine Comprehension Test data created by Richardson, Burges, and Renshaw (2013). I chose to utilize an approach which selects candidate answers based on a given entailment score. The candidate answers which are most likely (highest scored) are chosen. There are two primary scripts feature within this repository: `RTE_class.py` and `THM_class.py`. These scripts were both built in Python v3.11.4 with Jupyter v1.0.0.

## Using the Models

In order to use the models, the following packages are required (versions listed are what were used when creating the scripts, but the exact versions may not be necessary):
- NLTK v3.8.1
- Pandas v2.0.3
- NumPy v1.24.3
- Scikit-learn v1.3.0

Place the following code at the top of a script or notebook to import the models:

Textual entailment model:
`from RTE_class import *` 

Baseline model:
`from THM_class import *`

Additionally, `dep.py` contains the functions upon which the models are built as well as the necessary functions to import data. To see working examples of these models, see `machine_comp.ipynb`.