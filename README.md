# Transfer Learning for Versatile Training Free  High Content Screening Analyses

#### *[**Lien vers l'article.**](https://doi.org)*

### Requirements 
Python libraries: pip install -r requirements.txt


This repository contains code used to transform and analyse data from HCS in BioPhenics Platform at Curie Institute.
The two main scripts are deep_learning.py and normalisation.py

### From image to data using deeplearning.py

#### Usage : python deeplearning.py path/to/input.xlsx path/to/output.xlsx

This script needs a xlsx file as first argument and will output a xlsx compatible with normalisation script input with filename as in second argument.


### Specific normalization process for HCS data using normalisation.py

#### Usage : python normalisation.py path/to/input.xlsx path/to/parameters_file.json path/to/output.xlsx

This script will take a xlsx file as first argument, a txt file for parameters as second argument and the third is the filename of the output
