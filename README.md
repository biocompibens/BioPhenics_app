# Transfer Learning for Versatile Training Free  High Content Screening Analyses

#### *[**Lien vers l'article.**](https://doi.org)*

### Requirements 
Python version : 3.10

Python libraries: pip install -r requirements.txt

This repository contains code used to transform and analyse data from HCS in BioPhenics Platform at Curie Institute.
The two main scripts are deep_learning.py and normalisation.py

### From image to data using deeplearning.py

#### Usage : python deeplearning.py input.xlsx output.xlsx

This script will use data from input.xlsx to get images. Then pass them through a ResNet as described in *[**article link**](https://doi.org)*.
It will write the result in output.xlsx.

#### Input.xlsx specification

It needs to be an Excel file with a sheet named `Image_path` 
which contains columns named `Barcode`, `Wells`, `Content` and `Path`. Columns `Fields` and `Wave Length` are optional 
but if present, data will be concatenated for each value of `Wave Length` and aggregated per values of `Fields`.
Other columns are ignored. 

Note : One can modify the main part of the script to accept other file format (Class needs a Pandas' DataFrame)

### Specific normalization process for HCS data using normalisation.py

#### Usage : python normalisation.py data.xlsx parameters_file.json output.xlsx

This script will take an Excel file (data.xlsx) as first argument, a parameters file (parameters_file.json)
as second argument and the name of the output (output.xlsx) as third argument.

#### Data.xlsx specification

An Excel file with a sheet named `All_Data` where its content is disposed in columns.

Mandatory columns :
- Barcode = unique identifiant per plate
- Plate = unique name per layout (compounds disposition)
- Wells = plate localization, a letter for row (A-P) and a number for columns (1-24).
- Content = Name of the content
- Line = Name of the cell line
- Replicat = Replicate identifier
- At least one column of feature (will take every column where all values can be assimilated as numbers)

Other columns will be ignored.

#### file_parameters.json

A parameters instance will be created with this file. It should be a dict in JSON format, with some key info for normalisation process.
Value are :
- ctrl_neg = a list of content name to be considered as negative control.
- ctrl_pos = a list of content name to be considered as positive control.
- features = a list of column name to make the normalisation on.
- spatial_correction = name of the method in ``scripts.methods.CorrectionMethod`` to use as spatial correction method. (underscore can be replaced by space)
- sc_parms = dict to be passed as parameters to spatial correction method (see list of parameters for the chosen function)
- normalization = name of the method in ``scripts.methods.NormalisationMethod`` to use as normalization method. (underscore can be replaced by space)
- n_parms = dict to be passed as parameters to normalization method (see list of parameters for the chosen function)
- selection = name of the method in ``scripts.methods.HitSelection`` to use as hit selection. (underscore can be replaced by space)
- s_parms = dict to be passed as parameters to hit selection method. It can contain two values : 
  - "str_parms" = a list of rule to select hit based of. Each rule comprises 4 values :
    - 'include' = None, 'and' or 'or' indicate how to pile rules
    - 'feature' = name of the feature to base the selection on
    - 'relative' = '<', '>' or '><' indicate the  direction of the threshold ('><' means outside of [-|x|, |x|])
    - 'value' = threshold of the rule
  - "other_parms" = specific parameters for the chosen function in selection.
