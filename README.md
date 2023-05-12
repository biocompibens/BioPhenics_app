# Transfer Learning for Versatile Training Free  High Content Screening Analyses

#### *[**article link**](https://doi.org)*

### Requirements 
Python version : 3.10

Python libraries: pip install -r requirements.txt

This repository contains scripts used to transform and analyze HCS data  in a fully automated manner.
The two main scripts are featurization.py and normalisation.py

### From image to feature using featurization.py

#### Usage : python featurization.py images.xlsx features.xlsx

This script consumes rows from an input file images.xlsx to get image paths and pass them through a ResNet as described in *[**article link**](https://doi.org)*.
It will save the output in a file named feature.xlsx.

#### Input.xlsx specification

An example can be found at example/1_input_deepmodule (toy).xlsx. It needs to be an Excel file with a sheet named `Image_path` 
which contains four columns named `Barcode`, `Well`, `Content` and `Path`. If optional columns `Fields` and `Wave Length` are present, 
features will be concatenated for each value of `Wave Length` and aggregated per values of `Fields`.
Additional columns are ignored. 

Note : One can modify the main part of the script to accept other file format (underlying class expect a Pandas DataFrame)

### Feature normalization and analysis using analysis.py

#### Usage : python analysis.py features.xlsx parameters_file.json selected_hit.xlsx

This script takes as input an Excel file (example/2_data.xlsx) as first argument, 
it can be the output of the featurization.py script or of an handmade image analysis, 
a parameters file (example/2_parameters.json)
as second argument and the name of the output file (such as "selected_hit.xlsx") as third argument.

#### features.xlsx specification

An Excel file with a sheet named `All_Data`.

Mandatory columns :
- Barcode = unique id per plate
- Plate = unique name per layout (compounds disposition)
- Well = plate localization, a letter for row (A-P) and a number for columns (1-24).
- Content = Name of the content
- Line = Name of the cell line
- Replicate = Replicate identifier
- At least one column of feature (will take every additional column where all values can be detected as numbers)

Other columns will be ignored.

#### file_parameters.json

A json file with some key info for the normalization process.
Dictionary values are :
- ctrl_neg = a list of sample names to be considered as negative control.
- ctrl_pos = a list of sample names to be considered as positive control.
- features = a list of column names to apply the normalization process on. In case of features obtained by featurization.py, the special value '["deep learning feature"]' should be used.
- other_parms = a dict of parameters which can contain (among other things) the key `reduced_feature_space` with boolean value (True of False). If True, a PCA is launched on the feature space in order to keep 99% (default value) of the data variability.
- spatial_correction = name of the method in ``scripts.methods.CorrectionMethod`` to use as spatial correction method. (underscore can be replaced by space)
- sc_parms = dict to be passed as parameters to the spatial correction method (see list of parameters for the chosen function)
- normalization = name of the normalization method in ``scripts.methods.NormalisationMethod`` to be used. (underscore can be replaced by space)
- n_parms = dict to be passed as parameters to the normalization method (see list of parameters for the chosen function)
- selection = name of the method in ``scripts.methods.HitSelection`` to be used for hit selection. (underscore can be replaced by space)
- s_parms = dict to be passed as parameters to the hit selection method. It can contain two values : 
  - "str_parms" = a list of rules to select hits. Each rule comprises 4 values :
    - 'include' = None, 'and' or 'or' indicate how to pile rules
    - 'feature' = name of the feature to be used for selection (some hit selection method can output new feature name, ex: "linear discriminant analysis" will output a feature named "LDA". See method documentation for details)
    - 'relative' = '<', '>' or '><' indicate the  direction of the threshold ('><' means outside of [-|x|, |x|])
    - 'value' = threshold of the rule
  - "other_parms" = specific parameters for the chosen function in selection.
