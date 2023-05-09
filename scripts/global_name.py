ALL_DATA = "All_Data"
"""Name of the spreadsheet of the data by wells"""

IMG_PATH = "Image_path"
"""Name of the spreadsheet of the pictures path"""

FIELDS_DATA = "Fields"
"""Name of the spreadsheet of the fields data"""

# columns names in all Dataframes created by the program
REPLICAT = "Replicat"
"""The name of the column indicating the name of the replicat"""

WELLS = "Wells"
"""The name of the column indicating the position on the plate"""

ROW = "Row"
"""The name of the column indicating the row of the wells"""

COLUMN = "Column"
"""The name of the column indicating the columns of the wells"""

FIELDS = "Fields"
"""The name of the column of fields"""

BARCODE = "Barcode"
"""The name of the column of barcode"""

PLATE = "Plate"
"""The name of the column of plate"""

LIGNEE = "Line"
"""The name of the column of line"""

CONTEXTE = "Synthetic Context"
"""The name of the column of synthetic context (find in the experience file)"""

CONTENT = "Content"
"""The name of the column of compound"""

PATH = 'Path'
"""The name of the column path to image"""

WAVE = "Wave Length"
"""The name of the column wave length corresponding of the image"""

LABEL = "Label"
"""The name of the column that indicate if the compound is in the hitlist"""

SIRNA_NAME = "SiRNA name"

SIRNA = "Hits / Total"

PLATE_AND_WELLS = f"{PLATE}, {WELLS}"

MAHALANOBIS = "Mahalanobis distance"

COL_ORDERED = [PLATE, REPLICAT, CONTEXTE, LIGNEE,
               CONTENT, SIRNA_NAME, WELLS, BARCODE, SIRNA]
"""The order in which are sort dataframes"""

NOT_FEATURE = COL_ORDERED + [WAVE, LABEL, FIELDS, PATH, SIRNA, CONTENT]

CELL_COUNT = 'Cell Count'
LUMINESCENCE = 'Luminescence intensity'

LDA = "LDA"
LDA2 = "LDA_2"
LDA3 = "LDA_3"
LDA_COL = [LDA, LDA2, LDA3]

CLUSTER = "Cluster name"

SHEETNAMES = (
    ALL_DATA, "Normalized_data", "Outliers", "Img_path",
    'Parameters', "Curve_Fitting", "Corrected_data", "Transformed_data",
    'Median_Values', 'Non-pooled SiRNA hits', "Specific hit by lines"
)
"""Name of the sheetnames for creating an excel file"""

INFERENCE_WAVE_BOOL = 'inference_wavelength_bool'
FEATURE_NAME_PREFIX_DEEP_LEARNING = "Inference_"
DEEP_LEARNING_FEATURES = "deep learning features"
INFERENCE_BY_WAVE_POSSIBLE = "inference_by_wave_possible"
WAVELENGTH_NAMES = 'wavelength_names'
REDUCE_FEATURE_SPACE = 'reduced_feature_space'
LR_COLNAME = 'Logistic probability'
