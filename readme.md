# Step 01: Data proccessing. 

`step01_data_processing.R`, a data preprocessing pipeline designed for a specific dataset consisting of control group and PKU patients data. 

## Description 

The script consists of a series of operations that are carried out in five main steps: Library Imports, Function Definitions, Data Loading & Cleaning, Data Processing, and Data Merging & Saving. 

In brief, the operations include:

- Loading required libraries for data manipulation, visualization, and imputation.
- Defining functions for outlier detection, outlier removal, and missing data imputation.
- Loading data from specified file paths, followed by initial data cleaning operations such as column renaming, row filtering based on certain conditions, and date type conversion.
- Performing advanced data processing operations, including outlier removal and missing data imputation, using the previously defined functions.
- Merging the processed control group and PKU patients data into a single dataframe, then saving the dataframe as a Parquet file with Gzip compression.

## Usage 

To use this script, follow these steps:

1. Install the necessary R packages: `tidyverse`, `magrittr`, `ggbeeswarm`, `ggpubr`, `mixgb`, `readxl`, and `arrow`.
2. Set the file paths of the data files for control group and PKU patients in the "Data Loading and Cleaning" section.
3. Set the output file path in the "Data Merging and Saving" section.
4. Run the script in R.

# Step 02: Class Balancing via Oversampling.

`step02_classes_balancing_via_oversampling.py`, is used for data augmentation and balancing using Synthetic Minority Over-sampling Technique (SMOTE) in an imbalanced dataset. 

## Functionality

The script performs the following operations:

1. Reads in a dataset from a specified Parquet file.
2. Checks if the 'Group' column is present in the DataFrame.
3. Identifies all feature columns of type float64.
4. Converts the categorical 'Group' labels into numerical form using LabelEncoder from the `sklearn` library, creating the target vector.
5. Initializes a SMOTE instance for oversampling the minority class.
6. Uses SMOTE to balance the class distribution.
7. Creates a new DataFrame from the resampled data.
8. Saves this new DataFrame as a Parquet file with Gzip compression.

## Usage

The script requires Python 3 and the following Python libraries: `pandas`, `sklearn`, and `imblearn`. Install these libraries using pip:

# Step03 Flux Sampling Script Explanation

This script, named `step03_flux_sampling.py`, uses the COBRApy (COnstraints-Based Reconstruction and Analysis) toolbox to perform flux sampling on a metabolic model under two different conditions: healthy and PKU (Phenylketonuria). 

## Functionality

This script does the following:

1. Reads a metabolic model from a `.mat` file.
2. Prints basic information about the model, such as the type, number of reactions, metabolites, and compartments.
3. Adjusts the bounds of certain reactions in the model.
4. Sets the objective of the model.
5. Turns off certain reactions associated with specific metabolites.
6. Obtains regulatory reactions and sets their bounds.
7. Initializes an instance of OptGPSampler from COBRA.
8. Samples the solution space under both healthy and PKU conditions, generating 20,000 samples for each.
9. Saves these samples to `.parquet.gzip` files.

## Usage

You will need Python 3 installed, along with the `os`, `cobra`, `warnings`, `itertools`, and `numpy` Python libraries. You can install these libraries using pip.

# Step04: Flux Samples Preprocessing.

The script `step04_flux_samples_preprocessing.R` is responsible for preprocessing flux samples obtained from metabolic models under two conditions: healthy (CONTROL) and PKU (Phenylketonuria).

## Functionality

This script carries out the following operations:

1. Checks if a given row in the dataframe contains an "out" value. 
2. Identifies and labels outliers in a column as "out".
3. Removes rows containing outliers from a dataframe.
4. Reads the flux samples dataframes stored in parquet files.
5. Performs outlier removal on the CONTROL and PKU samples.
6. Writes back the cleaned flux samples dataframes to new parquet files.

## Usage

This script requires R and the following R libraries: `magrittr`, `tidyverse`, and `arrow`. 
