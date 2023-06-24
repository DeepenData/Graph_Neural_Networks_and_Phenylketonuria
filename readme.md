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

