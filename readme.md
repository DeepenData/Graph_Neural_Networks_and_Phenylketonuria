# `step01_data_preprocessing.R` 

This repository contains the R script `step01_data_preprocessing.R`, a data preprocessing pipeline designed for a specific dataset consisting of control group and PKU patients data. 

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
