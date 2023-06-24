# PKU Metabolic Modeling: Pipeline Overview

This repository contains a set of scripts that create a pipeline for metabolic modeling of Phenylketonuria (PKU), an inherited metabolic disorder characterized by an increased concentration of phenylalanine in the blood. The repository is organized as follows:

## Folder Structure

The repository has the following directory structure:

```
|-- src
|   |-- step01_data_processing.R
|   |-- step02_classes_balancing_via_oversampling.py
|   |-- step03_flux_sampling.py
|   |-- step04_flux_samples_preprocessing.R
```

- The `src` directory contains all the scripts that constitute the PKU metabolic modeling pipeline.
- The `results` directory should be created before running the scripts. It is the location where all output files from the scripts are stored.
- The `data` directory should contain all necessary data files and metabolic models, required for running the scripts.

## Step 01: Data Processing 

The first script, `step01_data_processing.R`, serves as a data preprocessing pipeline designed specifically for a dataset that includes control group and PKU patient data. It executes the following operations:

1. Import necessary libraries for data manipulation, visualization, and imputation.
2. Define functions for outlier detection, outlier removal, and missing data imputation.
3. Load and clean data from specified file paths.
4. Perform advanced data processing operations, such as outlier removal and missing data imputation.
5. Merge processed control group and PKU patient data into a single dataframe, then save the dataframe as a Parquet file.

## Step 02: Class Balancing via Oversampling

The second script, `step02_classes_balancing_via_oversampling.py`, addresses class imbalance in the dataset by oversampling the minority class using Synthetic Minority Over-sampling Technique (SMOTE). The operations carried out by this script include:

1. Read a dataset from a specified Parquet file.
2. Check for the presence of the 'Group' column in the DataFrame.
3. Identify all feature columns of type float64.
4. Convert the categorical 'Group' labels into numerical form, creating a target vector.
5. Initialize a SMOTE instance for oversampling the minority class.
6. Apply SMOTE to balance the class distribution.
7. Create a new DataFrame from the resampled data.
8. Save the new DataFrame as a Parquet file.

## Step 03: Flux Sampling

The third script, `step03_flux_sampling.py`, conducts flux sampling on a metabolic model under two different conditions: healthy and PKU. This script accomplishes the following tasks:

1. Reads a metabolic model from a `.mat` file.
2. Provides information about the model.
3. Adjusts the bounds of specific reactions in the model.
4. Sets the objective of the model.
5. Disables certain reactions associated with specific metabolites.
6. Retrieves regulatory reactions and sets their bounds.
7. Initializes an instance of OptGPSampler from COBRA.
8. Samples the solution space under both healthy and PKU conditions.
9. Saves these samples to `.parquet.gzip` files.

## Step 04: Flux Samples Preprocessing

The fourth script, `step04_flux_samples_preprocessing.R`, is responsible for preprocessing the flux samples obtained from the metabolic models. This script carries out the following operations:

1. Checks if a given row in the dataframe contains an "out" value. 
2. Identifies and labels outliers in a column as "out".
3. Removes rows containing outliers from a dataframe.
4. Reads the flux samples dataframes stored in parquet files.
5. Performs outlier removal on the CONTROL and PKU samples.
6. Writes the cleaned flux samples data

frames to new parquet files.

## Requirements and Usage

The pipeline requires Python 3, R, and several dependencies which are outlined in the description of each script. To use the pipeline, set the appropriate file paths in each script, install the necessary dependencies, and run the scripts in the order specified above.