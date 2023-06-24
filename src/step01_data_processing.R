# ---------------------------------------------------------------------
# 1. LIBRARY IMPORTS
# ---------------------------------------------------------------------

# Load the required libraries
library(tidyverse)   # For data manipulation and visualization
library(magrittr)    # For pipe operators
library(ggbeeswarm)  # For scatter plot with non-overlapping points
library(ggpubr)      # For creating publication ready plots
library(mixgb)       # For data imputation
library(readxl)      # For reading Excel files

# ---------------------------------------------------------------------
# 2. FUNCTIONS
# ---------------------------------------------------------------------

# Function to check whether a row contains the term "out"
check_in <- function(a_row) {
  return("out" %in% a_row) 
}

# Function to find occurrence of "out" in a dataframe
find_out_occurences <- function(df) {
  return(sapply(as.list(as.data.frame(t(df))), check_in) %>% as.vector())
}

# Function to replace outliers in a column with "out"
outliers_to_str <- function(a_col, first_qutl, second_qutl) {
  Q1 <- quantile(a_col, first_qutl, names = FALSE, na.rm = TRUE)
  Q3 <- quantile(a_col, second_qutl, names = FALSE, na.rm = TRUE)
  INTER <- Q3 - Q1
  lower_bound <- Q1 - 1.5 * INTER
  upper_bound <- Q3 + 1.5 * INTER
  a_col[a_col < lower_bound | a_col > upper_bound] <- "out"
  return(a_col)
}

# Function to remove outliers from a dataset
remove_outliers_patients <- function(df, first_qutl, second_qutl) {
  df %>% 
    select(-c(Phe)) %>%  # Exclude 'Phe' column from outlier detection
    mutate_all(outliers_to_str, first_qutl, second_qutl) %>%  # Replace outliers with "out" in all columns
    find_out_occurences -> out_ocurrences  # Identify rows containing "out"
  return(df[!out_ocurrences, ])  # Return the dataframe excluding rows with "out"
}

# Function to impute missing data using the mixgb method
impute_data_with_mixgb <- function(df) {
  for_mixgb <- data_clean(df)  # Clean the data prior to imputation
  names(for_mixgb) <- make.names(colnames(for_mixgb))  # Make column names suitable for 'mixgb'
  imputed.data <- mixgb(data = for_mixgb, m = 1, nthread = 20, pmm.k = 10, initial.num = 'median')  # Perform imputation
  imputed_data <- imputed.data[[1]] %>% as.data.frame()  # Convert the result to dataframe
  return(imputed_data)  # Return the imputed data
}

# ---------------------------------------------------------------------
# 3. DATA LOADING AND CLEANING
# ---------------------------------------------------------------------
# # Load and clean the CONTROL data
# Define the path of the data files of control patients
data_file_controls <- file.path(".", "metabolite_raw_data", "raw_data_to_extract_CONTROLS.csv")

# Load the data from the defined path
df1 <- read_csv2(data_file_controls) %>% 
  # Rename the columns 'Glt' and 'tir' to 'Glu' and 'Tyr' for better readability
  rename(Glu = Glt, Tyr = tir) %>%  
  # Select all columns except 'SA'
  select(-c("SA"))  

# Specify the variables in which rows with missing values will be dropped
vars_to_drop_NAs <- c("Fecha entrega informe","Fecha de nacimiento","Phe","Tyr", "Edad (días calculados)")

# Clean the control group data by:
healthy <- df1 %>% 
  # Remove rows with missing values in the specified variables
  drop_na(any_of(vars_to_drop_NAs)) %>%  
  # Filter out rows where 'Tyr' is less than or equal to zero
  filter(0 < Tyr) %>% 
  # Filter out rows where 'Phe' is less than 35 or greater than 120
  filter(35 <= Phe & Phe <= 120) %>%  
  # Filter out rows where 'Edad (días calculados)' is greater than 31
  filter(`Edad (días calculados)` <= 31) %>% 
  # Select all columns except 'FDIAG'
  select(-c("FDIAG")) %>% 
  # Convert the column 'Fecha entrega informe' from character to Date type
  mutate(`Fecha entrega informe` = as.Date(`Fecha entrega informe`,"%d-%m-%Y")) %>%  
  # Convert the column 'Fecha de nacimiento' from character to Date type
  mutate(`Fecha de nacimiento` = as.Date(`Fecha de nacimiento`,"%d-%m-%Y")) %>% 
  # Filter out rows where 'Fecha entrega informe' is earlier than '2009-01-01'
  filter(`Fecha entrega informe` > '2009-01-01')

# Define the columns that are to be dropped from the dataframe
to_drop <- c("RUT","Fecha entrega informe", "Fecha de nacimiento", "estudio molecular", "Ciudad", "Sexo", 
             "Edad (días calculados)", "FADIAG (mg por dL)", "TIRDIAG mg_por_DL" , "razon_FADIAG_TIRDIAG" , "COMUNA",
             "REGION" , "TDIAG")

# Filter only numeric columns from the 'healthy' dataframe for further processing
healthy_numeric <- healthy %>% 
  # Select all columns except those specified in 'to_drop'
  select(-c(to_drop)) %>%  
  # Select columns that are of type double
  select_if(is_double)

# # Load and clean the PKU data
# Define the path of the data file for PKU patients
data_xlsx_file_PKUs <- file.path(".", "metabolite_raw_data", "raw_data_to_extract_PKUs.xlsx")

# Load the PKU data from the defined path and remove irrelevant columns
# '...1' and 'SA' are removed for this particular analysis
df1_pku <- read_excel(data_xlsx_file_PKUs) %>% 
  select(-c("...1", "SA"))  

# Preprocess the PKU data
# PKU data cleaning and processing involves:
# 1. Filtering patients who were diagnosed at or before 31 days old
# 2. Converting 'Fecha.entrega.informe' and 'Fecha.Nacimiento' from character to Date type
# 3. Filtering out patients whose 'Phe' level is less than 360
# 4. Filtering out data collected before 2009
PKU <- df1_pku %>% 
  filter(edad_diagnostico_dias <= 31) %>% 
  mutate(`Fecha.entrega.informe` = as.Date(`Fecha.entrega.informe`, "%d-%m-%Y")) %>% 
  mutate(`Fecha.Nacimiento` = as.Date(`Fecha.Nacimiento`, "%d-%m-%Y")) %>% 
  filter(Phe >= 360) %>% 
  filter(`Fecha.entrega.informe` > '2009-01-01') 

# Define columns to be dropped from the PKU dataset
# These columns are not relevant to the specific analysis being performed
to_drop <- c("Paciente_PKU", "COMUNA", "REGION", "Condicion", "RUT", "Fecha.entrega.informe", "Fecha.Nacimiento", "estudio.molecular",
             "Ciudad", "Sexo", "edad_diagnostico_dias", "FADIAG..mg.por.dL.", "TIRDIAG.mg_por_dL", "razon_FADIAG_TIRDIAG" ,"TDIAG")

# Drop unwanted columns from the PKU dataset and keep only numeric columns for further processing
PKU_numeric <- PKU %>% 
  select(-c(to_drop)) %>% 
  select_if(is_double)
# # ---------------------------------------------------------------------
# # 4. DATA PROCESSING
# # ---------------------------------------------------------------------

# Remove outliers from healthy data and print remaining rows
healthy_outliers_removed <- remove_outliers_patients(healthy_numeric, .25, .75)
print(paste('remaining healthy patients: ', nrow(healthy_outliers_removed)))

# Impute healthy data
healthy_outliers_inputation_done <- impute_data_with_mixgb(healthy_outliers_removed)

# Remove outliers from PKU data and print remaining rows
PKU_outliers_removed <- remove_outliers_patients(PKU_numeric, 0.1, .9)
print(paste('remaining PKUs: ', nrow(PKU_outliers_removed)))

# Impute PKU data
PKU_outliers_inputation_done <- impute_data_with_mixgb(PKU_outliers_removed)

# # ---------------------------------------------------------------------
# # 5. DATA MERGING AND SAVING
# # ---------------------------------------------------------------------

# Add Group variable to both datasets
healthy_outliers_inputation_done['Group'] <- "Control"
PKU_outliers_inputation_done['Group'] <- "PKU"

# Check if both datasets have the same column names
assertthat::assert_that(all(names(PKU_outliers_inputation_done) == names(healthy_outliers_inputation_done)))

# Combine both datasets
combined_data <- rbind(PKU_outliers_inputation_done, healthy_outliers_inputation_done)

# Define the output file path
output_file_path <- file.path(".", "processed_data", "metabolites_outliers_removed_and_imputed.parquet.gzip")

# Save the combined data as a parquet file with gzip compression
arrow::write_parquet(combined_data, output_file_path, compression = "gzip")

# Print the output file path
print(paste("The combined data has been saved to: ", output_file_path))