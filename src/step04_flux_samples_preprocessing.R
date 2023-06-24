# Load necessary libraries
library(magrittr) # For pipe operations
library(tidyverse) # For data manipulation

# Function to check if an "out" value is in a given row
check_in <- function(a_row) {
  return("out" %in% a_row) # Return true if "out" is in the row, otherwise false
}

# Function to find all occurrences of "out" in a dataframe
find_out_occurences <- function(df) {
  # For each row in the transposed dataframe, check if it contains "out", 
  # convert the result to a vector
  return(sapply(as.list(as.data.frame(t(df))), check_in) %>% as.vector())
}

# Function to identify and mark outliers as "out" in a column
outliers_to_str <- function(a_col, first_qutl, second_qutl){
  # Calculate the first and third quartile
  Q1 <- quantile(a_col, first_qutl, names = FALSE, na.rm =T)
  Q3 <- quantile(a_col, second_qutl, names = FALSE, na.rm =T)
  # Calculate the interquartile range
  INTER <- Q3 - Q1
  # Define lower and upper bound for outliers
  lower_bound <- Q1 - 1.5*INTER
  upper_bound <- Q3 + 1.5*INTER
  # Mark outliers as "out"
  a_col[a_col < lower_bound | a_col > upper_bound] <- "out"
  return(a_col)
}

# Function to remove rows with outliers in a dataframe
remove_outliers_patients <- function(df, first_qutl, second_qutl) {
  # For each column, replace outliers with "out", then find rows containing "out"
  df %>% 
    mutate_all(outliers_to_str, first_qutl, second_qutl) %>% 
    find_out_occurences -> out_ocurrences
  # Return a dataframe without the outlier rows
  return(df[!out_ocurrences,])
}

# Define file paths for input data
flux_samples_CONTROL_20_000.parquet.gzip <- file.path(".", "results", "fluxes", "flux_samples_CONTROL_20_000.parquet.gzip")
flux_samples_PKU_20_000.parquet.gzip <- file.path(".", "results", "fluxes", "flux_samples_PKU_20_000.parquet.gzip")

# Read parquet files, remove outliers, and save cleaned data back to parquet files
# Read CONTROL samples, remove outliers using previously defined function
flux_samples_CONTROL_10_000 <- arrow::read_parquet(flux_samples_CONTROL_20_000.parquet.gzip) %>% remove_outliers_patients(0.01, .99)
# Read PKU samples, remove outliers using previously defined function
flux_samples_PKU_10_000 <- arrow::read_parquet(flux_samples_PKU_20_000.parquet.gzip) %>% remove_outliers_patients(0.01, .99)

# Write cleaned CONTROL samples back to a new parquet file
arrow::write_parquet(flux_samples_CONTROL_10_000, "./results/fluxes/CLEANED_flux_samples_CONTROL_20_000.parquet.gzip" ,compression = "gzip")
# Write cleaned PKU samples back to a new parquet file
arrow::write_parquet(flux_samples_PKU_10_000, "./results/fluxes/CLEANED_flux_samples_PKU_20_000.parquet.gzip",     compression = "gzip")
