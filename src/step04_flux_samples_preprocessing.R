# step04_flux_samples_preprocessing.R
library(magrittr)
library(tidyverse)

check_in            <- function(a_row){return("out" %in% a_row) }
find_out_occurences <- function(df){return(sapply(as.list(as.data.frame(t(df))), check_in) %>% as.vector())}
outliers_to_str     <- function(a_col, first_qutl, second_qutl){
  Q1 <- quantile(a_col, first_qutl, names = FALSE, na.rm =T)
  Q3 <- quantile(a_col, second_qutl, names = FALSE, na.rm =T)
  INTER <- Q3 - Q1
  lower_bound <- Q1 - 1.5*INTER
  upper_bound <- Q3 + 1.5*INTER
  a_col[a_col < lower_bound | a_col > upper_bound] <- "out"
  return(a_col)}


remove_outliers_patients <- function(df, first_qutl, second_qutl){
  df  %>% # dplyr::select(-c(Phe)) %>% 
    mutate_all(outliers_to_str, first_qutl, second_qutl)  %>% 
    find_out_occurences  -> out_ocurrences
  return(df[!out_ocurrences,])}

flux_samples_CONTROL_20_000.parquet.gzip <- file.path(".", "results", "fluxes", "flux_samples_CONTROL_20_000.parquet.gzip")
flux_samples_PKU_20_000.parquet.gzip     <- file.path(".", "results", "fluxes", "flux_samples_PKU_20_000.parquet.gzip")

flux_samples_CONTROL_10_000 <- arrow::read_parquet(flux_samples_CONTROL_20_000.parquet.gzip) %>% remove_outliers_patients(0.01, .99)
flux_samples_PKU_10_000 <- arrow::read_parquet(flux_samples_PKU_20_000.parquet.gzip)%>% remove_outliers_patients(0.01, .99)
# 
arrow::write_parquet(flux_samples_CONTROL_10_000, "./results/fluxes/CLEANED_flux_samples_CONTROL_20_000.parquet.gzip" ,compression = "gzip")
arrow::write_parquet(flux_samples_PKU_10_000, "./results/fluxes/CLEANED_flux_samples_PKU_20_000.parquet.gzip",     compression = "gzip")
# head(flux_samples_PKU_10_000)
print('done!')