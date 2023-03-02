
library("magrittr")
library("DESeq2")

# ----
# Descarga el archivo si no existe
URL_geo <- "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE148349&format=file"
GEO_tarfile <- "GSE148349_RAW.tar"

if (!file.exists(GEO_tarfile)) {
    download.file(URL_geo, GEO_tarfile)

    # Crea un directorio "archive" para descomprimir las cosas
    if (!dir.exists("archive")) {
        dir.create("archive")
    }

    untar(GEO_tarfile, exdir = "archive")
}

# -----
# Lista de muestras, constuida manualmente
sample_info <- read.delim2("samples_mouse.csv", sep = ",")
PAH_KO <- sample_info$replicated[sample_info$treatment == "PAH_KO"] # PKU
PAH_re <- sample_info$replicated[sample_info$treatment != "PAH_KO"] # Control

# -----
# Crea una matriz de conteos desde "GSE148349_RAW.tar", que contiene todas las
# matrices de conteos de cada muestra, como descargadas de la pagina de NCBI
archive_files <- list.files("archive/")

df <- list()

# Este loop genera una lista de conteos, donde cada columna es una muestra y cada
# fila es un ENSMUST (transcrito de raton). Necesita una reduccion para ser un df
for (sample in c(PAH_KO, PAH_re)) {
    sample_file <- archive_files[startsWith(archive_files, sample)]
    sample_file <- paste0("archive/", sample_file)

    df[[sample]] <- read.delim2(
        sample_file,
        sep = "\t", row.names = 1
    )["est_counts"]

    colnames(df[[sample]]) <- c(sample)
}

df %<>% data.frame # Convierte la lista en un dataframe propiamente tal

# -----
# Celulas HEK293T knock-out para PAH
# HEK293T PAH_KO <- c("GSM4462274","GSM4462275","GSM4462276")

# Estos son tratados para re-establecer PAH
# HEK293T PAH_re <- c("GSM4462268","GSM4462270","GSM4462272","GSM4462269","GSM4462271","GSM4462273")

# -----
# Convierte todos los GSMs en integros
df[] <- lapply(df, as.integer)

# Crea una matriz de conteos
mx <- df[c(PAH_re, PAH_KO)] %>% as.matrix()

sample_info$treatment <- as.factor(sample_info$treatment)

# -----
# Cosas de DESeq2
# requieren paralelizacion

dds <- DESeqDataSetFromMatrix(
    countData = mx,
    colData = sample_info,
    design = ~treatment
)

# Corre el pipeline de analisis estimando modelos
dds <- DESeq(
    dds,
    parallel = TRUE
)

res <- results(
    dds,
    contrast = c("treatment", "PAH_KO", "PAH_reestablecido"),
    parallel = TRUE
)

# Guarda el objeto
saveRDS(res, "RNA_results.RDS")
