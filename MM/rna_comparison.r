
# Lee los resultados del RNAseq
res <- readRDS("RNA_results.RDS")

# -----
# Carga un helper desde las funciones de 
# (en caso de no estar cargada)
library("S4Vectors")
library("magrittr")

df <- read.delim2("human_reads.tsv", sep = "\t", row.names = 1)

# Funcion que toma una lista de entrez y retorna un genesymbol
gene_symbol <- df$gene_symbol %>% DataFrame
rownames( gene_symbol ) <- df$entrez_id %>% as.character
entrez_to_gene <- function ( lista ) { gene_symbol[ lista ,] }

# Conjuntos
gs_C  <- read.csv2("entrez_IDs_C.csv")[,1]  %>% as.character %>% entrez_to_gene
gs_F  <- read.csv2("entrez_IDs_F.csv")[,1]  %>% as.character %>% entrez_to_gene
gs_CF <- read.csv2("entrez_IDs_CF.csv")[,1] %>% as.character %>% entrez_to_gene

conjuntos <- list(
    Concentrations=gs_C,
    Fluxes=gs_F,
    Concentrations_Fluxes=gs_CF
)

# -----
library("EnhancedVolcano")

pValue_cutoff     = 0.005
FoldChnage_cutoff = 2.0

df_res <- cbind( res, gene_symbol=df$gene_symbol ) %>% as.data.frame
df_res <- cbind( df_res, gene_symbol_sig=ifelse( 
                abs(df_res$log2FoldChange) > FoldChnage_cutoff, 
                df_res$gene_symbol, "NoSignificativo"
            )) %>% as.data.frame

only_genes_changed <- function ( list_genes ) {
    filter(  df_res , 
    gene_symbol %in% list_genes , 
    abs(log2FoldChange) > FoldChnage_cutoff, 
    pvalue < pValue_cutoff )$gene_symbol %>% 
    unique
}

for ( i in 1:length(conjuntos)) {
    plot <- EnhancedVolcano(
        df_res, 
        x = 'log2FoldChange',
        y = 'pvalue',
        # y = 'padj',
        title = 'Cambio de expreson PAH-reestablecido → PAH-KO',
        subtitle = paste0( names( conjuntos )[[i]], " - ", length( unique(conjuntos[[i]][ conjuntos[[i]] %in% df_res$gene_symbol_sig ]) ), " transcritos" ),
        pCutoff = pValue_cutoff,
        FCcutoff = FoldChnage_cutoff,
        pointSize = 0.4,
        
        # Cosas de los labels
        # lab = df_res$gene_symbol,
        # lab = "gene_symbol_sig",
        lab = df_res$gene_symbol_sig,
        # selectLab = unique(conjuntos[[i]][ conjuntos[[i]] %in% df_res$gene_symbol_sig ]),
        selectLab = c('TMEM176B','ADH1A'),
        labSize = 2.5,
        boxedLabels = TRUE,
        drawConnectors = TRUE,
        widthConnectors = 0.2,
        colConnectors = 'black',
        max.overlaps = 100, #Inf,

    ) + ggthemes::theme_few() + 
    # coord_flip() + 
    theme(legend.position="bottom")

    # Guarda el grafico
    ggsave( plot = plot, filename = paste0("volcano_", names( conjuntos )[[i]] ,".png") )
}

# ----
# Diagramas VENN
library("VennDiagram")
 
# ROBADO
# Prepare a palette of 3 colors with R colorbrewer:
library("RColorBrewer")
colour_palette <- brewer.pal(3, "Pastel2")

# Diagrama
venn.diagram(
        x = list(gs_C, gs_F, gs_CF),
        category.names = c("Concentrations" , "Fluxes" , "Confluxtration"),
        euler.d = TRUE, scaled = TRUE,

        filename = 'venn_diagramm.png',
        output=TRUE,
        
        # Output features
        imagetype="png" ,
        height = 480 , 
        width = 480 , 
        resolution = 300,
        compression = "lzw",
        
        # Circles
        lwd = 2,
        lty = 'blank',
        fill = colour_palette,
        
        # Numbers
        cex = .6,
        fontface = "bold",
        fontfamily = "sans",
        
        # Set names
        cat.cex = 0.6,
        cat.fontface = "bold",
        cat.default.pos = "outer",
        cat.pos = c(-27, 27, 135),
        cat.dist = c(0.055, 0.055, 0.085),
        cat.fontfamily = "sans",
        rotation = 1
)

# ----- 
# Un diagrama mas informativo pero que no se puede guardar
library(nVennR)
nvenn <- plotVenn(conjuntos, nCycles = 2000)
showSVG(nVennObj = nvenn, opacity = 0.1, borderWidth = 3)
