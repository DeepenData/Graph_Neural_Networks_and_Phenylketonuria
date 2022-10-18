library(tidyverse)      
library(ComplexHeatmap) 
#library(EnhancedVolcano)
#library(knitr)
#library(gridExtra) # to display a table
#library(latex2exp) # for TeX function which transform $$ to expression
library(PCAtools)
#library(devtools)
library(ggpubr)
library(ggplot2)
#library(ggplotify)
library(magrittr)
library(biomaRt)
library(tidyverse)
library(clusterProfiler)
library(ReactomePA)
library(org.Hs.eg.db)
#library(BiocGenerics)
library(reticulate)
library(umap)
data         <- read_csv("./results/data/results_for_R.csv") %>% dplyr::select(-c("...1")) #%>% dplyr::select(-c('0'))# %>% as.matrix()

data_matrix  <- data %>% select_if(is.numeric) 
#data_matrix <- data_matrix[,-1] 
colSums(data_matrix)
rowSums(data_matrix) -> edge_scores



edge_mask <- edge_scores > min(edge_scores); sum(edge_mask)
genes_mask <- data$genes != "['']";sum(genes_mask)
mask <- edge_mask & genes_mask; sum(mask)

processed_data <- as.matrix(data_matrix)[mask,]#%>% t 

right_annotation <-
      rowAnnotation(gap = unit(6, "points"),
      Ce = anno_barplot(bar_width = 0.02,width = unit(.7, "cm"),  
      border = T, edge_scores[mask], gp = gpar(col = 'black', fill = "green"))) 

suppressMessages(ht <- Heatmap(processed_data, row_km = 4))


ggsave(file.path(getwd(),'./results/figures/Heatmap_conected_Recon.png'), 
                  ggplotify::as.ggplot(ht), dpi = 90, width = 3, height = 19 )


heatmap_clusters <- waRRior::heatmap_extract_cluster(draw(ht), data_matrix[mask,],  which = "row")
edges_clusters   <-  heatmap_clusters[['Cluster']]


sum(edges_clusters == "1")
sum(edges_clusters == "2")
sum(edges_clusters == "3")
sum(edges_clusters == "4")
a_cluster = "4"

data_cluster_genes <- data[mask,][edges_clusters == a_cluster,]
view(data_cluster_genes)
###################################################################################
ensembl    <- useMart("ensembl")
Hs.ensembl <- useMart("ensembl",dataset="hsapiens_gene_ensembl")


genes_list <- data_cluster_genes %>% mutate(genes = genes %>% str_extract_all('\\d+(?=\\.)')) %>% .[['genes']] 

genes_symbols <- list()
idx           <- 0 

for(i in genes_list){
    idx          <- idx + 1
    genes        <- i  %>% str_extract_all('\\d+')%>% unlist()

    if(is.null(genes)){
                        genes_symbols[[idx]] <- ''
                        next} 

    temp         <- getBM(attributes=c('hgnc_symbol'),filters ='entrezgene_id',values = genes, mart = Hs.ensembl)
    names(temp)  <- NULL
    genes_symbols[[idx]] <- unlist(temp)
}

testit::assert(nrow(data_cluster_genes) == length(genes_symbols))
genes_symbols_col <- genes_symbols %>%  map(~ paste(.,collapse = ", ")) %>% unlist() %>% data.frame()


testit::assert(nrow(data_cluster_genes) == nrow(genes_symbols_col))

node1_node2_genes <- cbind(data_cluster_genes[c("node1", "node2")], genes_symbols_col) %>% purrr::set_names(c("Node 1", "Node 2", "Genes")) %>% dplyr::filter(Genes!="")
stable.p      <- node1_node2_genes %>% ggtexttable(rows = NULL,theme = ttheme("mOrange"))


ggsave('./results/figures/table_rxns_and_central_genes_symbols.png', stable.p, dpi = 80, width = 10, height = 43)

###################################################################################
###################################################################################
entrez_IDs <- genes_list%>% unlist() %>% unique()

list(entrez_IDs, entrez_IDs) %>% purrr::set_names(c("ff", "ll"))%>%compareCluster(fun='enrichPathway', pvalueCutoff=1e-2) -> panel_compareCluster
panel_compareCluster %>% dotplot -> enrichPathway_dotplot
ggsave(file.path(getwd(),'./results/figures/enrichPathway_dotplot.png'), enrichPathway_dotplot, dpi = 110, width = 7, height = 5 )



list(entrez_IDs, entrez_IDs)%>% purrr::set_names(c("ff", "ll"))%>%compareCluster(fun='enrichKEGG', pvalueCutoff=1e-2) -> panel_compareCluster
panel_compareCluster %>% dotplot -> enrichKEGG_dotplot
ggsave(file.path(getwd(),'./results/figures/enrichKEGG_dotplot.png'), enrichKEGG_dotplot, dpi = 110, width = 7, height = 5 )


library(DOSE)
list(entrez_IDs, entrez_IDs)%>% purrr::set_names(c("ff", "ll"))%>%compareCluster(fun='enrichDGN', pvalueCutoff=1e-5) -> panel_compareCluster
panel_compareCluster %>% dotplot -> enrichDGN_dotplot
ggsave(file.path(getwd(),'./results/figures/enrichDGN_dotplot.png'), enrichDGN_dotplot, dpi = 110, width = 7, height = 5 )
#########################################
#########################################
#######################################

temp         <- getBM(attributes=c('hgnc_symbol', 'entrezgene_description'),filters ='entrezgene_id',values = entrez_IDs, mart = Hs.ensembl)



#searchAttributes(mart = Hs.ensembl, pattern = 'entrez')
stable.p      <- temp %>% ggtexttable(rows = NULL,theme = ttheme("mOrange"))


ggsave('./results/figures/genes_symbols_and_entrez_description.png', stable.p, dpi = 80, width = 10, height = 49, limitsize = FALSE)
