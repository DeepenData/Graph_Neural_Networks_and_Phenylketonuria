---
title: "R Notebook"
output: html_notebook
---


```r
library(arrow)
library(tidyverse)
library(ComplexHeatmap) 
library(scales)
library(ggpubr)
library(ggplot2)
library(DOSE)
library(magrittr)
library(biomaRt)
library(tidyverse)
library(clusterProfiler)
library(ReactomePA)
library(org.Hs.eg.db)
library(fastcluster)
```



```r
setwd("/DeepenData/Repos/geometric_cobra")
Cluster_genes_concentration <- read_csv( "./results/dataframes/Cluster_genes_concentration.csv")
Cluster_genes_flux <- read_csv( "./results/dataframes/Cluster_genes_flux.csv")
Cluster_genes_concentration_plus_flux <- read_csv( "./results/dataframes/Cluster_genes_concentration_plus_flux.csv")
```

```r
get_gene_cluster<- function(df, a_cluster){
                  genes <-  df %>%  dplyr::filter(Cluster == a_cluster)   %>%  .[['genes']]  %>% str_extract_all('\\d+(?=\\.)') %>% unlist() %>% unique()
                  return(genes
                       
                        )}


get_clusters_list <- function(df){
                                clusters_list <- list(
                              "1" = get_gene_cluster(df, 1),
                              "2" = get_gene_cluster(df, 2))
                              return(clusters_list)
                              }



get_clusters_list(Cluster_genes_concentration)           -> concentration
get_clusters_list(Cluster_genes_flux)                    -> flux
get_clusters_list(Cluster_genes_concentration_plus_flux) -> concentration_plus_flux
```



```r
ensembl    <- useMart("ensembl")
Hs.ensembl <- useMart("ensembl",dataset="hsapiens_gene_ensembl")
genes_symbols <- list()
get_functional_annots <- function(entrez_IDs, transporters){
                                   temp <- getBM(attributes=c('hgnc_symbol',"entrezgene_id", 'entrezgene_description'),filters ='entrezgene_id',values =entrez_IDs, mart = Hs.ensembl)
                                   if(transporters){
                                     temp %<>% filter(str_detect(entrezgene_description,regex( '(solute.+?carrier)|transporter|aquapor|transport', ignore_case = T))) 
                                   }
                                   else {  temp %<>% filter(!str_detect(entrezgene_description,regex( '(solute.+?carrier)|transporter|aquapor|transport', ignore_case = T))) }
                                   
                                   
                                   
                                   return(arrange(temp, hgnc_symbol) %>% purrr::set_names(c("Symbol", "Entrez ID", "Description")))}


get_transporters_and_reactions <- function(clusters_list){
  transporters <- purrr::map2(clusters_list, c(T,T), get_functional_annots)
reactions    <- purrr::map2(clusters_list, c(F,F), get_functional_annots)
transporters_and_reactions <- list('reactions'=reactions,'transporters'=transporters)

purrr::map2(transporters_and_reactions$reactions, c("Entrez ID","Entrez ID"), pull) -> reaction_genes
purrr::map2(transporters_and_reactions$transporters, c("Entrez ID","Entrez ID"), pull) -> transporter_genes

return(list("reactions"= reaction_genes,  'transporters'=transporter_genes))
  
}
```



```r
get_transporters_and_reactions(concentration) -> transporters_and_reactions_concentration
get_transporters_and_reactions(flux) -> transporters_and_reactions_flux
get_transporters_and_reactions(concentration_plus_flux) -> transporters_and_reactions_concentration_plus_flux
```



```r
setwd("/DeepenData/Repos/geometric_cobra")
transporters_and_reactions_concentration %>% unlist %>% as.vector() %>% data.frame("C"= .) %>% write_csv("./results/dataframes/entrez_IDs_C.csv")

transporters_and_reactions_flux %>% unlist %>% as.vector()  %>% data.frame("F"= .) %>% write_csv("./results/dataframes/entrez_IDs_F.csv")

transporters_and_reactions_concentration_plus_flux%>% unlist %>% as.vector() %>% data.frame("CF"= .)  %>% write_csv("./results/dataframes/entrez_IDs_CF.csv")
```

-----

# Transporters_and_reactions_concentration



```r
transporters_and_reactions_concentration$reactions %>% 
  compareCluster(fun='enrichPathway', pvalueCutoff=1e-2, qvalueCutoff=1e-2)-> enrich_output
enrich_output %>% dotplot(showCategory = 10, font.size = 5, label_format =90)+
   theme( 
    legend.text=element_text(size=6),
    legend.position = "bottom",
    #legend.box = "vertical",
    text = element_text(size=6)) -> concentration_reactions_enrichPathway
```



```r
transporters_and_reactions_concentration$transporters %>% 
  compareCluster(fun='enrichPathway', pvalueCutoff=1e-5, qvalueCutoff=1e-5)-> enrich_output
enrich_output %>% dotplot(showCategory = 10, font.size = 5, label_format =90)+
   theme( 
    legend.text=element_text(size=6),
    legend.position = "bottom",
    #legend.box = "vertical",
    text = element_text(size=6)) -> concentration_transporters_enrichPathway
```



```r
transporters_and_reactions_concentration$reactions %>% compareCluster(fun='enrichDGN', pvalueCutoff=5e-2,  qvalueCutoff=5e-2)-> enrich_output
enrich_output %>% dotplot(showCategory = 5, font.size = 6, label_format =80)+
   theme( 
    legend.text=element_text(size=6),
    legend.position = "bottom",
    #legend.box = "vertical",
    text = element_text(size=6))  -> concentration_reactions_enrichDGN
```





```r
transporters_and_reactions_concentration$transporters %>% compareCluster(fun='enrichDGN', pvalueCutoff=5e-2,  qvalueCutoff=5e-2)-> enrich_output
enrich_output %>% dotplot(showCategory = 5, font.size = 6, label_format =80)+
   theme( 
    legend.text=element_text(size=6),
    legend.position = "bottom",
    #legend.box = "vertical",
    text = element_text(size=6))  -> concentration_transporters_enrichDGN
```




```r
concentration_panel <- ggarrange(concentration_reactions_enrichPathway, concentration_transporters_enrichPathway, concentration_reactions_enrichDGN, concentration_transporters_enrichDGN, common.legend = T, labels = c("a", "b", "c", "d"), nrow = 1)
```

##########################################
################---transporters_and_reactions_flux---#########################
##########################################



```r
transporters_and_reactions_flux$reactions %>% 
  compareCluster(fun='enrichPathway', pvalueCutoff=1e-3, qvalueCutoff=1e-3)-> enrich_output
enrich_output %>% dotplot(showCategory = 10, font.size = 5, label_format =90)+
   theme( 
    legend.text=element_text(size=6),
    legend.position = "bottom",
    #legend.box = "vertical",
    text = element_text(size=6)) -> flux_reactions_enrichPathway
```



```r
transporters_and_reactions_flux$transporters %>% 
  compareCluster(fun='enrichPathway', pvalueCutoff=1e-5, qvalueCutoff=1e-5)-> enrich_output
enrich_output %>% dotplot(showCategory = 10, font.size = 5, label_format =90)+
   theme( 
    legend.text=element_text(size=6),
    legend.position = "bottom",
    #legend.box = "vertical",
    text = element_text(size=6))-> flux_transporters_enrichPathway
```



```r
transporters_and_reactions_flux$reactions %>% compareCluster(fun='enrichDGN', pvalueCutoff=5e-2,  qvalueCutoff=5e-2)-> enrich_output
enrich_output %>% dotplot(showCategory = 5, font.size = 6, label_format =80)+
   theme( 
    legend.text=element_text(size=6),
    legend.position = "bottom",
    #legend.box = "vertical",
    text = element_text(size=6))-> flux_reactions_enrichDGN
```





```r
transporters_and_reactions_flux$transporters %>% compareCluster(fun='enrichDGN', pvalueCutoff=5e-2,  qvalueCutoff=5e-2)-> enrich_output
enrich_output %>% dotplot(showCategory = 5, font.size = 6, label_format =80)+
   theme( 
    legend.text=element_text(size=6),
    legend.position = "bottom",
    #legend.box = "vertical",
    text = element_text(size=6)) -> flux_transporters_enrichDGN
```


```r
flux_panel <- ggarrange(flux_reactions_enrichPathway, flux_transporters_enrichPathway, flux_reactions_enrichDGN, flux_transporters_enrichDGN, common.legend = T, labels = c("e", "f", "g", "h"), nrow = 1)
```
##########################################
################---transporters_and_reactions_concentration_plus_flux---#########################
##########################################

```r
transporters_and_reactions_concentration_plus_flux$reactions %>% 
  compareCluster(fun='enrichPathway', pvalueCutoff=5e-2, qvalueCutoff=5e-2)-> enrich_output
enrich_output %>% dotplot(showCategory = 10, font.size = 5, label_format =90)+
   theme( 
    legend.text=element_text(size=6),
    legend.position = "bottom",
    #legend.box = "vertical",
    text = element_text(size=6)) ->concentration_plus_flux_reactions_enrichPathway
```



```r
transporters_and_reactions_concentration_plus_flux$transporters %>% 
  compareCluster(fun='enrichPathway', pvalueCutoff=1e-2, qvalueCutoff=1e-2)-> enrich_output
enrich_output %>% dotplot(showCategory = 10, font.size = 5, label_format =90)+
   theme( 
    legend.text=element_text(size=6),
    legend.position = "bottom",
    #legend.box = "vertical",
    text = element_text(size=6)) -> concentration_plus_flux_transporters_enrichPathway
```



```r
transporters_and_reactions_concentration_plus_flux$reactions %>% compareCluster(fun='enrichDGN', pvalueCutoff=5e-2,  qvalueCutoff=5e-2)-> enrich_output
enrich_output %>% dotplot(showCategory = 5, font.size = 6, label_format =80)+
   theme( 
    legend.text=element_text(size=6),
    legend.position = "bottom",
    #legend.box = "vertical",
    text = element_text(size=6)) -> concentration_plus_flux_reactions_enrichDGN
```





```r
transporters_and_reactions_concentration_plus_flux$transporters %>% compareCluster(fun='enrichDGN', pvalueCutoff=5e-2,  qvalueCutoff=5e-2)-> enrich_output
enrich_output %>% dotplot(showCategory = 10, font.size = 6, label_format =80)+
   theme( 
    legend.text=element_text(size=6),
    legend.position = "bottom",
    #legend.box = "vertical",
    text = element_text(size=6)) -> concentration_plus_flux_transporters_enrichDGN
```


```r
concentration_flux_panel <- ggarrange(
  concentration_plus_flux_reactions_enrichPathway, concentration_plus_flux_transporters_enrichPathway, concentration_plus_flux_reactions_enrichDGN, concentration_plus_flux_transporters_enrichDGN, common.legend = T, labels = c("i", "j", "k", "l"), nrow = 1)
```



```r
ggarrange(concentration_panel, flux_panel, concentration_flux_panel, nrow = 3) -> full_panel
```



```r
setwd("/DeepenData/Repos/geometric_cobra")
ggsave("./results/figures/figure_4.png", full_panel, height = 7, width = 14, bg="white")
```






































