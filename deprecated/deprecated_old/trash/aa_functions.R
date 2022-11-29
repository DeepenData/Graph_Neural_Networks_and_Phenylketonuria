#author: "Alejandro Javier Acevedo Aracena, PhD - a.acevedo.aracena@gmail.com"
#date: "May 17th, 2021"
library(tidyverse)
library(ggplot2)
library(ggpubr)
# Functions for generating figure 1: 
plot_density <- function(df, titulo='sin título', fill = "deepskyblue4" , xlab = "Centrality", ylab = "# nodes", xlim = c(-0.04, 0.04)){cut = 0.005
df %>% gather() ->df_gathered
df_gathered['value'] %>% filter(abs(value) > cut) %>%
  ggplot( aes(x=value)) +
  geom_density(fill=fill, color="azure4", alpha=.8)+
  theme(legend.position="top" , plot.title = element_text(size=8.5)) +
  ylab(ylab) + ggtitle(titulo) +  xlim(xlim) +
  xlab(xlab) + geom_vline(xintercept=0, linetype="dashed", color = "black",  size=1) -> Density_plot_of_centralities
return(Density_plot_of_centralities)
}


get_nodes_by_type <- function(data, aggregation_method = "fold_change_geometric"){
  data[[aggregation_method]]-> FC_geometric_only_cents
  FC_geometric_only_cents %>% select(-c("ID","Name","Reaction","Flux", "Sensitivity" )) %>% filter(`Node type` == "Sink/Demand Astro")     -> SiDe_astro
  FC_geometric_only_cents %>% select(-c("ID","Name","Reaction","Flux", "Sensitivity" )) %>% filter(`Node type` == "Sink/Demand Neuron")    -> SiDe_neuron
  FC_geometric_only_cents %>% select(-c("ID","Name","Reaction","Flux", "Sensitivity" )) %>% filter(`Node type` == "Neuron")                -> Neuron
  FC_geometric_only_cents %>% select(-c("ID","Name","Reaction","Flux", "Sensitivity" )) %>% filter(`Node type` == "Astrocyte")             -> Astrocyte
  FC_geometric_only_cents %>% select(-c("ID","Name","Reaction","Flux", "Sensitivity" )) %>% filter(`Node type` == "Exchange")              -> Exchange
  return(list(SiDe_astro,SiDe_neuron,Neuron,Astrocyte,Exchange) %>% purrr::set_names(c('SiDe_astro','SiDe_neuron','Neuron','Astrocyte','Exchange')))
}






filter_centralities <- function(df){df %>% select(matches('^info|^close|^bet|^communi|^katz|^eigen'))}


get_nodes_vs_subsystems <- function(data_by_node_type){
data_by_node_type$Neuron %>% select(matches("astrocyte"))  %>% filter_centralities     ->Neuron_nodes_vs_astrocyte_subsystem
data_by_node_type$Neuron %>% select(matches("neuron"))     %>% filter_centralities     ->Neuron_nodes_vs_neuron_subsystem
data_by_node_type$Astrocyte %>% select(matches("astrocyte"))  %>% filter_centralities  ->Astrocyte_nodes_vs_astrocyte_subsystem
data_by_node_type$Astrocyte %>% select(matches("neuron"))     %>% filter_centralities  ->Astrocyte_nodes_vs_neuron_subsystem
return(list(Neuron_nodes_vs_astrocyte_subsystem,Neuron_nodes_vs_neuron_subsystem,Astrocyte_nodes_vs_astrocyte_subsystem,Astrocyte_nodes_vs_neuron_subsystem) %>% 
purrr::set_names(c('Neuron_nodes_vs_astrocyte_subsystem','Neuron_nodes_vs_neuron_subsystem','Astrocyte_nodes_vs_astrocyte_subsystem','Astrocyte_nodes_vs_neuron_subsystem')))
}

generate_subplots <- function(nodes_vs_subsystems){
  pluck(nodes_vs_subsystems, 'Neuron_nodes_vs_astrocyte_subsystem') %>%
    plot_density(titulo = 'Astrocytic energy metabolism', fill = 'brown4', xlab = NULL, xlim = c(-0.072, 0.07))                                        -> NnAs
  
  pluck(nodes_vs_subsystems,'Neuron_nodes_vs_neuron_subsystem') %>%
    plot_density(titulo = 'Neuronal energy metabolism', fill = 'darkorange4', xlab ="Neuronal nodal contribution" , xlim = c(-0.072, 0.07))               -> NnNs
  
  pluck(nodes_vs_subsystems,'Astrocyte_nodes_vs_astrocyte_subsystem' )%>%
    plot_density(titulo = 'Astrocytic energy metabolism', fill = 'darkolivegreen4', xlab = "Astrocytic nodal contribution", xlim = c(-0.099, 0.14), ylab = NULL) -> AnAs
  
  pluck(nodes_vs_subsystems,'Astrocyte_nodes_vs_neuron_subsystem' )%>%
    plot_density(titulo = 'Neuronal energy metabolism', fill = 'darkseagreen4', xlab = NULL, xlim = c(-0.099, 0.14), ylab = NULL)                                   -> AnNs
  
  return(list(NnAs,NnNs,AnAs,AnNs) %>% 
           purrr::set_names(c('NnAs','NnNs','AnAs','AnNs')))
  
  
}

# Functions for generating figure 2: 
get_total_centralities    <- function(data, aggregation_method = "fold_change_geometric"){
  data[[aggregation_method]]-> df
  df %>% filter_centralities %>% abs %>% rowSums }


library(WGCNA)
get_correlation_matrices <-function(data, aggregation_method = "fold_change_geometric"){
  data[[aggregation_method]]-> df
  df %>% filter_centralities %>% transposeBigData   %>% cor(method = "kendall") -> corrM
  return(corrM)}

get_node_types <- function(data, aggregation_method = "fold_change_geometric"){
  data[[aggregation_method]]-> df
  node_types <- df$`Node type`
  return(node_types)
}

get_node_IDs <- function(data, aggregation_method = "fold_change_geometric"){
  data[[aggregation_method]]-> df
  node_ID <- df$ID
  return(node_ID)
}

library(ComplexHeatmap)
generate_heatmap <- function(my.corr.mat,total.abs.centrality, node_types){
  set.seed(1)
  left_annotation <- rowAnnotation(`Nodes` = node_types, col = list( `Nodes` = c(
    "Astrocyte"        = "deepskyblue3", 
    'Exchange'   = 'darkgreen',
    'Sink/Demand Astro'='darkgoldenrod',
    "Neuron"           = "brown2",
    'Sink/Demand Neuron'= 'blueviolet' )))
  
  right_annotation <-   
    rowAnnotation(gap = unit(12, "points"),Ce = anno_barplot(bar_width = 0.01,width = unit(1.5, "cm"), border = T,total.abs.centrality, gp = gpar(col = 'azure4')))
  

  
  ht <- Heatmap(my.corr.mat, name = "Correlation",  left_annotation =left_annotation,
                right_annotation=right_annotation,
                clustering_distance_columns  = function(m)   dist(m, method = 'euclidean'),
                cluster_columns              = function(x) fastcluster::hclust(dist(x), "median"),
                
                clustering_distance_rows   = function(m)   dist(m, method = 'euclidean'),
                cluster_rows               = function(x) fastcluster::hclust(dist(x), "median"),
                row_km = 2,
                column_km = 2,
                border = TRUE,
                row_dend_width    = unit(5, "cm"),
                column_dend_height = unit(2,"cm"),
                row_gap = unit(2, "mm"),
                column_gap = unit(2, "mm"),
                width = unit(8, "cm"), 
                height = unit(8, "cm"),
                column_title = c( "Neuronal cluster" , "Astrocytic cluster"),
                column_title_gp = gpar(fontsize = 10),
                row_title_rot  = 0,
                show_column_names    = F,
                show_row_names    = F,
                row_names_gp = gpar(fontsize = 8),
                row_title = c("Neuronal\n cluster", "Astrocytic\n cluster"),
                row_title_gp = gpar(fontsize = 10))
  return(ht)
  
}


get_heatmap_data <- function(ht,node_types, df, aggregation_method = "fold_change_geometric"){
  df[[aggregation_method]]-> df

row_order(ht)[[2]] -> neuron_cluster
row_order(ht)[[1]] -> astro_cluster 
node_types   -> Nodes
df -> data
data$ID[neuron_cluster]     -> neuron_cluster_names
data$ID[astro_cluster]      -> astro_cluster_names
ht@matrix %>% as.data.frame -> heatmap_matrix


heatmap_matrix[neuron_cluster_names,neuron_cluster_names] -> ht_Neuronal_cluster
heatmap_matrix[astro_cluster_names,astro_cluster_names]   -> ht_Astrocytic_cluster
heatmap_matrix[astro_cluster_names,neuron_cluster_names]  -> ht_Neuron_vs_Astrocyte

ht_Neuronal_cluster %>% as.matrix %>% c -> `Neuronal_cluster`
ht_Astrocytic_cluster%>% as.matrix %>% c  ->`Astrocytic_cluster` 
ht_Neuron_vs_Astrocyte%>% as.matrix %>% c  -> `Neuron_vs_Astrocyte`

data.frame(`Neuronal_cluster`) %>% gather  -> A
data.frame(`Astrocytic_cluster`) %>% gather   -> B
data.frame( `Neuron_vs_Astrocyte`) %>% gather -> C
quads <- rbind(A,B,C)
colnames(quads) <- c("Comparison","Node correlation")

return(list(ht_Neuronal_cluster,ht_Astrocytic_cluster,ht_Neuron_vs_Astrocyte,heatmap_matrix,quads) %>% 
purrr::set_names(c('ht_Neuronal_cluster','ht_Astrocytic_cluster','ht_Neuron_vs_Astrocyte','heatmap_matrix','quads')))  
}
  
plot_quads_corrs <- function(quads, my_comparisons){
   ggplot(quads, aes(x=Comparison, y=`Node correlation`, fill=Comparison)) + 
    geom_violin(trim=F, colour = "azure4")+  
    stat_compare_means( vjust= -1., hjust= 0, comparisons = my_comparisons, method = "wilcox.test", p.adjust.method = "bonferroni", label = "p.signif")+
    scale_fill_manual(values = alpha(c("deepskyblue3", "blueviolet", "brown2"), .8)) + 
    theme(plot.title = element_text(hjust = 0.5),legend.position="none",  axis.title.x=element_blank(),
          axis.text.y = element_text(angle = 45, hjust = 1), axis.text.x = element_text(angle = 50, hjust = 1))  + 
    scale_y_continuous(limits=c(-0.7, 2)) + geom_hline(yintercept=0, linetype="dashed", color = "darkred",  size=1)-> corr_comparisons
  corr_comparisons  }



library(ProjectionBasedClustering)
library(PCAtools)
library(magrittr)

get_pca <- function(my.corr.mat, total.abs.centrality, node_types){

p <- pca(my.corr.mat)

cbind(p$rotated$PC1, p$rotated$PC2) %>% as.matrix %>% as.data.frame-> my_pca
my_pca %>%  cbind(node_types) %>% set_colnames(c('PC1','PC2','node')) -> to.pca.scatter
total.abs.centrality -> Ce
b <- ggplot(to.pca.scatter, aes(x = PC1, y = PC2)) 
b +   scale_color_manual(labels = c("Astro", "Exch",'Neu','Si/De Ast', 'Si/De Neu'), values =  c("deepskyblue3",'darkgreen','brown2','darkgoldenrod','blueviolet')) +
  theme(legend.position="right", legend.box = "vertical")+ geom_point(aes( size =  Ce,color = node))-> pca_node_centrality

pca_node_centrality
}
# Functions for generating figure 3: 
library(readr)



import_phpp_data <- function(path0){
  
  import_file <- function( file0 ){   read_csv(file.path(path0, file0), col_names = FALSE) %>% reduce(c)   }
  
  c("glucose_uptake.csv", 'oxygen_uptake.csv', 'objective.csv', 'GLUVESSEC_Neuron.csv', 'L_LACt2r_Int.csv', 'NaEX_Neuron.csv', 'Y_o2_glc.csv', 'Y_ATP_glc.csv') -> my_files
  my_imported_files <- purrr::map(my_files, import_file) %>% purrr::set_names(my_files %>% stringr::str_replace(., '.csv', ''))
  
  
  phpp_axes                 <- data.frame(glucose_uptake = my_imported_files[["glucose_uptake"]], oxygen_uptake = my_imported_files[["oxygen_uptake"]])
  
  data.frame(phpp_axes, objective = my_imported_files[['objective']])  -> phpp_objective
  data.frame(phpp_axes, GLUVESSEC_Neuron= my_imported_files[['GLUVESSEC_Neuron']])  -> phpp_GLUVESSEC_Neuron
  data.frame(phpp_axes, L_LACt2r_Int= my_imported_files[['L_LACt2r_Int']])  -> phpp_L_LACt2r_Int
  data.frame(phpp_axes, NaEX_Neuron= my_imported_files[['NaEX_Neuron']])  -> phpp_NaEX_Neuron
  data.frame(phpp_axes, Y_o2_glc= my_imported_files[['Y_o2_glc']])  -> phpp_Y_o2_glc
  data.frame(phpp_axes, Y_ATP_glc= my_imported_files[['Y_ATP_glc']])  -> phpp_Y_ATP_glc
  
  return(list(phpp_objective,phpp_GLUVESSEC_Neuron,phpp_L_LACt2r_Int,phpp_NaEX_Neuron,phpp_Y_o2_glc,phpp_Y_ATP_glc) %>% 
  purrr::set_names(c('phpp_objective','phpp_GLUVESSEC_Neuron','phpp_L_LACt2r_Int','phpp_NaEX_Neuron','phpp_Y_o2_glc','phpp_Y_ATP_glc')))
  
}
library(ggplot2)
library(tidyverse)
library(metR)
library(ggpubr)
library(Cairo)
library(magrittr)
get_phpp_contours <- function(df, inter_colors, breaks_lines, label_placement_fraction, nudge_y){
  df %>% 
    signif(digits = 10) %>% unique %>% set_names(c('x','y','z'))  -> df2
  
  df2 %>%ggplot(aes(x=x, y=y, z=z)) +
    #labs(inherit.aes = T, colour="Tff")+
    geom_contour_fill(inherit.aes = T,  breaks =  seq(0,max(df2['z']), inter_colors),  colour = "white", size = 0, alpha = 1) +
    geom_contour(inherit.aes = T, colour = "white",  breaks = breaks_lines, size = 1 )+
    geom_text_contour(breaks = breaks_lines , inherit.aes = T,  colour = "white" , skip = 0,nudge_x = 0,nudge_y = nudge_y,
                      label.placement = label_placement_fraction(frac = label_placement_fraction)) +
    scale_fill_continuous(type = "viridis")+
    scale_x_continuous(expand = c(0,0))+
    scale_y_continuous(expand = c(0,-nudge_y)) +
    theme(legend.text =  element_text(angle = 60, size = 6), legend.title = element_text(size = 8) ,  legend.key.width = unit(.5, "cm"), legend.key.height=unit(2, "mm") ,   legend.position="bottom",        legend.box="horizontal",  axis.title.y=element_blank(), axis.title.x=element_blank()) +
    
    geom_point(aes(x=3.519, y=32.407), size = 3, colour = "red") +
    geom_segment(aes(x= 3.519, xend=3.519 , y= 13, yend=32.407 ), colour="black", lwd=.5, linetype = 'dotted')+
    geom_segment(aes(x= 1, xend=3.519 , y= 32.407, yend=32.407 ), colour="black", lwd=.5, linetype = 'dotted')
}

get_fba_data <- function(data, aggregation_method = "fold_change_geometric"){
  data[[aggregation_method]]-> df
  df %>% dplyr::select(c("ID","Name","Reaction","Flux", "Sensitivity", "Node type" )) %>%
    
    mutate(`Node type` = ifelse(str_detect(`Node type`, regex('Sink|Demand',ignore_case = T)), 'Terminal', `Node type`))
  
}

library(scales)
library(reshape2)
get_optimality_values <- function(data){
  
  pseudoLog10 <- function(x) { asinh(x/2)/log(10) }
  
  data %>% select(c("ID", "Sensitivity", 'Flux'))   %>% column_to_rownames('ID') %>% abs %>% as.matrix() %>% pseudoLog10  -> data.log
  data.log %>% rescale(to = c(0,1))%>% as.data.frame %>% set_rownames(rownames(data.log))-> data.log.rescaled
  merge(data.log.rescaled %>% rownames_to_column('ID') ,  data[, c('ID','Node type')] ) -> data.log.rescaled
  
  
  data.log.rescaled %>% mutate(total = Sensitivity+Flux) -> data.log.rescaled_total
  data.log.rescaled_total %>% arrange(desc(total))   -> data.log.rescaled_total2
  data.log.rescaled_total2 %<>% filter(total > 0.001)
  data.log.rescaled_total2 %>% select(-c( "total")) %>% melt -> optimality_values
  
  return(list(optimality_values,data.log.rescaled_total2) %>% 
 purrr::set_names(c('optimality_values','optimality_df')))

}

library(tidyverse)
library(readxl)


c('PGM', 'ACYP', 'PGI', 'PGK','PYK', 'HEX1', 'DPGase', 'TPI', 'PFK', 'ENO', 'GAPD', 'DPGM', 'FBA', 'G3PD2m' ,
  'ACYP_Neuron', 'DPGM_Neuron', 'DPGase_Neuron', 'ENO_Neuron', 'FBA_Neuron', 'G3PD2m_Neuron', 'GAPD_Neuron', 'HEX1_Neuron', 'PFK_Neuron', 'PGI_Neuron', 'PGK_Neuron', 'PGM_Neuron', 'PYK_Neuron', 'TPI_Neuron' ,
  'ATPS4m_Neuron', 'CYOOm2_Neuron', 'CYOR-u10m_Neuron', 'NADH2-u10m_Neuron', 'PPA_Neuron', 'PPAm_Neuron' ,
  'PPAm', 'ATPS4m', 'CYOOm2', 'CYOR-u10m', 'NADH2-u10m', 'PPA', 'GLCt1r', 'GLCt1r_Neuron', 'GLCt2r') -> subsystems

filter_out_subsystems <- function(df){df[!(df$ID %in% subsystems),]}

get_genes_from_module <- function(hubs){     hubs %>% 
    dplyr::select(matches('gene')) %>% purrr::reduce(c) %>% na.exclude() %>% str_c(collapse = ", ") %>% 
    str_split(pattern=",") %>%str_extract_all('\\d+') %>% unlist() %>% str_trim %>% unique()}
#Functions
get_genes_from_each_hub <- function(file, cut_off_optimality = 0.01 , cut_off_centrality =  0.01){
  
  #central_hubs <- readxl::read_excel(file)%>% dplyr::select(-"Optimality")%>%filter(Centrality > cut_off_centrality)%>% filter_out_subsystems
  #optimal_hubs <- readxl::read_excel(file)%>% dplyr::select(-"Centrality")%>%filter(Optimality > cut_off_optimality)%>% filter_out_subsystems
  
  central_hubs <- readr::read_csv(file)%>% dplyr::select(-"Optimality")%>%filter(Centrality > cut_off_centrality)%>% filter_out_subsystems
  optimal_hubs <- readr::read_csv(file)%>% dplyr::select(-"Centrality")%>%filter(Optimality > cut_off_optimality)%>% filter_out_subsystems
  
  central_optimal_hubs <-  readr::read_csv(file)%>% filter_out_subsystems
  
  intersect(central_hubs$ID, optimal_hubs$ID) -> hyper_IDs
  setdiff(central_hubs$ID,hyper_IDs)          -> central_IDs
  setdiff(optimal_hubs$ID,hyper_IDs)         -> optimal_IDs
  
  central_hubs         %>% filter(ID %in% central_IDs) %>%  get_genes_from_module -> central_hub_genes
  optimal_hubs         %>% filter(ID %in% optimal_IDs) %>%  get_genes_from_module -> optimal_hub_genes
  central_optimal_hubs %>% filter(ID %in% hyper_IDs) %>% get_genes_from_module -> hyper_hub_genes
  
  list_of_genes_by_hubs <- list(optimal_hub_genes,central_hub_genes,hyper_hub_genes) %>% set_names(c('optimal_hub_genes','central_hub_genes','hyper_hub_genes'))
  
  return(list_of_genes_by_hubs)
  
}


get_pure_gene_groups <- function(list_of_genes_by_hubs){
  
  list_of_genes_by_hubs$optimal_hub_genes -> optimal_hub_genes
  list_of_genes_by_hubs$central_hub_genes ->central_hub_genes
  list_of_genes_by_hubs$hyper_hub_genes ->hyper_hub_genes
  
  
  Type_5                <- Reduce(intersect, list(central_hub_genes,optimal_hub_genes,  hyper_hub_genes))
  Type_4_hyper_optimal  <- setdiff( Reduce(intersect, list(optimal_hub_genes,  hyper_hub_genes)), Type_5)
  Typer_4_hyper_central <- setdiff( Reduce(intersect, list(central_hub_genes,  hyper_hub_genes)), Type_5)
  
  Type_3                <- setdiff(hyper_hub_genes, c(Type_5, Type_4_hyper_optimal, Typer_4_hyper_central) )
  
  
  Type_2                <-  setdiff(Reduce(intersect, list(central_hub_genes,optimal_hub_genes)), c(Type_5, Type_4_hyper_optimal, Typer_4_hyper_central, Type_3) )
  Type_1_optimal        <-  setdiff(optimal_hub_genes, c(Type_2, Type_3, Typer_4_hyper_central, Type_4_hyper_optimal, Type_5) )
  Type_1_central        <-  setdiff(central_hub_genes, c(Type_2, Type_3, Typer_4_hyper_central, Type_4_hyper_optimal, Type_5) )
  
  list(Type_1_optimal, Type_1_central, Type_2, Type_3, Type_4_hyper_optimal, Typer_4_hyper_central, Type_5) %>% 
    purrr::set_names('Type_1_optimal', 'Type_1_central', 'Type_2', 'Type_3', 'Type_4_hyper_optimal', 'Typer_4_hyper_central', 'Type_5') -> my_list
  # sanity check
  
  c(optimal_hub_genes, central_hub_genes, hyper_hub_genes) %>% unique() -> all_genes
  all((Reduce(c, my_list) %in% all_genes) & (all_genes %in% Reduce(c, my_list)) ) -> sanity_check
  
  if (sanity_check){message('all classified')}
  
  return(my_list)
}

hubs_genes_to_rxns <- function(file, genes){
  
  file.path(figure_folder_path,file) %>% readr::read_csv() %>%
    dplyr::filter( recon3_genes %in%  genes | Lewis2010_genes%in%  genes | VMH_gene_IDs%in%  genes | KEGG_genes%in%  genes) -> hubs
  return(hubs)
  
}

cell_and_modules <- function(figure_folder_path, file,  cell,cut_off_optimality, cut_off_centrality ){
  
  file.path(figure_folder_path,file) %>% 
    get_genes_from_each_hub(cut_off_optimality = cut_off_optimality , cut_off_centrality = cut_off_centrality) -> list_of_genes_by_hubs
  hubs_genes_to_rxns(file, list_of_genes_by_hubs$optimal_hub_genes ) -> optimal_hub_genes
  hubs_genes_to_rxns(file, list_of_genes_by_hubs$central_hub_genes ) -> central_hub_genes
  hubs_genes_to_rxns(file, list_of_genes_by_hubs$hyper_hub_genes ) ->   hyper_hub_genes
  library(magrittr)
  optimal_hub_genes %<>% dplyr::select(c(ID, Centrality, Optimality)) %>% mutate(Node = cell, Module = "Optimal")
  central_hub_genes %<>% dplyr::select(c(ID, Centrality, Optimality)) %>% mutate(Node = cell, Module = "Central")
  hyper_hub_genes   %<>% dplyr::select(c(ID, Centrality, Optimality)) %>% mutate(Node = cell, Module = "Hyper")
  cell_df <- rbind(optimal_hub_genes, central_hub_genes, hyper_hub_genes)
  return(cell_df)}

plot_stack <- function(neuron_modules, astrocyte_modules){
  
  
  get_node_and_module <- function(df){ df %>% dplyr::select(Node,Module)}
  
  list(neuron_modules,
       astrocyte_modules) %>% map(get_node_and_module) %>% purrr::reduce(rbind) %>% set_colnames(c("Cell","Hub")) -> Cell_Module
  Cell_Module %>% dplyr::count(Cell) %>% filter(Cell == "Astrocyte") %>% .[[2]] -> astro_total
  Cell_Module %>% dplyr::count(Cell) %>% filter(Cell == "Neuron")    %>% .[[2]] -> neuron_total
  Cell_Module %>% dplyr::count(Cell,Hub) %>% filter(Cell == "Astrocyte") %>% .[["n"]] -> modules_astro
  Cell_Module %>% dplyr::count(Cell,Hub) %>% filter(Cell == "Neuron") %>% .[["n"]]    -> modules_neuron
  Cell_Module %>% dplyr::count(Cell,Hub) %>% mutate("Percentage %" =  100* c(modules_astro/astro_total,modules_neuron/neuron_total) %>% round(4)  ) -> my_counts
  
  ggplot(my_counts, aes(fill=Hub, x=Cell, y=`Percentage %`,  label = `Percentage %`)) + 
    geom_bar(stat = "identity") + scale_fill_manual(values = alpha(c("chartreuse3", "darkorange1", "darkorchid3"), 1)) + 
    theme(axis.title.x =  element_blank(), legend.position = 'right', axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1,size = 12))+
    geom_text(angle = 90,size = 4.5, position = position_stack(vjust = 0.5)) -> plot.my_counts_stacked
  
  plot.my_counts_stacked
}

get_ht_quadrans <- function(file, null_diagonal = T){
  A <- read_rds(file)
  if(null_diagonal)  {diag(A) <- NA}
  return(A)}

get_quads <- function(centralhubs, heatmap_matrix.rds){
  
  
  heatmap_matrix <- get_ht_quadrans(heatmap_matrix.rds)
  
  only_neuronal <- centralhubs %>% filter(Node == "Neuron") %>% .[["ID"]] %>% str_replace_all("\\,",'') %>% str_replace("\\_",'') %>% str_replace("\\-",'') %>% str_replace("\\,",'')
  only_astrocyte <-  centralhubs %>% filter(Node == "Astrocyte") %>% .[["ID"]] %>% str_replace("\\,",'') %>% str_replace("\\_",'') %>% str_replace("\\-",'') %>% str_replace("\\,",'')
  
  heatmap_matrix%>%colnames() %>% str_replace("\\,",'') %>% str_replace("\\_",'') %>% str_replace("\\-",'') %>% str_replace("\\,",'')-> fixed_colnames
  heatmap_matrix%>%rownames() %>% str_replace("\\,",'') %>% str_replace("\\_",'') %>% str_replace("\\-",'') %>% str_replace("\\,",'')-> fixed_rownames
  #str_match(fixed_colnames, ',') %>% na.exclude()
  #setdiff(c(only_neuronal,only_astrocyte),fixed_colnames)
  colnames(heatmap_matrix)<- fixed_colnames
  rownames(heatmap_matrix)<- fixed_rownames
  heatmap_matrix[only_neuronal, only_neuronal] %>% as.matrix %>% .[upper.tri(.)] %>% c-> Neuronal_cluster
  heatmap_matrix[only_astrocyte,only_astrocyte]%>% as.matrix %>% .[upper.tri(.)] %>% c-> Astrocytic_cluster
  heatmap_matrix[only_neuronal, only_astrocyte]%>% as.matrix %>% .[upper.tri(.)] %>% c-> Neuron_vs_Astro
  
  data.frame(Neuronal_cluster) %>% gather -> A
  data.frame(Astrocytic_cluster) %>% gather -> B
  data.frame(Neuron_vs_Astro) %>% gather -> C
  quads <- rbind(A,B,C) %>% na.omit()
  
  colnames(quads) <- c("Comparison","Nodes correlations")
  
  return(quads)
}




























