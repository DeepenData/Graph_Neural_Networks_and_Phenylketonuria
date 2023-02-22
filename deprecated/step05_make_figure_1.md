---
title: "R Notebook"
output: html_notebook
---


```r
library(tidyverse)
library(magrittr)
library(ggbeeswarm)
```

```
## Error in library(ggbeeswarm): there is no package called 'ggbeeswarm'
```

```r
library(ggpubr)
setwd("/DeepenData/Repos/geometric_cobra")

#metabolite_data        <- read_csv("./metabolites_data/metabolite_data.csv")
#subsys_fluxes_non_zero <- arrow::read_parquet("./results/flux/subsys_fluxes_non_zero.parquet.gzip")



get_vars_by_quntl <- function(df){
                                df %>% 
                                summarise(across(where(is.double),  ~ mean(.x, na.rm = TRUE))) %>% t -> vars_mean
                              q_25 <- quantile(vars_mean, 0.25, na.rm = TRUE)[[1]]
                              q_50 <- quantile(vars_mean, 0.5, na.rm = TRUE)[[1]]
                              q_75 <- quantile(vars_mean, 0.75, na.rm = TRUE)[[1]]
                              
                              vars_q_1 <- vars_mean[vars_mean < q_25,] %>% names 
                              vars_q_2 <- vars_mean[q_25 <= vars_mean & vars_mean < q_50,] %>% names 
                              vars_q_3 <- vars_mean[q_50 <= vars_mean &   vars_mean < q_75,] %>% names 
                              vars_q_4 <- vars_mean[q_75 <= vars_mean ,] %>% names 
                              return(list(vars_q_1, vars_q_2, vars_q_3, vars_q_4))
}

foo <- function(plot){
                                                  p <- ggplot_build(plot)
                             p$data[[1]] <-   p$data[[1]] %>%
                               mutate(diff = abs(x-round(x)), x = case_when(group %% 2 == 0 ~ round(x) + diff,
                                                    TRUE ~ round(x) - diff)) %>%    select(-diff)
                             
                             
                             return(
                               (ggplot_gtable(p))
                             )
                             
                            }
plot_split_violins <- function(data, quntl_num,vars_by_quntl ,legend.position, logscale, alpha, size_marker, color_1, color_2){
                          colours <- c(alpha(c( color_1), alpha), alpha(c(color_2), 1))
                          #data %>% get_vars_by_quntl -> vars_by_quntl
                          
                          
                          
                          p <- data %>% dplyr::select(c(vars_by_quntl[[quntl_num]], "Group")) %>%  pivot_longer(-Group)  %>% 
                          ggplot(aes(name,value,color= Group))+ 
                          geom_quasirandom(size= c(size_marker)) + #scale_fill_manual(values = colours, name = "") +
                          theme_minimal()+
                          guides(color = guide_legend(override.aes = list(size = 3, alpha = 1) ))+ 
                          theme(axis.title.x = element_blank(),axis.text.x = element_text(angle = 45, vjust = 1, hjust=1),
                                  axis.title.y = element_blank(), legend.position=legend.position,
                                 legend.text = element_text(size=7))+scale_color_discrete(name="")+
                            scale_colour_manual(values = colours, name = "") 
                          
                          if(logscale){p <-p  + coord_trans( y="log2") }
                          
                          return(foo(p) %>% as_ggplot)  }

generate_right_swarms <- function(metabolite_data, color_1="dodgerblue2", color_2="darkorange"){
            metabolite_data %>% dplyr::select(-c('Phe')) %>% get_vars_by_quntl -> vars_by_quntl
            
            vars_by_quntl[[5]] <- c('Phe')
            a_subplot <-plot_split_violins(metabolite_data, 5 , vars_by_quntl, 'top', TRUE, .1, .6, color_1=color_1, color_2=color_2)
            b_subplot <- plot_split_violins(metabolite_data, 4 , vars_by_quntl, 'none', F, .02, .01, color_1=color_1, color_2=color_2)
            c_subplot <- plot_split_violins(metabolite_data, 3 , vars_by_quntl, 'none', F, .02, .01, color_1=color_1, color_2=color_2)
            d_subplot <- plot_split_violins(metabolite_data, 2 , vars_by_quntl, 'none', F, .02, .01, color_1=color_1, color_2=color_2)
            e_subplot <- plot_split_violins(metabolite_data, 1 , vars_by_quntl, 'none', F, .02, .01, color_1=color_1, color_2=color_2)
            right_swarms <- ggarrange(b_subplot, c_subplot, d_subplot, e_subplot, ncol = 1, labels = c("b","c","d","e"), vjust = 1.1, hjust = .5)
            ggarrange(a_subplot, right_swarms, widths = c(.55, 1), labels = c("a",""), vjust = 1.1) -> swarm_panel
            return(swarm_panel)
}


make_swarm_panel <- function(df){
                            right_swarms <- generate_right_swarms(df)
                            a_subplot <-plot_split_violins(df, 5 , vars_by_quntl, 'top', TRUE, .1, .6)
                            ggarrange(a_subplot, right_swarms, widths = c(.55, 1), labels = c("a",""), vjust = 1.1) -> swarm_panel
                            return(swarm_panel)
}
```



```r
setwd("/DeepenData/Repos/geometric_cobra")

metabolites_outliers_imputed     <- arrow::read_parquet("./metabolites_data/metabolites_outliers_imputed.parquet.gzip")
rownames(metabolites_outliers_imputed) <- NULL
generate_right_swarms(metabolites_outliers_imputed) -> concentration_subplot
```

```
## Error in geom_quasirandom(size = c(size_marker)): could not find function "geom_quasirandom"
```

```r
# a_subplot <-plot_split_violins(metabolites_outliers_imputed, 5 , vars_by_quntl, 'top', TRUE, .1, .6)
```





```r
setwd("/DeepenData/Repos/geometric_cobra")

augmented_metabolite_data <- arrow::read_parquet("./results/dataframes/concentrations/augmented_metabolite_data_v2.parquet.gzip") 
augmented_data <- augmented_metabolite_data %>%mutate(label = dplyr::if_else(label == 1, "PKU", "Control")) %>% rename(Group = label)

aug_PKUs <-augmented_data %>% dplyr::filter(Group == "PKU")
generate_right_swarms(aug_PKUs, color_1 = "darkorange", color_2 = "darkorange") -> concentration_subplot_augmented
```

```
## Error in geom_quasirandom(size = c(size_marker)): could not find function "geom_quasirandom"
```








```r
make_umap <- function(data, legend.position){UMAPM <- umap::umap(data  %>% select_if(is.numeric) %>% 
                                                                        as.matrix(), n_neighbors = 20, scale = T, n_threads = 20, fast_sgd = F, metric = 'cosine', spread = 10)
                       
                                                       

p <- ggplot(as_tibble(UMAPM$layout), aes(V1, V2, colour= data[["Group"]]))
                                                        UMAP_reduction <-  p + geom_point(size= .2) + xlab('UMAP dimension 1') + ylab('UMAP dimension 2')+
                                                          scale_colour_manual(values =  c(alpha(c( "dodgerblue2"), .5), alpha(c("darkorange"), 1)))+ 
                                                          theme(legend.position=legend.position, text = element_text(size = 9), axis.title.y = element_text(size = 7),  axis.title.x = element_text(size = 7))   +
                                                             guides(color = guide_legend(override.aes = list(size = 3) ) )+guides(color = guide_legend(title = "Group", override.aes = list(size = 3) )) 
                                                        return(UMAP_reduction)}

#metabolite_data %>% make_umap('none') -> umap_plot_1
```

```r
setwd("/DeepenData/Repos/geometric_cobra")

metabolites_outliers_imputed     <- arrow::read_parquet("./metabolites_data/metabolites_outliers_imputed.parquet.gzip") # %>% rename( Group = label) #%>% mutate(Group = if_else(Group == 1,'PKU','Control')) #%>% make_umap('none')-> umap_plot_1

metabolites_outliers_imputed %>% make_umap('none') -> umap_plot_1
```

```
## Error in loadNamespace(x): there is no package called 'umap'
```



```r
setwd("/DeepenData/Repos/geometric_cobra")

augmented_metabolite_data <- arrow::read_parquet("./results/dataframes/concentrations/augmented_metabolite_data_v2.parquet.gzip")
augmented_metabolite_data %>% 
  rename( Group = label) %>% mutate(Group = if_else(Group == 1,'PKU','Control')) %>% make_umap('none')-> umap_plot_2
```

```
## Error in loadNamespace(x): there is no package called 'umap'
```

```r
ggarrange(umap_plot_1, umap_plot_2, labels = c("f","g"), vjust = 1.1, hjust = .2) -> umap_panel
```

```
## Error in ggarrange(umap_plot_1, umap_plot_2, labels = c("f", "g"), vjust = 1.1, : object 'umap_plot_1' not found
```

```r
ggarrange(concentration_subplot, umap_panel, ncol = 1, heights = c(1,.25)) -> left_panel
```

```
## Error in ggarrange(concentration_subplot, umap_panel, ncol = 1, heights = c(1, : object 'concentration_subplot' not found
```


############################################
##########-------FLUX----------------------


```r
setwd("/DeepenData/Repos/geometric_cobra")

all_flux_samples <-  arrow::read_parquet("./results/dataframes/fluxes/all_flux_samples.parquet.gzip")
all_flux_samples  %<>% 
  rename( Group = label) %>% mutate(Group = if_else(Group == 1,'PKU','Control'))
```

```r
subsys<- c('DHPR2',
 'PHETA1m',
 'PHYCBOXL',
 'PPOR',
 'PTHPS',
 'THBPT4ACAMDASE',
 'r0403',
 'r0545',
 'r0547',
 'RE0830C',
 'RE0830N',
 'RE1709C',
 'RE1709N',
 'RE2660C',
 'RE2660N',
 'PHLAC',
 'DHPR',
 'r0398',
 'PHETA1',
 'HMR_6729',
 'HMR_6755',
 'HMR_6770',
 'HMR_6782',
 'HMR_6790',
 'HMR_6854',
 'HMR_6874',
 'HMR_6876',
 "Group")
```



```r
colours <- c(alpha(c( "dodgerblue"), .8), alpha(c("darkorange"), .8))

                       p <- all_flux_samples %>% dplyr::select(subsys) %>% sample_n(500)  %>%  
                       pivot_longer(-Group)  %>% 
                          
                            
                            ggplot(aes(name,value,color= Group))+ 
                          geom_quasirandom(size= c(.1)) + #scale_fill_manual(values = colours, name = "") +
                          theme_minimal()+
                          guides(color = guide_legend(override.aes = list(size = 5, alpha = 1) ))+ 
                          theme(axis.title.x = element_blank(),axis.text.x = element_text(angle = 45, vjust = 1, hjust=1),
                                  axis.title.y = element_blank(), legend.position="none",axis.text = element_text(size = 7),
                                 legend.text = element_text(size=7))+scale_color_discrete(name="")+
                            scale_colour_manual(values = colours, name = "")
```

```
## Warning: Using an external vector in selections was deprecated in tidyselect 1.1.0.
## ℹ Please use `all_of()` or `any_of()` instead.
##   # Was:
##   data %>% select(subsys)
## 
##   # Now:
##   data %>% select(all_of(subsys))
## 
## See <https://tidyselect.r-lib.org/reference/faq-external-vector.html>.
```

```
## Error in geom_quasirandom(size = c(0.1)): could not find function "geom_quasirandom"
```

```r
foo(p)%>% as_ggplot -> flux_subplot
```

```
## Error in options(ggplot2_plot_env = env): object 'p' not found
```

######################################################################################
######################################################################################
#########################################################################

```r
setwd("/DeepenData/Repos/geometric_cobra")

#img <- png::readPNG("./results/graphs/for_visualizations/control_concentration.png")

control_concentration.png <- ggplot() + background_image( png::readPNG("./results/graphs/for_visualizations/control_concentration.png"))
PKU_concentration.png     <- ggplot() + background_image( png::readPNG("./results/graphs/for_visualizations/PKU_concentration.png"))

control_fluxes.png     <- ggplot() + background_image( png::readPNG("./results/graphs/for_visualizations/control_fluxes.png"))
PKU_fluxes.png         <- ggplot() + background_image( png::readPNG("./results/graphs/for_visualizations/PKU_fluxes.png"))



ggarrange(control_concentration.png, PKU_concentration.png, control_fluxes.png, PKU_fluxes.png, labels = c("i","","j","")) ->  graphs_panel

ggarrange(umap_panel, flux_subplot,graphs_panel, nrow = 3, heights = c(.3,.4,1), labels = c("","h",""), vjust = -.1) -> right_panel
```

```
## Error in ggarrange(umap_panel, flux_subplot, graphs_panel, nrow = 3, heights = c(0.3, : object 'umap_panel' not found
```



```r
ggarrange(concentration_subplot,right_panel, ncol = 2, widths = c(1,.9) ) -> full_panel
```

```
## Error in ggarrange(concentration_subplot, right_panel, ncol = 2, widths = c(1, : object 'concentration_subplot' not found
```


```r
setwd("/DeepenData/Repos/geometric_cobra")



ggsave('./results/figures/figure_1.png', full_panel, height = 7, width = 9, bg = "white")
```

```
## Error in grid.draw(plot): object 'full_panel' not found
```



#
#
#

```r
plot_swarm       <- function(df, size, alpha){
                              df%>% pivot_longer(everything()) -> long
                              ggplot(long,aes(name, value)) + geom_quasirandom(size=size, alpha = alpha, aes(color='red4')) -> my_plot
                              return(my_plot)}







p <- subsys_fluxes_non_zero[,colSums(subsys_fluxes_non_zero)%>%abs > 1] %>% sample_n(1000) %>% abs %>% plot_swarm(.2,.2) 
```

```
## Error in sample_n(., 1000): object 'subsys_fluxes_non_zero' not found
```

```r
flux_subplot <- p + ylim(c(0,5.9)) + 
  xlab('Reaction')+ ylab('Flux') + theme_minimal() +  theme(legend.position = 'none', axis.text.x = element_text(angle = 45, vjust = 1, hjust=1))
```

```
## Error in eval(expr, envir, enclos): object 'p' not found
```



```r
umap_panel <-  ggarrange(umap_plot_1, umap_plot_2, widths = c(1,1))
```

```
## Error in ggarrange(umap_plot_1, umap_plot_2, widths = c(1, 1)): object 'umap_plot_1' not found
```

```r
ggarrange(umap_panel, graph, flux_subplot, ncol = 1, heights = c(.4,1,.55), labels = c("f","g","h"),  vjust = 1.1, hjust = .5) -> right_panel
```

```
## Error in ggarrange(umap_panel, graph, flux_subplot, ncol = 1, heights = c(0.4, : object 'umap_panel' not found
```

```r
ggarrange(swarm_panel, right_panel, ncol = 2, widths = c(1,.7)) -> figure_1_panel
```

```
## Error in ggarrange(swarm_panel, right_panel, ncol = 2, widths = c(1, 0.7)): object 'swarm_panel' not found
```



```r
setwd("/DeepenData/Repos/geometric_cobra")

ggsave("./results/figures/Figure_1.png", 
       figure_1_panel, height=6.5, width=8, bg = "white")
```

```
## Error in grid.draw(plot): object 'figure_1_panel' not found
```


#####################################################################################################################################################
######################################################################################################################################################

```r
get_data_subset <- function(df , condition, aminoacids){
                                        df %<>%  select(!dplyr::matches('SA')) %>%
                                        dplyr::filter(Group==condition) %>% 
                                        select_if(is.numeric) 
                                        if(aminoacids){df %<>% select(!dplyr::matches('C\\d+'))
                                        } else {df %<>% select(dplyr::matches('C\\d+'))}
                                        return(df)}
 
AAs_PKU        <- get_data_subset(metabolite_data, "PKU", T)
```

```
## Error in select(., !dplyr::matches("SA")): object 'metabolite_data' not found
```

```r
AcylCs_PKU     <- get_data_subset(metabolite_data, "PKU", F)
```

```
## Error in select(., !dplyr::matches("SA")): object 'metabolite_data' not found
```

```r
AAs_Control    <- get_data_subset(metabolite_data, "Control", T)
```

```
## Error in select(., !dplyr::matches("SA")): object 'metabolite_data' not found
```

```r
AcylCs_Control <- get_data_subset(metabolite_data, "Control", F)
```

```
## Error in select(., !dplyr::matches("SA")): object 'metabolite_data' not found
```


```r
library(ComplexHeatmap)

plot_densityHeatmap <- function(df, title, pallete){
    return(
   ggplotify::as.ggplot(densityHeatmap(df %>% scale , ylim = c(-1.5, 1.5), cluster_columns = TRUE, title = title, ylab = "", col = pallete,
                                                     title_gp = gpar(fontsize = 6), tick_label_gp = gpar(fontsize = 6), quantile_gp = gpar(fontsize = 6),
                                                     column_names_gp = gpar(fontsize = 6),column_names_rot = 50 )) 
    )}


FLUX_PKU_plot  <- subsys_fluxes_non_zero  %>% plot_densityHeatmap("", rev(RColorBrewer::brewer.pal(11, "RdYlGn")))
```

```
## Error in scale(.): object 'subsys_fluxes_non_zero' not found
```


```r
library(ggpubr)

AAs_Control_plot  <- AAs_Control %>% plot_densityHeatmap("", rev(RColorBrewer::brewer.pal(11, "Spectral")))
```

```
## Error in scale(.): object 'AAs_Control' not found
```

```r
AcylCs_Control_plot <- AcylCs_Control %>% plot_densityHeatmap("", rev(RColorBrewer::brewer.pal(11, "Spectral")))
```

```
## Error in scale(.): object 'AcylCs_Control' not found
```

```r
AAs_PKU_plot <- AAs_PKU %>% plot_densityHeatmap("", rev(RColorBrewer::brewer.pal(11, "Spectral")))
```

```
## Error in scale(.): object 'AAs_PKU' not found
```

```r
AcylCs_PKU_plot <- AcylCs_PKU %>% plot_densityHeatmap("", rev(RColorBrewer::brewer.pal(11, "Spectral")))
```

```
## Error in scale(.): object 'AcylCs_PKU' not found
```

```r
densityHeatmap_panel <- ggarrange(AAs_Control_plot, AcylCs_Control_plot,
                                  AAs_PKU_plot, AcylCs_PKU_plot, 
                                  ncol = 1 ,  common.legend = T, heights = c(.75,1,.75,1), labels = c("a","b","c","d"))
```

```
## Error in ggarrange(AAs_Control_plot, AcylCs_Control_plot, AAs_PKU_plot, : object 'AAs_Control_plot' not found
```


```r
library(umap)
```

```
## Error in library(umap): there is no package called 'umap'
```

```r
make_umap <- function(AAs_Control, AcylCs_Control,AAs_PKU, AcylCs_PKU){
                                                          data_control <-cbind(AAs_Control,AcylCs_Control)
                                                        data_PKU     <- cbind(AAs_PKU,AcylCs_PKU)
                                                        data_control["Group"] <- "Control"
                                                        data_PKU["Group"] <- "PKU"
                                                        
                                                        data <- rbind(data_PKU,data_control)
                                                        
                                                        UMAPM <- umap(data  %>% select_if(is.numeric) %>% 
                                                                        as.matrix(), n_neighbors = 50, scale = T, n_threads = 20, fast_sgd = F, metric = 'cosine', spread = 10)
                                                        p <- ggplot(as_tibble(UMAPM$layout), aes(V1, V2, colour= data[["Group"]]))
                                                        UMAP_reduction <-  p + geom_point(size= .2) + xlab('UMAP dimension 1') + ylab('UMAP dimension 2')+
                                                          scale_colour_manual(values = c("cadetblue", "blue"))+ theme(text = element_text(size = 9))   +
                                                             guides(color = guide_legend(override.aes = list(size = 3) ) )+guides(color = guide_legend(title = "Group", override.aes = list(size = 3) )) 
                                                        return(UMAP_reduction)}

umap_original_data<- make_umap(AAs_Control, AcylCs_Control,AAs_PKU, AcylCs_PKU)
```

```
## Error in cbind(AAs_Control, AcylCs_Control): object 'AAs_Control' not found
```

```r
umap_original_data
```

```
## Error in eval(expr, envir, enclos): object 'umap_original_data' not found
```


```r
setwd("/DeepenData/Repos/geometric_cobra")

augmented_metabolite_data <- arrow::read_parquet("./results/dataframes/augmented_metabolite_data_v2.parquet.gzip")
```

```
## Error: IOError: Failed to open local file './results/dataframes/augmented_metabolite_data_v2.parquet.gzip'. Detail: [errno 2] No such file or directory
```

```r
augmented_metabolite_data <- rename(augmented_metabolite_data, Condicion = label) %>% dplyr::mutate(Condicion = if_else(Condicion== 1, 'PKU', 'Control'))


AAs_PKU_2        <- get_data_subset(augmented_metabolite_data, "PKU", T)
```

```
## Error in `dplyr::filter()`:
## ! Problem while computing `..1 = Group == condition`.
## Caused by error in `mask$eval_all_filter()`:
## ! object 'Group' not found
```

```r
AcylCs_PKU_2     <- get_data_subset(augmented_metabolite_data, "PKU", F)
```

```
## Error in `dplyr::filter()`:
## ! Problem while computing `..1 = Group == condition`.
## Caused by error in `mask$eval_all_filter()`:
## ! object 'Group' not found
```

```r
AAs_Control_2    <- get_data_subset(augmented_metabolite_data, "Control", T)
```

```
## Error in `dplyr::filter()`:
## ! Problem while computing `..1 = Group == condition`.
## Caused by error in `mask$eval_all_filter()`:
## ! object 'Group' not found
```

```r
AcylCs_Control_2 <- get_data_subset(augmented_metabolite_data, "Control", F)
```

```
## Error in `dplyr::filter()`:
## ! Problem while computing `..1 = Group == condition`.
## Caused by error in `mask$eval_all_filter()`:
## ! object 'Group' not found
```

```r
umap_augmented_data <- make_umap(AAs_Control_2, AcylCs_Control_2,AAs_PKU_2, AcylCs_PKU_2)
```

```
## Error in cbind(AAs_Control, AcylCs_Control): object 'AAs_Control_2' not found
```

```r
umap_augmented_data
```

```
## Error in eval(expr, envir, enclos): object 'umap_augmented_data' not found
```

```r
umap_panel <- ggarrange(umap_original_data, umap_augmented_data, common.legend = T, labels = c("e","f"), vjust = .5, hjust = .5)
```

```
## Error in ggarrange(umap_original_data, umap_augmented_data, common.legend = T, : object 'umap_original_data' not found
```

```r
setwd("/DeepenData/Repos/geometric_cobra")

img <- png::readPNG("./results/figures/RECON_graph.png")
```

```
## Error in png::readPNG("./results/figures/RECON_graph.png"): unable to open ./results/figures/RECON_graph.png
```

```r
graph <- ggplot() + background_image(img)
```

```
## Error in annotation_raster(raster.img, xmin = -Inf, xmax = Inf, ymin = -Inf, : object 'img' not found
```

```r
UMAP_plus_graph <- ggarrange(umap_panel, graph,FLUX_PKU_plot, ncol = 1, heights = c(1, 2.5, 1), labels = c("","g","h"))
```

```
## Error in ggarrange(umap_panel, graph, FLUX_PKU_plot, ncol = 1, heights = c(1, : object 'umap_panel' not found
```

```r
ggarrange(densityHeatmap_panel, UMAP_plus_graph, widths = c(.95,1), ncol = 2) -> panel
```

```
## Error in ggarrange(densityHeatmap_panel, UMAP_plus_graph, widths = c(0.95, : object 'densityHeatmap_panel' not found
```

```r
ggsave("/DeepenData/Repos/geometric_cobra/results/figures/Figure_1.png", 
       panel, height=8, width=9, bg = "white")
```

```
## Error in grid.draw(plot): object 'panel' not found
```



```r
setwd("/DeepenData/Repos/geometric_cobra")

augmented_metabolite_data <- arrow::read_parquet("./results/dataframes/augmented_metabolite_data_v2.parquet.gzip") 
```

```
## Error: IOError: Failed to open local file './results/dataframes/augmented_metabolite_data_v2.parquet.gzip'. Detail: [errno 2] No such file or directory
```

```r
augmented_metabolite_data
```

```
## # A tibble: 15,004 × 46
##    Condi…¹   Phe   Met   Val Leu.Ile   Tyr   Pro   Arg   Gly   Ala   Asp   Glu   Cit    Orn    SA    C0    C2    C3    C4 C4OH.…² C5.OH…³    C5  C5DC   C5.1
##    <chr>   <dbl> <dbl> <dbl>   <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl>  <dbl> <dbl> <dbl> <dbl> <dbl> <dbl>   <dbl>   <dbl> <dbl> <dbl>  <dbl>
##  1 PKU      238.  27.3 155.    178.   97   174.  17.3  206.  272.   14.9  157. 21.2  115.   0.26   27.0  7.52 0.63  0.11   0.02     0.13   0.11 0.02  0.06  
##  2 PKU      996.  26.4 154.    128.   97.8  99.6 18.8   22.3 112.   61.6  174. 26.1   92.6  0.21   20.5  9.63 0.95  0.11   0.02     0.12   0.11 0.02  0.07  
##  3 PKU     1462.  16.8 131.    160.   73.6  77.6 34.1  316.  314.   56.4  463. 11.2    6.89 0.187  24.4  9.79 0.963 0.171  0.0471   0.269  0.1  0.108 0.09  
##  4 PKU     1082   18.8  85.3   126.   46.0  93.8 13.0  155.   50.9  40.6  116. 12.2   38.9  0.55   21.8  5.54 0.68  0.13   0.05     0.13   0.07 0.06  0.06  
##  5 PKU     1467.  13.2 101.    142.   58.3 127.   1.67 133.  123.   44.3  365.  9.34  39.1  0.47   24.7  6.59 1.21  0.11   0.02     0.12   0.11 0.03  0.01  
##  6 PKU      689.  15.6 122.    125.   69.0  50.2  4.5  303.  281.   53.6  185.  5.09 108.   0      20.0  9.59 0.78  0.17   0.09     0.18   0.1  0.05  0.01  
##  7 PKU      362.  17.5 127.    165.   86.3 159.  14.7  302.  166.   38.0  381.  5.5   95.9  0.58   17.4  5.76 0.37  0.14   0.08     0.27   0.09 0.02  0     
##  8 PKU     1440.  19.9  66.8    85.7  62.5  93.8 19.4  215.  174.   16.5  235. 14.0   13.1  0.47   20.0  7.03 0.46  0.08   0.05     0.14   0.09 0.05  0.01  
##  9 PKU      761.  11.4 106.    127.   36.9  87.0  6.69 162.  104.   32.3  359. 23.7   63.5  0      18.1  9.2  0.81  0.16   0        0.16   0.11 0.12  0.0233
## 10 PKU      732.  19.7 102.    105.   45.8  86.4 12.4  132.  133.   25.3  146. 21.0   45.7  0.08   23.2 12.0  1.29  0.09   0.02     0.16   0.1  0.05  0.02  
## # … with 14,994 more rows, 22 more variables: C6 <dbl>, C6DC <dbl>, C8 <dbl>, C8.1 <dbl>, C10 <dbl>, C10.1 <dbl>, C10.2 <dbl>, C12 <dbl>, C12.1 <dbl>,
## #   C14 <dbl>, C14.1 <dbl>, C14.2 <dbl>, C14OH <dbl>, C16 <dbl>, C16OH <dbl>, C16.1 <dbl>, C16.1OH <dbl>, C18 <dbl>, C18OH <dbl>, C18.1 <dbl>,
## #   C18.1OH <dbl>, C18.2 <dbl>, and abbreviated variable names ¹​Condicion, ²​C4OH.C3DC, ³​C5.OH.C4DC
```




























