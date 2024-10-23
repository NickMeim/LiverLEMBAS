library(tidyverse)
library(ggplot2)
library(ggpubr)
# library(ggrepel)
# library(ggbreak)
library(doFuture)
# parallel: set number of workers
cores <- 16
registerDoFuture()
plan(multisession,workers = cores)
source("plotting_functions.R")
source('functions_translation.R')
# library(ropls)
# library(caret)

### Load the signaling node activities-------------
#load node annotation
annot <- read.delim('../../../Artificial-Signaling-Network/DrugsANNSignaling/data/uniprot-reviewed_yes+AND+organism__Homo+sapiens+(Human)+[9606]_.tab')
annot <- annot %>% select(c('node'='Entry'),c('protein'='Gene.names...primary..')) %>% unique()

# load signaling nodes
singaling_activity <- data.table::fread('../results/allmodels/InSilicoKOs/signaling_nodes_conditions.csv') 
singaling_activity<- singaling_activity %>% select(-V1)
singaling_activity <- singaling_activity %>% filter(type!="KO+control") %>% filter(type!='control')
gc()
annot <- annot %>% filter(node %in% colnames(singaling_activity))
annot <- left_join(data.frame(node = colnames(singaling_activity)[1:1893]),annot)
annot_transformed <- annot %>% group_by(protein) %>% mutate(counts = n_distinct(node)) %>%
  ungroup()
annot_transformed <- annot_transformed %>% mutate(protein = ifelse(counts>1,node,protein)) %>%
  select(-counts)
print(all(colnames(singaling_activity)[1:1893]==annot_transformed$node))
colnames(singaling_activity)[1:1893] <- annot_transformed$protein
gc()

## Define MAS and fibrosis Regulons
gNAS <- c("ARPC5", "YWHAH", "ARF4", "TNFRSF12A", "ADHFE1",
          "USP33", "CD52", "ACVR2B", "ING5", "ASB3", "IFI30",
          "ERVW-1", "YWHAZ", "ERBB3", "KPNA2", "COQ10B", "MAGI1",
          "MAPRE1", "ABCA6")

gFib <- c("TIMP1", "MYLI2B", "LUM", "ZNF395", "AKAP9",
          "ACTR2", "LGALS3", "MAPRE1", "FRK", "ANKRD28",
          "IGFBP7", "YWHAZ", "USP33", "CD59", "TAX1BP3",
          "FAM221A", "ADHFE1", "TNFRSF12A")
# Build regulon-like object to do non-parametric enrichment with VIPER
sig_regulon <- rbind(
  data.frame(source = "MAS", target = gNAS, mor = c(1,1,1,1,-1,-1,1,-1,-1,-1,1,-1,1,-1,1,1,-1,1,-1)),
  data.frame(source = "Fibrosis stage", target = gFib, mor = c(1,1,1,-1,-1,1,1,1,-1,-1,1,1,-1,1,1,-1,-1,-1)))

# ### Infer MAS activity----------------------------
# for (i in 0:49){
#   expreriment <- as.matrix(singaling_activity %>% filter(model_no==i) %>% filter(condition!='untreated') %>%
#                              dplyr::select(-model_no,-type) %>% column_to_rownames('condition'))
#   sample_ids <- rownames(expreriment)
#   control <- as.matrix(singaling_activity %>% filter(model_no==i) %>% filter(condition=='untreated') %>%
#                          dplyr::select(-model_no,-type) %>% column_to_rownames('condition'))
#   expreriment <- apply(expreriment,1,function(x) x-control)
#   rownames(expreriment) <- colnames(singaling_activity)[1:1893]
#   colnames(expreriment) <- sample_ids
#   # Run decoupleR viper
#   x <- decoupleR::run_viper(expreriment, net = sig_regulon,minsize = 1) %>% 
#     select(source,condition,score) %>% spread(source,score) %>% 
#     column_to_rownames('condition')
#   print(paste0('Finished model ',i))
# }
# XKO_diff <- do.call(rbind,XKO)