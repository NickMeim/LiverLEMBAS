library(tidyverse)
library(forcats)
library(doFuture)
# parallel: set number of workers
cores <- 16
registerDoFuture()
plan(multisession,workers = cores)
source("plotting_functions.R")

## Load KO data---------------------------
results_ko <- data.table::fread('../results/allmodels/InSilicoKOs/TFs_all_conditions.csv') %>% 
  column_to_rownames('V1')
results_ko <- results_ko %>% filter(type!="KO+control") %>% filter(type!='control')

## Load cosine W_opt, W_translatable
Wm_tot <- readRDS('../../FattyLiverModeling/results/Wm_Kostrzewski_total.rds')
Wm_opt <- readRDS('../../FattyLiverModeling/results/Wm_Kostrzewski_extra.rds')
Wm_combo <- readRDS('../../FattyLiverModeling/results/Wm_Kostrzewski_combo.rds')

## Infer TF activity
dorotheaData = read.table('../../FattyLiverModeling/data/dorothea.tsv', sep = "\t", header=TRUE)
confidenceFilter = is.element(dorotheaData$confidence, c('A', 'B'))
dorotheaData = dorotheaData[confidenceFilter,]
colnames(dorotheaData)[1] <- 'source'
colnames(Wm_opt) <- c('LV1','LV2')
colnames(Wm_combo) <- c('TC1','TC2')
# Estimate TF activities
# Estimate TF activities
TF_opt <- decoupleR::run_viper(Wm_opt, dorotheaData,minsize = 5,verbose = TRUE)
TF_opt <- TF_opt %>% select(source,condition,score) %>% spread(source,score) %>% 
  column_to_rownames('condition')
TF_opt <- t(TF_opt)
# saveRDS(TF_opt,'../../FattyLiverModeling/results/Wtfs_kostrzewski_extra.rds')
TF_tc <- decoupleR::run_viper(Wm_combo, dorotheaData,minsize = 5,verbose = TRUE)
TF_tc <- TF_tc %>% select(source,condition,score) %>% spread(source,score) %>% 
  column_to_rownames('condition')
TF_tc <- t(TF_tc)
#Load annotation file to replace with gene names
annot <- read.delim('../../../Artificial-Signaling-Network/DrugsANNSignaling/data/uniprot-reviewed_yes+AND+organism__Homo+sapiens+(Human)+[9606]_.tab')
annot <- annot %>% select(c('node'='Entry'),c('TF'='Gene.names...primary..')) %>% unique()
annot <- annot %>% filter(node %in% colnames(results_ko))
# annot <- annot %>% column_to_rownames('node')
names_annot <- annot$node
annot <- left_join(data.frame(node=colnames(results_act)[1:104]),annot)
print(all(colnames(results_act)[1:104]==annot$node))
colnames(results_act)[1:104] <- annot$TF
colnames(results_ko)[1:104] <- annot$TF

TF_opt <- TF_opt[which(rownames(TF_opt) %in% colnames(results_ko)),]
TF_opt <- TF_opt[colnames(results_ko)[1:104],]
TF_opt <- t(TF_opt)

TF_tc <- TF_tc[which(rownames(TF_tc) %in% colnames(results_ko)),]
TF_tc <- TF_tc[colnames(results_ko)[1:104],]
TF_tc <- t(TF_tc)


## Calculate the cosine similarity at each model for extra LVs------------------------
annot <- read.delim('../../../Artificial-Signaling-Network/DrugsANNSignaling/data/uniprot-reviewed_yes+AND+organism__Homo+sapiens+(Human)+[9606]_.tab')
annot <- annot %>% select(c('node'='Entry'),c('condition'='Gene.names...primary..')) %>% unique()
annot <- rbind(annot,
               data.frame(node=c('control','untreated','DMSO'),condition=c('control','untreated','DMSO')))
annot <- annot %>% mutate(condition = ifelse(condition=='',node,condition))
extraLV_sim <- foreach(i = seq(0,49)) %dopar% {
  expreriment <- as.matrix(results_ko %>% filter(model_no==i) %>% filter(condition!='untreated') %>%
                             select(-type,-model_no) %>% column_to_rownames('condition'))
  control <- as.matrix(results_ko %>% filter(model_no==i) %>% filter(condition=='untreated') %>%
                         select(-type,-model_no) %>% column_to_rownames('condition'))
  expreriment <- apply(expreriment,1,function(x) x-control)
  expreriment <- t(expreriment)
  colnames(expreriment) <- colnames(TF_opt)
  cosine_sim <- matrix(NA, nrow = nrow(expreriment), ncol = nrow(TF_opt))
  for (j in 1:nrow(TF_opt)) {
    cosine_sim[, j] <- apply(expreriment, 1, function(x) lsa::cosine(TF_opt[j, ], x))
  }
  colnames(cosine_sim) <- c('extraLV1','extraLV2')
  rownames(cosine_sim) <- rownames(expreriment)
  cosine_sim <- as.data.frame(cbind(cosine_sim,model_no=i)) %>% rownames_to_column('node')
  # cosine_sim <- cosine_sim %>% separate(node,into = c('node','suffix')) %>% select(-suffix)## only for KO+ control
  left_join(cosine_sim,annot) %>% select(condition,extraLV1,extraLV2,model_no)
}
extraLV_sim <- do.call(rbind,extraLV_sim)