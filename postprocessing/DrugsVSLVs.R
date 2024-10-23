library(dorothea)
library(tidyverse)
library(ggplot2)
library(ggpubr)
library(forcats)
library(doFuture)
# parallel: set number of workers
cores <- 16
registerDoFuture()
plan(multisession,workers = cores)
source("plotting_functions.R")

### Load in drug perturbations-------------------------------------------
TFdrugs <- data.table::fread('../results/allmodels/predictions/TFs_all_conditions.csv') %>% 
  column_to_rownames('V1') %>% separate(condition,into=c('drug','dose','time'),sep="\\+") %>%
  select(-time)
TFdrugs_control <- TFdrugs %>% filter(dose == 'Control') %>% select(-dose)
colnames(TFdrugs_control)[4] <- 'control_activity'
TFdrugs <- TFdrugs %>% filter(dose != 'Control')
TFdrugs <- left_join(TFdrugs,TFdrugs_control)
gc()
TFdrugs <- TFdrugs %>% mutate(activity = activity - control_activity) %>% select(-control_activity)

## Load cosine W_opt, W_translatable
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
annot <- annot %>% select(c('TF'='Entry'),c('TF_symbol'='Gene.names...primary..')) %>% unique()
annot <- annot %>% filter(TF %in% TFdrugs$TF)
# annot <- annot %>% column_to_rownames('node')
names_annot <- annot$TF
TFdrugs <- left_join(TFdrugs,annot) %>% select(-TF)
TFdrugs <- TFdrugs %>% select(drug,dose,model_no,c('TF'='TF_symbol'),activity)
# TFdrugs <- TFdrugs %>% spread('TF','activity')
gc()
TF_opt <- TF_opt[which(rownames(TF_opt) %in% TFdrugs$TF),]
TF_opt <- t(TF_opt)

TF_tc <- TF_tc[which(rownames(TF_tc) %in% TFdrugs$TF),]
TF_tc <- TF_tc[colnames(TF_opt),]
TF_tc <- t(TF_tc)


## Calculate the cosine similarity at each model for extra LVs------------------------
extraLV_sim <- foreach(i = seq(0,49)) %dopar% {
  expreriment <- as.matrix(TFdrugs %>% filter(model_no==i) %>% 
                             mutate(condition = paste0(drug,'+',dose)) %>%
                             select(-drug,-dose,-model_no) %>% 
                             spread('TF','activity') %>% 
                             column_to_rownames('condition'))
  expreriment <- expreriment[,colnames(TF_opt)]
  cosine_sim <- matrix(NA, nrow = nrow(expreriment), ncol = nrow(TF_opt))
  for (j in 1:nrow(TF_opt)) {
    cosine_sim[, j] <- apply(expreriment, 1, function(x) lsa::cosine(TF_opt[j, ], x))
  }
  colnames(cosine_sim) <- c('extraLV1','extraLV2')
  rownames(cosine_sim) <- rownames(expreriment)
  cosine_sim <- as.data.frame(cbind(cosine_sim,model_no=i)) %>% rownames_to_column('condition')
  cosine_sim %>% select(condition,extraLV1,extraLV2,model_no)
}
extraLV_sim <- do.call(rbind,extraLV_sim)

# visualize--------------------------------
summary  <-extraLV_sim %>% group_by(condition) %>% 
  summarise(mu1 = mean(extraLV1),mu2 = mean(extraLV2))

num_models <- length(unique(extraLV_sim$model_no))
plotting_LV1_extra <- extraLV_sim %>% gather('LV','cosine similarity',-condition,-model_no) %>% 
  filter(LV == "extraLV1") %>% group_by(condition) %>%
  mutate(mu = mean(`cosine similarity`),
         std = sd(`cosine similarity`),
         se = sd(`cosine similarity`)/sqrt(num_models)) %>% 
  ungroup() %>%
  select(condition,mu,std,se) %>% unique()
top20_lv1 <- plotting_LV1_extra$condition[order(-abs(plotting_LV1_extra$mu))][1:20]
plotting_LV1_extra <- plotting_LV1_extra %>%
  mutate(significant = ifelse(condition %in% top20_lv1, 
                              condition, NA))
plt_LV1_extra <- ggplot(plotting_LV1_extra,aes(x=as.numeric(reorder(condition,mu)),y=mu)) +
  geom_point(size=0.5,color = '#CD5C5C') +
  geom_ribbon(aes(ymax = mu+se,ymin=mu-se,x = as.numeric(reorder(condition, mu))),
              fill = '#CD5C5C',alpha=0.4)+
  geom_text_repel(aes(label=significant),size=0.75*size_annotation, max.overlaps=80,
                  box.padding = 0.5, nudge_x = 1, nudge_y = 0.1)+
  scale_y_continuous(breaks = seq(-0.5,0.5,0.1),limits = c(-0.5,0.5))+
  geom_hline(yintercept = 0,linetype='dashed',lwd=1,color='black')+
  labs(x = 'Rank', y = 'average cosine similarity')
plt_LV1_extra <- add_theme(plt_LV1_extra)
print(plt_LV1_extra)
ggsave(filename = "../figures/drugs_cosine_with_LV1.pdf", plot = plt_LV1_extra, units = "cm", width = 12, height = 12)
#### Repeat for LV2
plotting_LV2_extra <- extraLV_sim %>% gather('LV','cosine similarity',-condition,-model_no) %>% 
  filter(LV == "extraLV2") %>% group_by(condition) %>%
  mutate(mu = mean(`cosine similarity`),
         std = sd(`cosine similarity`),
         se = sd(`cosine similarity`)/sqrt(num_models)) %>% 
  ungroup() %>%
  select(condition,mu,std,se) %>% unique()
top20_lv2 <- plotting_LV2_extra$condition[order(-abs(plotting_LV2_extra$mu))][1:20]
plotting_LV2_extra <- plotting_LV2_extra %>%
  mutate(significant = ifelse(condition %in% top20_lv2, 
                              condition, NA))
plt_LV2_extra <- ggplot(plotting_LV2_extra,aes(x=as.numeric(reorder(condition,mu)),y=mu)) +
  geom_point(size=0.5,color = '#CD5C5C') +
  geom_ribbon(aes(ymax = mu+se,ymin=mu-se,x = as.numeric(reorder(condition, mu))),
              fill = '#CD5C5C',alpha=0.4)+
  geom_text_repel(aes(label=significant),size=0.75*size_annotation, max.overlaps=80,
                  box.padding = 0.5, nudge_x = 1, nudge_y = 0.1)+
  scale_y_continuous(breaks = seq(-0.5,0.5,0.1),limits = c(-0.5,0.5))+
  geom_hline(yintercept = 0,linetype='dashed',lwd=1,color='black')+
  labs(x = 'Rank', y = 'average cosine similarity')
plt_LV2_extra <- add_theme(plt_LV2_extra)
print(plt_LV2_extra)
ggsave(filename = "../figures/drugs_cosine_with_LV2.pdf", plot = plt_LV2_extra, units = "cm", width = 12, height = 12)

## Calculate the cosine similarity at each model for TCs------------------------
TC_sim <- foreach(i = seq(0,49)) %dopar% {
  expreriment <- as.matrix(TFdrugs %>% filter(model_no==i) %>% 
                             mutate(condition = paste0(drug,'+',dose)) %>%
                             select(-drug,-dose,-model_no) %>% 
                             spread('TF','activity') %>% 
                             column_to_rownames('condition'))
  expreriment <- expreriment[,colnames(TF_tc)]
  cosine_sim <- matrix(NA, nrow = nrow(expreriment), ncol = nrow(TF_tc))
  for (j in 1:nrow(TF_tc)) {
    cosine_sim[, j] <- apply(expreriment, 1, function(x) lsa::cosine(TF_tc[j, ], x))
  }
  colnames(cosine_sim) <- c('TC1','TC2')
  rownames(cosine_sim) <- rownames(expreriment)
  cosine_sim <- as.data.frame(cbind(cosine_sim,model_no=i)) %>% rownames_to_column('condition')
  cosine_sim %>% select(condition,TC1,TC2,model_no)
}
TC_sim <- do.call(rbind,TC_sim)

# visualize
summary  <-TC_sim %>% group_by(condition) %>% 
  summarise(mu1 = mean(TC1),mu2 = mean(TC2))


num_models <- length(unique(TC_sim$model_no))
plotting_TC1_extra <- TC_sim %>% gather('LV','cosine similarity',-condition,-model_no) %>% 
  filter(LV == "TC1") %>% group_by(condition) %>%
  mutate(mu = mean(`cosine similarity`),
         std = sd(`cosine similarity`),
         se = sd(`cosine similarity`)/sqrt(num_models)) %>% 
  ungroup() %>%
  select(condition,mu,std,se) %>% unique()
top20_tc1 <- plotting_TC1_extra$condition[order(-abs(plotting_TC1_extra$mu))][1:20]
plotting_TC1_extra <- plotting_TC1_extra %>%
  mutate(significant = ifelse(condition %in% top20_tc1, 
                              condition, NA))
plt_TC1_extra <- ggplot(plotting_TC1_extra,aes(x=as.numeric(reorder(condition,mu)),y=mu)) +
  geom_point(size=0.5,color = '#CD5C5C') +
  geom_ribbon(aes(ymax = mu+se,ymin=mu-se,x = as.numeric(reorder(condition, mu))),
              fill = '#CD5C5C',alpha=0.4)+
  geom_text_repel(aes(label=significant),size=0.75*size_annotation, max.overlaps=80,
                  box.padding = 0.5, nudge_x = 1, nudge_y = 0.1)+
  scale_y_continuous(breaks = seq(-0.5,0.5,0.1),limits = c(-0.5,0.5))+
  geom_hline(yintercept = 0,linetype='dashed',lwd=1,color='black')+
  labs(x = 'Rank', y = 'average cosine similarity')
plt_TC1_extra <- add_theme(plt_TC1_extra)
print(plt_TC1_extra)
ggsave(filename = "../figures/drugs_cosine_with_TC1.pdf", plot = plt_TC1_extra, units = "cm", width = 12, height = 12)

#### Repeat for LV2
plotting_TC2_extra <- TC_sim %>% gather('LV','cosine similarity',-condition,-model_no) %>% 
  filter(LV == "TC2") %>% group_by(condition) %>%
  mutate(mu = mean(`cosine similarity`),
         std = sd(`cosine similarity`),
         se = sd(`cosine similarity`)/sqrt(num_models)) %>% 
  ungroup() %>%
  select(condition,mu,std,se) %>% unique()
top20_tc2 <- plotting_TC2_extra$condition[order(-abs(plotting_TC2_extra$mu))][1:20]
plotting_TC2_extra <- plotting_TC2_extra %>%
  mutate(significant = ifelse(condition %in% top20_tc2, 
                              condition, NA))
plt_TC2_extra <- ggplot(plotting_TC2_extra,aes(x=as.numeric(reorder(condition,mu)),y=mu)) +
  geom_point(size=0.5,color = '#CD5C5C') +
  geom_ribbon(aes(ymax = mu+se,ymin=mu-se,x = as.numeric(reorder(condition, mu))),
              fill = '#CD5C5C',alpha=0.4)+
  geom_text_repel(aes(label=significant),size=0.75*size_annotation, max.overlaps=80,
                  box.padding = 0.5, nudge_x = 1, nudge_y = 0.1)+
  scale_y_continuous(breaks = seq(-0.5,0.5,0.1),limits = c(-0.5,0.5))+
  geom_hline(yintercept = 0,linetype='dashed',lwd=1,color='black')+
  labs(x = 'Rank', y = 'average cosine similarity')
plt_TC2_extra <- add_theme(plt_TC2_extra)
print(plt_TC2_extra)
ggsave(filename = "../figures/drugs_cosine_with_TC2.pdf", plot = plt_TC2_extra, units = "cm", width = 12, height = 12)