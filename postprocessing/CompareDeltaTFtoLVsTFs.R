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

### Load in silico results-------------------------------------------
# TFs_to_keep <- c('STAT1','STAT2','STAT3','STAT5A','STAT6','IRF9',
#                  'NFKB1','NFKB2','SMAD3','IRF1',
#                  'E2F2','E2F3','TP53','HIF1A','AR')
TFs_to_keep <- c('STAT2','STAT1','IRF9','IRF1')

#Load KOs results
results_ko <- data.table::fread('../results/allmodels/InSilicoKOs/TFs_all_conditions.csv') %>% 
  column_to_rownames('V1')
results_ko <- results_ko %>% filter(type!="KO+control") %>% filter(type!='control')
# untreated
# results_ko <- results_ko %>% gather('node','activity',-type,-condition,-model_no)
#Load activation results
results_act <- data.table::fread('../results/allmodels/InSilicoActivations/TFs_all_conditions.csv') %>% 
  column_to_rownames('V1')
results_act <- results_act %>% filter(type!="KO+control") %>% filter(type!='control')
# results_act <- results_act %>% gather('node','activity',-type,-condition,-model_no)

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

# visualize KOs--------------------------------
summary  <-extraLV_sim %>% group_by(condition) %>% 
  summarise(mu1 = mean(extraLV1),mu2 = mean(extraLV2))
  
plotting_df <- extraLV_sim %>%
  filter(condition %in% c("TGFBR1","TGFBR2",
                          'IFNGR1','IFNGR2',
                          'IFNAR1','IFNAR2',
                          "STAT1","STAT2",
                          "IRF1","IRF9",
                          "IL9R","IL21R","IL15RA","JAK3",
                          "TP53"))
plotting_df <- plotting_df %>% gather('LV','cosine similarity',-condition,-model_no)
plotting_df <- plotting_df %>%
  group_by(LV) %>%
  mutate(condition = fct_reorder(condition, abs(`cosine similarity`), .fun = median)) %>%
  ungroup()
ggboxplot(plotting_df, #%>% mutate(`cosine similarity` = abs(`cosine similarity`))
          x = 'condition',y='cosine similarity',color = 'condition',add = 'jitter')+
  ylab('cosine similarity')+
  scale_y_continuous(breaks = seq(-0.5,0.5,0.1),limits = c(-0.6,0.6))+
  geom_hline(yintercept = 0,linetype='solid',lwd=0.5,color='black')+
  stat_compare_means(comparisons = list(c('IFNAR2','TGFBR2'),
                                        c('IRF9','TGFBR2'),
                                        c('IRF1','TGFBR2')),
                     label.y = c(0.4,0.45,0.5),
                     method = 'wilcox.test')+
  facet_wrap(~LV)+
  theme(text = element_text(family = 'Arial',size=20),
        # axis.text.x = element_text(angle = 45),
        axis.text.x = element_blank(),
        axis.line.x = element_blank(),
        axis.ticks.x = element_blank())

num_models <- length(unique(extraLV_sim$model_no))
plotting_LV1_extra_KOs <- extraLV_sim %>% gather('LV','cosine similarity',-condition,-model_no) %>% 
  filter(LV == "extraLV1") %>% group_by(condition) %>%
  mutate(mu = mean(`cosine similarity`),
         std = sd(`cosine similarity`),
         se = sd(`cosine similarity`)/sqrt(num_models)) %>% 
  ungroup() %>%
  select(condition,mu,std,se) %>% unique()
top10_lv1 <- plotting_LV1_extra_KOs$condition[order(-abs(plotting_LV1_extra_KOs$mu))][1:10]
plotting_LV1_extra_KOs <- plotting_LV1_extra_KOs %>%
  mutate(significant = ifelse(condition %in% top10_lv1 | condition %in% c("TGFBR1","TGFBR2",
                                                                          'IFNGR1','IFNGR2',
                                                                          'IFNAR1','IFNAR2',
                                                                          "STAT1","STAT2",
                                                                          "IRF1","IRF9",
                                                                          "TP53"), 
                              condition, NA))
plt_LV1_extra_KOs <- ggplot(plotting_LV1_extra_KOs,aes(x=as.numeric(reorder(condition,mu)),y=mu)) +
  geom_point(size=0.5,color = '#CD5C5C') +
  geom_ribbon(aes(ymax = mu+se,ymin=mu-se,x = as.numeric(reorder(condition, mu))),
              fill = '#CD5C5C',alpha=0.4)+
  geom_text_repel(aes(label=significant),size=3.5, max.overlaps=80,
                  box.padding = 0.5, nudge_x = 1, nudge_y = 0.1)+
  scale_y_continuous(breaks = seq(-0.5,0.5,0.1),limits = c(-0.5,0.5))+
  geom_hline(yintercept = 0,linetype='dashed',lwd=1,color='black')+
  labs(x = 'Rank', y = 'average cosine similarity')
plt_LV1_extra_KOs <- add_theme(plt_LV1_extra_KOs) + theme(text = element_text(size=20))
print(plt_LV1_extra_KOs)
ggsave(filename = "../figures/KOs_cosine_with_LV1.pdf", plot = plt_LV1_extra_KOs, units = "cm", width = 12, height = 12)
#### Repeat for LV2
plotting_LV2_extra_KOs <- extraLV_sim %>% gather('LV','cosine similarity',-condition,-model_no) %>% 
  filter(LV == "extraLV2") %>% group_by(condition) %>%
  mutate(mu = mean(`cosine similarity`),
         std = sd(`cosine similarity`),
         se = sd(`cosine similarity`)/sqrt(num_models)) %>% 
  ungroup() %>%
  select(condition,mu,std,se) %>% unique()
top10_lv2 <- plotting_LV2_extra_KOs$condition[order(-abs(plotting_LV2_extra_KOs$mu))][1:10]
plotting_LV2_extra_KOs <- plotting_LV2_extra_KOs %>%
  mutate(significant = ifelse(condition %in% top10_lv2 | condition %in% c("TGFBR1","TGFBR2",
                                                                          'IFNGR1','IFNGR2',
                                                                          'IFNAR1','IFNAR2',
                                                                          "STAT1","STAT2",
                                                                          "IRF1","IRF9",
                                                                          "TP53"), 
                              condition, NA))
plt_LV2_extra_KOs <- ggplot(plotting_LV2_extra_KOs,aes(x=as.numeric(reorder(condition,mu)),y=mu)) +
  geom_point(size=0.5,color = '#CD5C5C') +
  geom_ribbon(aes(ymax = mu+se,ymin=mu-se,x = as.numeric(reorder(condition, mu))),
              fill = '#CD5C5C',alpha=0.4)+
  geom_text_repel(aes(label=significant),size=0.75*size_annotation, max.overlaps=80,
                  box.padding = 0.5, nudge_x = 1, nudge_y = 0.1)+
  scale_y_continuous(breaks = seq(-0.5,0.5,0.1),limits = c(-0.5,0.5))+
  geom_hline(yintercept = 0,linetype='dashed',lwd=1,color='black')+
  labs(x = 'Rank', y = 'average cosine similarity')
plt_LV2_extra_KOs <- add_theme(plt_LV2_extra_KOs)
print(plt_LV2_extra_KOs)
ggsave(filename = "../figures/KOs_cosine_with_LV2.pdf", plot = plt_LV2_extra_KOs, units = "cm", width = 12, height = 12)

#### Repeat for activation---------------------------------------------
extraLV_sim_act <- foreach(i = seq(0,49)) %dopar% {
  expreriment <- as.matrix(results_act %>% filter(model_no==i) %>% filter(condition!='untreated') %>%
                             select(-type,-model_no) %>% column_to_rownames('condition'))
  control <- as.matrix(results_act %>% filter(model_no==i) %>% filter(condition=='untreated') %>%
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
extraLV_sim_act <- do.call(rbind,extraLV_sim_act)

# visualize
summary  <-extraLV_sim_act %>% group_by(condition) %>% 
  summarise(mu1 = mean(extraLV1),mu2 = mean(extraLV2))

plotting_df <- extraLV_sim_act %>%
  filter(condition %in% c('untreated','control',
                          "TGFBR1","TGFBR2",
                          'IFNGR1','IFNGR2',
                          'IFNAR1','IFNAR2',
                          "IL10RA","IL6",
                          "STAT1","STAT2",
                          "IRF1","IRF9",
                          "E2F1","TP53","SRC","CRK",
                          "RAF1","SOCS3","JAK3","IL2RB"))
plotting_df <- plotting_df %>% gather('LV','cosine similarity',-condition,-model_no)
plotting_df <- plotting_df %>%
  group_by(LV) %>%
  mutate(condition = fct_reorder(condition, abs(`cosine similarity`), .fun = median)) %>%
  ungroup()
ggboxplot(plotting_df, #%>% mutate(`cosine similarity` = abs(`cosine similarity`))
          x = 'condition',y='cosine similarity',color = 'condition',add = 'jitter')+
  ylab('cosine similarity')+
  scale_y_continuous(breaks = seq(-0.5,0.5,0.1),limits = c(-0.6,0.6))+
  geom_hline(yintercept = 0,linetype='solid',lwd=0.5,color='black')+
  stat_compare_means(comparisons = list(c('STAT2','TGFBR1'),
                                        c('CRK','TGFBR2'),
                                        c('IFNAR1','TGFBR2')),
                     label.y = c(0.3,0.35,0.38),
                     method = 'wilcox.test')+
  facet_wrap(~LV)+
  theme(text = element_text(family = 'Arial',size=20),
        # axis.text.x = element_text(angle = 45),
        axis.text.x = element_blank(),
        axis.line.x = element_blank(),
        axis.ticks.x = element_blank())

num_models <- length(unique(extraLV_sim_act$model_no))
plotting_LV1_extra_acts <- extraLV_sim_act %>% gather('LV','cosine similarity',-condition,-model_no) %>% 
  filter(LV == "extraLV1") %>% group_by(condition) %>%
  mutate(mu = mean(`cosine similarity`),
         std = sd(`cosine similarity`),
         se = sd(`cosine similarity`)/sqrt(num_models)) %>% 
  ungroup() %>%
  select(condition,mu,std,se) %>% unique()
top10_lv1 <- plotting_LV1_extra_acts$condition[order(-abs(plotting_LV1_extra_acts$mu))][1:10]
plotting_LV1_extra_acts <- plotting_LV1_extra_acts %>%
  mutate(significant = ifelse(condition %in% top10_lv1 | condition %in% c("TGFBR1","TGFBR2",
                                                                          'IFNGR1','IFNGR2',
                                                                          'IFNAR1','IFNAR2',
                                                                          "STAT1","STAT2",
                                                                          "IRF1","IRF9",
                                                                          "TP53"), 
                              condition, NA))
plt_LV1_extra_acts <- ggplot(plotting_LV1_extra_acts,aes(x=as.numeric(reorder(condition,mu)),y=mu)) +
  geom_point(size=0.5,color = '#CD5C5C') +
  geom_ribbon(aes(ymax = mu+se,ymin=mu-se,x = as.numeric(reorder(condition, mu))),
              fill = '#CD5C5C',alpha=0.4)+
  geom_text_repel(aes(label=significant),size=0.75*size_annotation, max.overlaps=80,
                  box.padding = 0.5, nudge_x = 1, nudge_y = 0.1)+
  scale_y_continuous(breaks = seq(-0.4,0.4,0.1),limits = c(-0.35,0.3))+
  geom_hline(yintercept = 0,linetype='dashed',lwd=1,color='black')+
  labs(x = 'Rank', y = 'average cosine similarity')
plt_LV1_extra_acts <- add_theme(plt_LV1_extra_acts)
print(plt_LV1_extra_acts)
ggsave(filename = "../figures/Acts_cosine_with_LV1.pdf", plot = plt_LV1_extra_acts, units = "cm", width = 12, height = 12)

#### Repeat for LV2
plotting_LV2_extra_acts <- extraLV_sim_act %>% gather('LV','cosine similarity',-condition,-model_no) %>% 
  filter(LV == "extraLV2") %>% group_by(condition) %>%
  mutate(mu = mean(`cosine similarity`),
         std = sd(`cosine similarity`),
         se = sd(`cosine similarity`)/sqrt(num_models)) %>% 
  ungroup() %>%
  select(condition,mu,std,se) %>% unique()
top10_lv2 <- plotting_LV2_extra_acts$condition[order(-abs(plotting_LV2_extra_acts$mu))][1:10]
plotting_LV2_extra_acts <- plotting_LV2_extra_acts %>%
  mutate(significant = ifelse(condition %in% top10_lv2 | condition %in% c("TGFBR1","TGFBR2",
                                                                          'IFNGR1','IFNGR2',
                                                                          'IFNAR1','IFNAR2',
                                                                          "STAT1","STAT2",
                                                                          "IRF1","IRF9",
                                                                          "TP53"), 
                              condition, NA))
plt_LV2_extra_atcs <- ggplot(plotting_LV2_extra_acts,aes(x=as.numeric(reorder(condition,mu)),y=mu)) +
  geom_point(size=0.5,color = '#CD5C5C') +
  geom_ribbon(aes(ymax = mu+se,ymin=mu-se,x = as.numeric(reorder(condition, mu))),
              fill = '#CD5C5C',alpha=0.4)+
  geom_text_repel(aes(label=significant),size=0.75*size_annotation, max.overlaps=80,
                  box.padding = 0.5, nudge_x = 1, nudge_y = 0.1)+
  scale_y_continuous(breaks = seq(-0.5,0.5,0.1),limits = c(-0.5,0.5))+
  geom_hline(yintercept = 0,linetype='dashed',lwd=1,color='black')+
  labs(x = 'Rank', y = 'average cosine similarity')
plt_LV2_extra_atcs <- add_theme(plt_LV2_extra_atcs)
print(plt_LV2_extra_atcs)
ggsave(filename = "../figures/Acts_cosine_with_LV2.pdf", plot = plt_LV2_extra_atcs, units = "cm", width = 12, height = 12)

## Calculate the cosine similarity at each model for TCs------------------------
annot <- read.delim('../../../Artificial-Signaling-Network/DrugsANNSignaling/data/uniprot-reviewed_yes+AND+organism__Homo+sapiens+(Human)+[9606]_.tab')
annot <- annot %>% select(c('node'='Entry'),c('condition'='Gene.names...primary..')) %>% unique()
annot <- rbind(annot,
               data.frame(node=c('control','untreated','DMSO'),condition=c('control','untreated','DMSO')))
annot <- annot %>% mutate(condition = ifelse(condition=='',node,condition))
TC_sim <- foreach(i = seq(0,49)) %dopar% {
  expreriment <- as.matrix(results_ko %>% filter(model_no==i) %>% filter(condition!='untreated') %>%
                             select(-type,-model_no) %>% column_to_rownames('condition'))
  control <- as.matrix(results_ko %>% filter(model_no==i) %>% filter(condition=='untreated') %>%
                         select(-type,-model_no) %>% column_to_rownames('condition'))
  expreriment <- apply(expreriment,1,function(x) x-control)
  expreriment <- t(expreriment)
  colnames(expreriment) <- colnames(TF_tc)
  cosine_sim <- matrix(NA, nrow = nrow(expreriment), ncol = nrow(TF_tc))
  for (j in 1:nrow(TF_tc)) {
    cosine_sim[, j] <- apply(expreriment, 1, function(x) lsa::cosine(TF_tc[j, ], x))
  }
  colnames(cosine_sim) <- c('TC1','TC2')
  rownames(cosine_sim) <- rownames(expreriment)
  cosine_sim <- as.data.frame(cbind(cosine_sim,model_no=i)) %>% rownames_to_column('node')
  # cosine_sim <- cosine_sim %>% separate(node,into = c('node','suffix')) %>% select(-suffix)## only for KO+ control
  left_join(cosine_sim,annot) %>% select(condition,TC1,TC2,model_no)
}
TC_sim <- do.call(rbind,TC_sim)

# visualize
summary  <-TC_sim %>% group_by(condition) %>% 
  summarise(mu1 = mean(TC1),mu2 = mean(TC2))

plotting_df <- TC_sim %>%
  filter(condition %in% c("TGFBR1","TGFBR2",
                          'IFNGR1','IFNGR2',
                          'IFNAR1','IFNAR2',
                          "IL10RA","IL6",
                          "STAT1","STAT2",
                          "IRF1","IRF9",
                          "SPI1","ERG","TP53","SMAD3",
                          "CEBPB"))
plotting_df <- plotting_df %>% gather('LV','cosine similarity',-condition,-model_no)
plotting_df <- plotting_df %>%
  group_by(LV) %>%
  mutate(condition = fct_reorder(condition, `cosine similarity`, .fun = median)) %>%
  ungroup()
ggboxplot(plotting_df%>% mutate(`cosine similarity` = `cosine similarity`), #%>% mutate(`cosine similarity` = abs(`cosine similarity`))
          x = 'condition',y='cosine similarity',color = 'condition',add = 'jitter')+
  ylab('cosine similarity')+
  scale_y_continuous(breaks = seq(-0.4,0.4,0.1),limits = c(-0.5,0.5))+
  geom_hline(yintercept = 0,linetype='solid',lwd=0.5,color='black')+
  stat_compare_means(comparisons = list(c('STAT2','TGFBR1'),
                                        c('STAT1','TGFBR1'),
                                        c('IRF1','TGFBR1')),
                     label.y = c(0.3,0.35,0.38),
                     method = 'wilcox.test')+
  facet_wrap(~LV)+
  theme(text = element_text(family = 'Arial',size=20),
        # axis.text.x = element_text(angle = 45),
        axis.text.x = element_blank(),
        axis.line.x = element_blank(),
        axis.ticks.x = element_blank())


num_models <- length(unique(TC_sim$model_no))
plotting_TC1_extra_KOs <- TC_sim %>% gather('LV','cosine similarity',-condition,-model_no) %>% 
  filter(LV == "TC1") %>% group_by(condition) %>%
  mutate(mu = mean(`cosine similarity`),
         std = sd(`cosine similarity`),
         se = sd(`cosine similarity`)/sqrt(num_models)) %>% 
  ungroup() %>%
  select(condition,mu,std,se) %>% unique()
top10_tc1 <- plotting_TC1_extra_KOs$condition[order(-abs(plotting_TC1_extra_KOs$mu))][1:10]
plotting_TC1_extra_KOs <- plotting_TC1_extra_KOs %>%
  mutate(significant = ifelse(condition %in% top10_tc1 | condition %in% c("TGFBR1","TGFBR2",
                                                                          'IFNGR1','IFNGR2',
                                                                          'IFNAR1','IFNAR2',
                                                                          "STAT1","STAT2",
                                                                          "IRF1","IRF9",
                                                                          "TP53"), 
                              condition, NA))
plt_TC1_extra_KOs <- ggplot(plotting_TC1_extra_KOs,aes(x=as.numeric(reorder(condition,mu)),y=mu)) +
  geom_point(size=0.5,color = '#CD5C5C') +
  geom_ribbon(aes(ymax = mu+se,ymin=mu-se,x = as.numeric(reorder(condition, mu))),
              fill = '#CD5C5C',alpha=0.4)+
  geom_text_repel(aes(label=significant),size=0.75*size_annotation, max.overlaps=80,
                  box.padding = 0.5, nudge_x = 1, nudge_y = 0.1)+
  scale_y_continuous(breaks = seq(-0.5,0.5,0.1),limits = c(-0.5,0.5))+
  geom_hline(yintercept = 0,linetype='dashed',lwd=1,color='black')+
  labs(x = 'Rank', y = 'average cosine similarity')
plt_TC1_extra_KOs <- add_theme(plt_TC1_extra_KOs)
print(plt_TC1_extra_KOs)
ggsave(filename = "../figures/KOs_cosine_with_TC1.pdf", plot = plt_TC1_extra_KOs, units = "cm", width = 12, height = 12)

#### Repeat for LV2
plotting_TC2_extra_KOs <- TC_sim %>% gather('LV','cosine similarity',-condition,-model_no) %>% 
  filter(LV == "TC2") %>% group_by(condition) %>%
  mutate(mu = mean(`cosine similarity`),
         std = sd(`cosine similarity`),
         se = sd(`cosine similarity`)/sqrt(num_models)) %>% 
  ungroup() %>%
  select(condition,mu,std,se) %>% unique()
top10_tc2 <- plotting_TC2_extra_KOs$condition[order(-abs(plotting_TC2_extra_KOs$mu))][1:10]
plotting_TC2_extra_KOs <- plotting_TC2_extra_KOs %>%
  mutate(significant = ifelse(condition %in% top10_tc2 | condition %in% c("TGFBR1","TGFBR2",
                                                                          'IFNGR1','IFNGR2',
                                                                          'IFNAR1','IFNAR2',
                                                                          "STAT1","STAT2",
                                                                          "IRF1","IRF9",
                                                                          "TP53"), 
                              condition, NA))
plt_TC2_extra_KOs <- ggplot(plotting_TC2_extra_KOs,aes(x=as.numeric(reorder(condition,mu)),y=mu)) +
  geom_point(size=0.5,color = '#CD5C5C') +
  geom_ribbon(aes(ymax = mu+se,ymin=mu-se,x = as.numeric(reorder(condition, mu))),
              fill = '#CD5C5C',alpha=0.4)+
  geom_text_repel(aes(label=significant),size=0.75*size_annotation, max.overlaps=80,
                  box.padding = 0.5, nudge_x = 1, nudge_y = 0.1)+
  scale_y_continuous(breaks = seq(-0.5,0.5,0.1),limits = c(-0.5,0.5))+
  geom_hline(yintercept = 0,linetype='dashed',lwd=1,color='black')+
  labs(x = 'Rank', y = 'average cosine similarity')
plt_TC2_extra_KOs <- add_theme(plt_TC2_extra_KOs)
print(plt_TC2_extra_KOs)
ggsave(filename = "../figures/KOs_cosine_with_TC2.pdf", plot = plt_TC2_extra_KOs, units = "cm", width = 12, height = 12)




#### Repeat for activation----------------------------------------
TC_sim_act <- foreach(i = seq(0,49)) %dopar% {
  expreriment <- as.matrix(results_act %>% filter(model_no==i) %>% filter(condition!='untreated') %>%
                             select(-type,-model_no) %>% column_to_rownames('condition'))
  control <- as.matrix(results_act %>% filter(model_no==i) %>% filter(condition=='untreated') %>%
                         select(-type,-model_no) %>% column_to_rownames('condition'))
  expreriment <- apply(expreriment,1,function(x) x-control)
  expreriment <- t(expreriment)
  colnames(expreriment) <- colnames(TF_tc)
  cosine_sim <- matrix(NA, nrow = nrow(expreriment), ncol = nrow(TF_tc))
  for (j in 1:nrow(TF_tc)) {
    cosine_sim[, j] <- apply(expreriment, 1, function(x) lsa::cosine(TF_tc[j, ], x))
  }
  colnames(cosine_sim) <- c('TC1','TC2')
  rownames(cosine_sim) <- rownames(expreriment)
  cosine_sim <- as.data.frame(cbind(cosine_sim,model_no=i)) %>% rownames_to_column('node')
  # cosine_sim <- cosine_sim %>% separate(node,into = c('node','suffix')) %>% select(-suffix)
  left_join(cosine_sim,annot) %>% select(condition,TC1,TC2,model_no)
}
TC_sim_act <- do.call(rbind,TC_sim_act)

# visualize
summary  <-TC_sim_act %>% group_by(condition) %>% 
  summarise(mu1 = mean(TC1),mu2 = mean(TC2))

plotting_df <- TC_sim_act %>%
  filter(condition %in% c('untreated','control',
                          "TGFBR1","TGFBR2",
                          'IFNGR1','IFNGR2',
                          'IFNAR1','IFNAR2',
                          "IL10RA","IL6",
                          "STAT1","STAT2",
                          "IRF1","IRF9",
                          "SPI1","ERG","TP53","SMAD3",
                          "CEBPB"))
plotting_df <- plotting_df %>% gather('LV','cosine similarity',-condition,-model_no)
plotting_df <- plotting_df %>%
  group_by(LV) %>%
  mutate(condition = fct_reorder(condition, `cosine similarity`, .fun = median)) %>%
  ungroup()
ggboxplot(plotting_df, #%>% mutate(`cosine similarity` = abs(`cosine similarity`))
          x = 'condition',y='cosine similarity',color = 'condition',add = 'jitter')+
  ylab('cosine similarity')+
  scale_y_continuous(breaks = seq(-0.4,0.4,0.1),limits = c(-0.5,0.5))+
  geom_hline(yintercept = 0,linetype='solid',lwd=0.5,color='black')+
  stat_compare_means(comparisons = list(c('STAT2','TGFBR1'),
                                        c('STAT1','TGFBR2'),
                                        c('IFNAR1','TGFBR2')),
                     label.y = c(0.3,0.35,0.38),
                     method = 'wilcox.test')+
  facet_wrap(~LV)+
  theme(text = element_text(family = 'Arial',size=20),
        # axis.text.x = element_text(angle = 45),
        axis.text.x = element_blank(),
        axis.line.x = element_blank(),
        axis.ticks.x = element_blank())

num_models <- length(unique(TC_sim_act$model_no))
plotting_TC1_extra_acts <- TC_sim_act %>% gather('LV','cosine similarity',-condition,-model_no) %>% 
  filter(LV == "TC1") %>% group_by(condition) %>%
  mutate(mu = mean(`cosine similarity`),
         std = sd(`cosine similarity`),
         se = sd(`cosine similarity`)/sqrt(num_models)) %>% 
  ungroup() %>%
  select(condition,mu,std,se) %>% unique()
top10_tc1 <- plotting_TC1_extra_acts$condition[order(-abs(plotting_TC1_extra_acts$mu))][1:10]
plotting_TC1_extra_acts <- plotting_TC1_extra_acts %>%
  mutate(significant = ifelse(condition %in% top10_tc1 | condition %in% c("TGFBR1","TGFBR2",
                                                                          'IFNGR1','IFNGR2',
                                                                          'IFNAR1','IFNAR2',
                                                                          "STAT1","STAT2",
                                                                          "IRF1","IRF9",
                                                                          "TP53"), 
                              condition, NA))
plt_TC1_extra_acts <- ggplot(plotting_TC1_extra_acts,aes(x=as.numeric(reorder(condition,mu)),y=mu)) +
  geom_point(size=0.5,color = '#CD5C5C') +
  geom_ribbon(aes(ymax = mu+se,ymin=mu-se,x = as.numeric(reorder(condition, mu))),
              fill = '#CD5C5C',alpha=0.4)+
  geom_text_repel(aes(label=significant),size=0.75*size_annotation, max.overlaps=80,
                  box.padding = 0.5, nudge_x = 1, nudge_y = 0.1)+
  scale_y_continuous(breaks = seq(-0.5,0.5,0.1),limits = c(-0.5,0.5))+
  geom_hline(yintercept = 0,linetype='dashed',lwd=1,color='black')+
  labs(x = 'Rank', y = 'average cosine similarity')
plt_TC1_extra_acts <- add_theme(plt_TC1_extra_acts)
print(plt_TC1_extra_acts)
ggsave(filename = "../figures/Acts_cosine_with_TC1.pdf", plot = plt_TC1_extra_acts, units = "cm", width = 12, height = 12)

#### Repeat for LV2
plotting_TC2_extra_acts <- TC_sim_act %>% gather('LV','cosine similarity',-condition,-model_no) %>% 
  filter(LV == "TC2") %>% group_by(condition) %>%
  mutate(mu = mean(`cosine similarity`),
         std = sd(`cosine similarity`),
         se = sd(`cosine similarity`)/sqrt(num_models)) %>% 
  ungroup() %>%
  select(condition,mu,std,se) %>% unique()
top10_tc2 <- plotting_TC2_extra_acts$condition[order(-abs(plotting_TC2_extra_acts$mu))][1:10]
plotting_TC2_extra_acts <- plotting_TC2_extra_acts %>%
  mutate(significant = ifelse(condition %in% top10_tc2 | condition %in% c("TGFBR1","TGFBR2",
                                                                          'IFNGR1','IFNGR2',
                                                                          'IFNAR1','IFNAR2',
                                                                          "STAT1","STAT2",
                                                                          "IRF1","IRF9",
                                                                          "TP53"), 
                              condition, NA))
plt_TC2_extra_acts <- ggplot(plotting_TC2_extra_acts,aes(x=as.numeric(reorder(condition,mu)),y=mu)) +
  geom_point(size=0.5,color = '#CD5C5C') +
  geom_ribbon(aes(ymax = mu+se,ymin=mu-se,x = as.numeric(reorder(condition, mu))),
              fill = '#CD5C5C',alpha=0.4)+
  geom_text_repel(aes(label=significant),size=0.75*size_annotation, max.overlaps=80,
                  box.padding = 0.5, nudge_x = 1, nudge_y = 0.1)+
  scale_y_continuous(breaks = seq(-0.5,0.5,0.1),limits = c(-0.5,0.5))+
  geom_hline(yintercept = 0,linetype='dashed',lwd=1,color='black')+
  labs(x = 'Rank', y = 'average cosine similarity')
plt_TC2_extra_acts <- add_theme(plt_TC2_extra_acts)
print(plt_TC2_extra_acts)
ggsave(filename = "../figures/Acts_cosine_with_TC2.pdf", plot = plt_TC2_extra_acts, units = "cm", width = 12, height = 12)




### Compare only IFN, TGFb (and markers like IRF1, IRF9, STAT1, STAT2 and TGFb markers)--------------
plotting_df <- extraLV_sim %>%
  filter(condition %in% c("TGFBR1","TGFBR2",
                          'IFNGR1','IFNGR2',
                          'IFNAR1','IFNAR2',
                          'JAK3','TYK2',
                          "STAT1","STAT2",
                          "IRF1","IRF9"))
plotting_df <- plotting_df %>% gather('LV','cosine similarity',-condition,-model_no)
plotting_df$condition <- factor(plotting_df$condition,
                                levels = c("TGFBR1","TGFBR2",
                                           'IFNGR1','IFNGR2',
                                           'IFNAR1','IFNAR2',
                                           'JAK3','TYK2',
                                           "STAT1","STAT2",
                                           "IRF1","IRF9"))
boxplotLV1 <- ggboxplot(plotting_df %>% filter(LV=='extraLV1'),
          x = 'condition',y='cosine similarity',color = 'condition',add = 'jitter')+
  ylab('cosine similarity')+
  scale_y_continuous(breaks = seq(-0.5,0.5,0.1),limits = c(-0.3,0.6))+
  geom_hline(yintercept = 0,linetype='solid',lwd=0.5,color='black')+
  stat_compare_means(comparisons = list(c('IFNGR1','TGFBR2'),
                                        c('IFNGR2','TGFBR2')),
                     label.y = c(0.42,0.5),
                     method = 'wilcox.test')+
  theme(text = element_text(family = 'Arial',size=20),
        # axis.text.x = element_text(angle = 45),
        axis.text.x = element_blank(),
        axis.line.x = element_blank(),
        axis.ticks.x = element_blank())
boxplotLV1 <- add_theme(boxplotLV1) + 
  theme_pubr() + 
  theme(text = element_text(size=20),
        axis.text.x = element_blank(),
        axis.title.x = element_blank(),
        legend.position = 'bottom')
print(boxplotLV1)
ggsave(filename = "../figures/compare_tgfb_ifn.pdf", plot = boxplotLV1, units = "cm", width = 18, height = 14)

### repeat for TC1
plotting_df <- TC_sim %>%
  filter(condition %in% c("TGFBR1","TGFBR2",
                          'IFNGR1','IFNGR2',
                          'IFNAR1','IFNAR2',
                          'JAK3','TYK2',
                          "STAT1","STAT2",
                          "IRF1","IRF9"))
plotting_df <- plotting_df %>% gather('LV','cosine similarity',-condition,-model_no)
plotting_df$condition <- factor(plotting_df$condition,
                                levels = c("TGFBR1","TGFBR2",
                                           'IFNGR1','IFNGR2',
                                           'IFNAR1','IFNAR2',
                                           'JAK3','TYK2',
                                           "STAT1","STAT2",
                                           "IRF1","IRF9"))
boxplotTC1 <- ggboxplot(plotting_df %>% filter(LV=='TC1'),
                        x = 'condition',y='cosine similarity',color = 'condition',add = 'jitter')+
  ylab('cosine similarity')+
  scale_y_continuous(breaks = seq(-0.5,0.5,0.1),limits = c(-0.3,0.6))+
  geom_hline(yintercept = 0,linetype='solid',lwd=0.5,color='black')+
  stat_compare_means(comparisons = list(c('IFNGR1','TGFBR2'),
                                        c('IFNGR2','TGFBR2')),
                     label.y = c(0.42,0.5),
                     method = 'wilcox.test')+
  theme(text = element_text(family = 'Arial',size=20),
        # axis.text.x = element_text(angle = 45),
        axis.text.x = element_blank(),
        axis.line.x = element_blank(),
        axis.ticks.x = element_blank())
boxplotTC1 <- add_theme(boxplotTC1) + theme_pubr() + theme(axis.text.x = element_blank(),
                                                           axis.title.x = element_blank(),
                                                           legend.position = 'bottom')
print(boxplotTC1)
ggsave(filename = "../figures/compare_tgfb_ifn_TCs.pdf", plot = boxplotTC1, units = "cm", width = 16, height = 12)

### compare similarity between LV1 and TC1
plotting_df_lv <- extraLV_sim %>%
  filter(condition %in% c("TGFBR1","TGFBR2",
                          'IFNAR1','IFNAR2')) %>%
  select(-extraLV2)
plotting_df_tc <- TC_sim %>%
  filter(condition %in% c("TGFBR1","TGFBR2",
                          'IFNAR1','IFNAR2'))%>%
  select(-TC2)

plotting_df_all <- left_join(plotting_df_lv,plotting_df_tc)
plotting_df_all <- plotting_df_all %>% 
  gather('extra basis compared','cosine similarity',-condition,-model_no)
plotting_df_all$condition <- factor(plotting_df_all$condition,
                                   levels = c("TGFBR1","TGFBR2",
                                              'IFNAR1','IFNAR2'))
ggboxplot(plotting_df_all,
          x = 'condition',y='cosine similarity',color = 'extra basis compared',add = 'jitter')+
  ylab('cosine similarity')+
  scale_y_continuous(breaks = seq(-0.5,0.5,0.1),limits = c(-0.5,0.6))+
  geom_hline(yintercept = 0,linetype='solid',lwd=0.5,color='black')+
  stat_compare_means(aes(group = `extra basis compared`),
                     label = 'p.signif',
                     method = 'wilcox.test',
                     paired = TRUE)+
  theme(text = element_text(family = 'Arial',size=20),
        axis.text.x = element_text(angle = 45),
        axis.line.x = element_blank(),
        axis.ticks.x = element_blank())
ggsave('../figures/TC1_vs_LV1.png',
       units = "cm", width = 16, height = 12,
       dpi = 600)

