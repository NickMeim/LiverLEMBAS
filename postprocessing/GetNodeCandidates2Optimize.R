library(tidyverse)
library(forcats)
library(doFuture)
library(ropls)
library(patchwork)
# parallel: set number of workers
cores <- 16
registerDoFuture()
plan(multisession,workers = cores)
source("plotting_functions.R")
source('functions_translation.R')

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
annot <- left_join(data.frame(node=colnames(results_ko)[1:104]),annot)
print(all(colnames(results_ko)[1:104]==annot$node))
colnames(results_ko)[1:104] <- annot$TF
# reorder TFopt and TFtc
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
  left_join(cosine_sim,annot) %>% select(condition,extraLV1,extraLV2,model_no)
}
extraLV_sim <- do.call(rbind,extraLV_sim)

## get a nice dataframe that will be used for prioritization
num_models <- length(unique(extraLV_sim$model_no))
averaged_lv_similarity <- extraLV_sim %>% gather('LV','cosine similarity',-condition,-model_no) %>% 
  filter(LV == "extraLV1") %>% group_by(condition) %>%
  mutate(mu = mean(`cosine similarity`),
         std = sd(`cosine similarity`),
         se = sd(`cosine similarity`)/sqrt(num_models)) %>% 
  ungroup() %>%
  select(condition,mu,std,se) %>% unique()

#### Calculate cosine for TC1---------------------------------------
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

## get a nice dataframe that will be used for prioritization
num_models <- length(unique(TC_sim$model_no))
averaged_tc_similarity <- TC_sim %>% gather('LV','cosine similarity',-condition,-model_no) %>% 
  filter(LV == "TC1") %>% group_by(condition) %>%
  mutate(mu = mean(`cosine similarity`),
         std = sd(`cosine similarity`),
         se = sd(`cosine similarity`)/sqrt(num_models)) %>% 
  ungroup() %>%
  select(condition,mu,std,se) %>% unique()

### Rank and get candidates-------------------------------
tc_ranked <- averaged_tc_similarity %>% select(condition,mu) %>% unique()
tc_ranked <- tc_ranked %>% mutate(rank= rank(abs(mu))) # give high priority (small rank number) to those that are not important in TC1
lv_ranked <- averaged_lv_similarity  %>% select(condition,mu) %>% unique()
lv_ranked <- lv_ranked %>% mutate(rank= rank(-abs(mu)))
candidates_combo <- left_join(lv_ranked,tc_ranked,by='condition')
candidates_combo <- candidates_combo %>% mutate(rank = 0.5*(rank.x+rank.y))

## Get top 100 LV candiates for doing combos
final_candidates <- lv_ranked %>% filter(rank<=100) 
final_candidates <- left_join(final_candidates,annot)
saveRDS(final_candidates,'../results/allmodels/final_node_candidates.rds')

## Simulate combination of TGFBR1 and IFNAR1---------------
simulation <- data.table::fread('../results/allmodels/combo_simulation/TFs_all_conditions.csv') %>% 
  select(-V1)
simulation_samesign <- data.table::fread('../results/allmodels/combo_simulation_samesign/TFs_all_conditions.csv') %>% 
  select(-V1)
#Load annotation file to replace with gene names
annot <- read.delim('../../../Artificial-Signaling-Network/DrugsANNSignaling/data/uniprot-reviewed_yes+AND+organism__Homo+sapiens+(Human)+[9606]_.tab')
annot <- annot %>% select(c('node'='Entry'),c('TF'='Gene.names...primary..')) %>% unique()
annot <- annot %>% filter(node %in% colnames(simulation))
annot <- left_join(data.frame(node = colnames(simulation)[1:104]),annot)
print(all(colnames(simulation)[1:104]==annot$node))
colnames(simulation)[1:104] <- annot$TF
print(all(colnames(simulation_samesign)[1:104] ==annot$node ))
colnames(simulation_samesign)[1:104] <- annot$TF

# Load human data, infer TF activity and train model to predict MAS, Fibrosis from TFs
## Load the all the data to be used
dataset_names <- c("Govaere", "Kostrzewski", "Wang", "Feaver")
ref_dataset <- "Govaere"
target_dataset <- "Kostrzewski"
# Load
data_list <- load_datasets(dataset_names, dir_data = '../../FattyLiverModeling/data/')
# Run PCA
tmp <- process_datasets(data_list, filter_variance = F)
data_list <- tmp$data_list
plt_list <- tmp$plt_list
# Define matrices of interest
Yh <- as.matrix(data_list[[ref_dataset]]$metadata  %>% select(nas_score,Fibrosis_stage)) #keep both Fibrosis and NAS
colnames(Yh) <- c('NAS','fibrosis')
Xh <- data_list[[ref_dataset]]$data_center %>% t()
if (ref_dataset=='Hoang'){
  sex_inferred <- data_list[[ref_dataset]]$metadata$sex
}else if (ref_dataset=='Pantano') {
  sex_inferred <- data_list[[ref_dataset]]$metadata$Sex
}else{
  sex_inferred <- apply(as.matrix(Xh[,c('RPS4Y1')]),2,sign) 
  sex_inferred <- 1*(sex_inferred>0)
  sex_inferred <- ifelse(sex_inferred==1,'male','female')
}
# Get matrices of target dataset
Xm <- data_list[[target_dataset]]$data_center %>% t()
# Get Wm as the PC space of the MPS data when averaging tech replicates to capture variance due to experimental factors
Wm <- data_list[[target_dataset]]$Wm_group %>% as.matrix()

# Infer TF activity
dorotheaData = read.table('../../FattyLiverModeling/data/dorothea.tsv', sep = "\t", header=TRUE)
confidenceFilter = is.element(dorotheaData$confidence, c('A', 'B'))
dorotheaData = dorotheaData[confidenceFilter,]
colnames(dorotheaData)[1] <- 'source'
# Estimate TF activities
TF_human <- decoupleR::run_viper(t(Xh), dorotheaData,minsize = 5,verbose = TRUE)
TF_human <- TF_human %>% select(source,condition,score) %>% spread(source,score) %>% 
  column_to_rownames('condition')
TF_human <- as.matrix(TF_human)
## keep only TFs in LEMBAS
TF_human <- TF_human[,colnames(simulation)[1:104]]
## train one model with all the data
plsr_model <- opls(x = TF_human, 
                   y = Yh,
                   predI = 8,
                   crossvalI = 1,
                   scaleC = "center",
                   fig.pdfC = "none",
                   info.txtC = "none")
plot(Yh[,1],predict(plsr_model,TF_human)[,1])
plot(Yh[,2],predict(plsr_model,TF_human)[,2])
print(diag(cor(Yh,predict(plsr_model,TF_human))))
# Convert them to differential TF activity
annot <- read.delim('../../../Artificial-Signaling-Network/DrugsANNSignaling/data/uniprot-reviewed_yes+AND+organism__Homo+sapiens+(Human)+[9606]_.tab')
annot <- annot %>% select(c('node'='Entry'),c('condition'='Gene.names...primary..')) %>% unique()
annot <- rbind(annot,
               data.frame(node=c('control','untreated','DMSO'),condition=c('control','untreated','DMSO'))) %>% 
  filter(node %in% results_ko$condition)
annot <- annot %>% group_by(condition) %>% mutate(counts = n_distinct(node)) %>% ungroup() %>%
  mutate(condition = ifelse(counts>1,node,condition)) %>% select(-counts) %>% unique()
annot <- annot %>% mutate(condition = ifelse(condition=='',node,condition))
Yinsilico <- foreach(i = seq(0,49)) %dopar% {
  stimuli <- as.numeric(simulation$condition[simulation$model_no==i & simulation$type!='untreated'])
  expreriment <- as.matrix(simulation %>% filter(model_no==i) %>% filter(type!='untreated') %>%
                             select(-type,-model_no,-condition))
  expreriment[expreriment>1] <-1
  expreriment[expreriment<0] <-0
  expreriment <- log((expreriment+1e-6)/(1 - expreriment + 1e-6))
  colnames(expreriment) <- colnames(simulation)[1:104]
  ## predict
  yhat <- as.data.frame(predict(plsr_model,expreriment)) %>% mutate(model_no=i) %>% 
    mutate(IFNAR1 = stimuli) %>% mutate(TGFRB1 = -IFNAR1)
  yhat
}
Yinsilico <- do.call(rbind,Yinsilico)
control <- as.matrix(simulation %>% filter(type=='untreated') %>%
                       select(-type,-model_no,-condition))
control[control>1] <-1
control[control<0] <-0
control <- log((control+1e-6)/(1 - control + 1e-6))
colnames(control) <- colnames(simulation)[1:104]
Ycontrol <- predict(plsr_model,control)
mean_control_mas <- mean(Ycontrol[,1]) 
se_control_mas <- sd(Ycontrol[,1])/sqrt(50)
mean_control_fib <- mean(Ycontrol[,2]) 
se_control_fib <- mean(Ycontrol[,2]) /sqrt(50)
Yinsilico <- Yinsilico %>% gather('phenotype','score',-model_no,-TGFRB1,-IFNAR1) %>%
  group_by(phenotype,IFNAR1) %>% mutate(mu = mean(score)) %>% mutate(se=sd(score)/sqrt(50))

## repeat for when stimulating with same sign
annot <- read.delim('../../../Artificial-Signaling-Network/DrugsANNSignaling/data/uniprot-reviewed_yes+AND+organism__Homo+sapiens+(Human)+[9606]_.tab')
annot <- annot %>% select(c('node'='Entry'),c('condition'='Gene.names...primary..')) %>% unique()
annot <- rbind(annot,
               data.frame(node=c('control','untreated','DMSO'),condition=c('control','untreated','DMSO'))) %>% 
  filter(node %in% results_ko$condition)
annot <- annot %>% group_by(condition) %>% mutate(counts = n_distinct(node)) %>% ungroup() %>%
  mutate(condition = ifelse(counts>1,node,condition)) %>% select(-counts) %>% unique()
annot <- annot %>% mutate(condition = ifelse(condition=='',node,condition))
Yinsilico_samesign <- foreach(i = seq(0,49)) %dopar% {
  stimuli <- as.numeric(simulation_samesign$condition[simulation_samesign$model_no==i & simulation_samesign$type!='untreated'])
  expreriment <- as.matrix(simulation_samesign %>% filter(model_no==i) %>% filter(type!='untreated') %>%
                             select(-type,-model_no,-condition))
  expreriment[expreriment>1] <-1
  expreriment[expreriment<0] <-0
  expreriment <- log((expreriment+1e-6)/(1 - expreriment + 1e-6))
  colnames(expreriment) <- colnames(simulation_samesign)[1:104]
  ## predict
  yhat <- as.data.frame(predict(plsr_model,expreriment)) %>% mutate(model_no=i) %>% 
    mutate(IFNAR1 = stimuli) %>% mutate(TGFRB1 = -IFNAR1)
  yhat
}
Yinsilico_samesign <- do.call(rbind,Yinsilico_samesign)
Yinsilico_samesign <- Yinsilico_samesign %>% gather('phenotype','score',-model_no,-TGFRB1,-IFNAR1) %>%
  group_by(phenotype,IFNAR1) %>% mutate(mu = mean(score)) %>% mutate(se=sd(score)/sqrt(50))

### Visualize simulation
Yall <- rbind(Yinsilico %>% mutate(stimulation = 'IFNAR1*TGFBR1<0'),
              Yinsilico_samesign %>% mutate(stimulation = 'IFNAR1*TGFBR1>0'))%>%
  mutate(phenotype = ifelse(phenotype=='NAS','MAS','Fibrosis stage'))
# saveRDS(Yall,'../results/allmodels/combo_simulation_expanded/Yall_same_stim.rds')
mas_plot <- ggplot(Yall %>% filter(phenotype=='MAS') %>% select(IFNAR1,mu,se,stimulation,phenotype) %>% unique(),
       aes(x=IFNAR1,y=mu,color=stimulation,fill=stimulation))+
  xlab('stimulation strength') + ylab('score') + 
  scale_y_continuous(limits = c(min(0,min(Yall$score[Yall$phenotype=='MAS'])),8))+
  # geom_point(size=1)+
  geom_line()+
  # geom_errorbar(aes(ymax=mu+se,ymin=mu-se),alpha=0.5)+
  geom_ribbon(aes(ymax=mu+se,ymin=mu-se),alpha=0.5,color=NA)+
  geom_ribbon(aes(x=IFNAR1,
                  ymin = mean_control_mas - se_control_mas, 
                  ymax = mean_control_mas + se_control_mas), 
              fill = "grey",color='grey', alpha = 0.2,
              linetype = "dashed", show.legend = FALSE) +
  annotate('text',x=-3000,
           y=3.8,color= 'grey',fontface ='bold',
           label="untreated score", 
           hjust = 0 , 
           size=4)+
  facet_wrap(~phenotype)
mas_plot <- add_theme(mas_plot) + theme(legend.position = 'top',text = element_text(size=20))
print(mas_plot)
ggsave(filename = "../figures/ifna_tgfb_combo_mas_plot.pdf", plot = mas_plot, 
       device = cairo_pdf,
       units = "cm", width = 12, height = 12)

fib_plot <- ggplot(Yall %>% filter(phenotype!='MAS') %>% select(IFNAR1,mu,se,stimulation,phenotype) %>% unique(),
                   aes(x=IFNAR1,y=mu,color=stimulation,fill=stimulation))+
  xlab('stimulation strength') + ylab('score') + 
  scale_y_continuous(limits = c(min(0,min(Yall$score[Yall$phenotype!='MAS'])),5))+
  # geom_point(size=1)+
  geom_line()+
  # geom_errorbar(aes(ymax=mu+se,ymin=mu-se),alpha=0.5)+
  geom_ribbon(aes(ymax=mu+se,ymin=mu-se),alpha=0.5,color=NA)+
  geom_ribbon(aes(x=IFNAR1,
                  ymin = mean_control_fib - se_control_fib, 
                  ymax = mean_control_fib + se_control_fib), 
              fill = "grey",color='grey', alpha = 0.2,
              linetype = "dashed", show.legend = FALSE) +
  annotate('text',x=-3000,
           y=1.8,color= 'grey',fontface ='bold',
           label="untreated score", 
           hjust = 0 , 
           size=4)+
  facet_wrap(~phenotype)
fib_plot <- add_theme(fib_plot) + theme(legend.position = 'top',text = element_text(size=20))
print(fib_plot)
ggsave(filename = "../figures/ifna_tgfb_combo_fibrosis_plot.pdf", plot = fib_plot, 
       device = cairo_pdf,
       units = "cm", width = 12, height = 12)

p <- (mas_plot + fib_plot) +
  plot_layout(guides = "collect") &
  theme(legend.position = "top")
print(p)
ggsave(filename = "../figures/ifn_tgfb_combo.png", plot = p, 
       units = "cm", width = 24, height = 12)
ggsave(filename = "../figures/ifn_tgfb_combo.pdf", plot = p, device = cairo_pdf,
       units = "cm", width = 24, height = 12)  
