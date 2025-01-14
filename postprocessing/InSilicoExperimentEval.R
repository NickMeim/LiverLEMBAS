library(tidyverse)
library(ggplot2)
library(ggpubr)
library(ggrepel)
library(ggbreak)
library(doFuture)
# parallel: set number of workers
cores <- 16
registerDoFuture()
plan(multisession,workers = cores)
source("plotting_functions.R")
source('functions_translation.R')
library(ropls)
library(caret)
### Load in silico results-------------------------------------------
# Load KOs results
results_ko <- data.table::fread('../results/2 hour/allmodels/InSilicoKOs/TFs_all_conditions.csv') %>% 
  column_to_rownames('V1')
results_ko <- results_ko %>% filter(type!="KO+control") %>% filter(type!='control')
# untreated
#Load activation results
results_act <- data.table::fread('../results/2 hour/allmodels/InSilicoActivations/TFs_all_conditions.csv') %>% 
  column_to_rownames('V1')
results_act <- results_act %>% filter(type!="KO+control") %>% filter(type!='control')
#Load annotation file to replace with gene names
annot <- read.delim('../../../Artificial-Signaling-Network/DrugsANNSignaling/data/uniprot-reviewed_yes+AND+organism__Homo+sapiens+(Human)+[9606]_.tab')
annot <- annot %>% select(c('node'='Entry'),c('TF'='Gene.names...primary..')) %>% unique()
annot <- annot %>% filter(node %in% colnames(results_ko))
annot <- left_join(data.frame(node = colnames(results_act)[1:104]),annot)
print(all(colnames(results_act)[1:104]==annot$node))
colnames(results_act)[1:104] <- annot$TF
colnames(results_ko)[1:104] <- annot$TF

# Load human data, infer TF activity and train model to predict MAS, Fibrosis from TFs-------------------------------------------
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
TF_human <- TF_human[,colnames(results_ko)[1:104]]
# TF_human <- 1/(1+exp(-TF_human))

# Train PLSR model to predict Yh from TFS
# Use first CV existing splits 
cv_res <- data.frame()
for (i in 1:10){
  gex_train <- readRDS(paste0('../../FattyLiverModeling/preprocessing/TrainingValidationData/WholePipeline/crossfoldPLSR/Xh_train',i,'.rds'))
  train_inds <- which(rownames(TF_human) %in% rownames(gex_train))
  x_train <- TF_human[train_inds,]
  y_train <- Yh[train_inds,]
  x_test <- TF_human[-train_inds,]
  y_test <- Yh[-train_inds,]
  
  plsr_model <- opls(x = x_train, 
                     y = y_train,
                     predI = 8,
                     crossvalI = 1,
                     scaleC = "center",
                     fig.pdfC = "none",
                     info.txtC = "none")
  yhat <- predict(plsr_model,x_train)
  train_r <- diag(cor(yhat,y_train))
  # evaluate in test set
  yhat_test <- predict(plsr_model,x_test)
  test_r <- diag(cor(yhat_test,y_test))
  
  ## shuffle labels model
  y_train_shuffled <- y_train[sample.int(nrow(y_train)),]
  rownames(y_train_shuffled) <- rownames(y_train)
  plsr_model <- opls(x = x_train, 
                     y = y_train_shuffled,
                     predI = 8,
                     crossvalI = 1,
                     scaleC = "center",
                     fig.pdfC = "none",
                     info.txtC = "none")
  # evaluate in test set
  yhat_test <- predict(plsr_model,x_test)
  test_shuffled <- diag(cor(yhat_test,y_test))
  
  # combine results
  cv_res <- rbind(cv_res,
                  data.frame(train = train_r,test = test_r, shuffled = test_shuffled) %>% 
                    rownames_to_column('phenotype') %>% 
                    mutate(phenotype=ifelse(phenotype=='NAS','MAS','Fibrosis stage')) %>% 
                    mutate(fold=i) %>%
                    select(fold,phenotype,train,test,c('shuffled Y'='shuffled')))
  
}
cv_res <- cv_res %>% gather('set','r',-fold,-phenotype)
cv_plot <- ggboxplot(cv_res,x='set',y='r',color = 'set',add='jitter')+
  scale_y_continuous(breaks = seq(-0.5,1,0.1),limits = c(NA,1.1)) +
  ylab('pearson correlation')+
  geom_hline(yintercept = 0,linetype='dashed',lwd=1,color='black')+
  stat_compare_means(comparisons = list(c('train','test'),
                                        c('test','shuffled Y'),
                                        c('train','shuffled Y')),
                     method = 'wilcox.test',
                     paired = TRUE,
                     label.y = c(0.8,0.8,0.95))+
  facet_wrap(~phenotype)
cv_plot <- add_theme(cv_plot) + theme(legend.position = 'none')
print(cv_plot)
# ggsave(filename = "../figures/plsr_human_TFs_CV.png", plot = cv_plot, units = "cm", width = 9, height = 6)
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

# Convert them to differential TF activity--------------------------------------
annot <- read.delim('../../../Artificial-Signaling-Network/DrugsANNSignaling/data/uniprot-reviewed_yes+AND+organism__Homo+sapiens+(Human)+[9606]_.tab')
annot <- annot %>% select(c('node'='Entry'),c('condition'='Gene.names...primary..')) %>% unique()
annot <- rbind(annot,
               data.frame(node=c('control','untreated','DMSO'),condition=c('control','untreated','DMSO'))) %>% 
  filter(node %in% results_ko$condition)
annot <- annot %>% group_by(condition) %>% mutate(counts = n_distinct(node)) %>% ungroup() %>%
  mutate(condition = ifelse(counts>1,node,condition)) %>% select(-counts) %>% unique()
annot <- annot %>% mutate(condition = ifelse(condition=='',node,condition))
# expreriment <- as.matrix(results_act %>% filter(model_no==i) %>% filter(condition!='untreated') %>%
#                            select(-type,-model_no) %>% column_to_rownames('condition'))
# control <- as.matrix(results_act %>% filter(model_no==i) %>% filter(condition=='untreated') %>%
#                        select(-type,-model_no) %>% column_to_rownames('condition'))
# expreriment <- apply(expreriment,1,function(x) x-control)
# expreriment <- t(expreriment)
XKO <- foreach(i = seq(0,49)) %dopar% {
  expreriment <- as.matrix(results_ko %>% filter(model_no==i) %>%
                             select(-type,-model_no) %>% column_to_rownames('condition'))
  rows <- rownames(expreriment)
  
  expreriment[expreriment>1] <-1
  expreriment[expreriment<0] <-0
  expreriment <- log((expreriment+1e-6)/(1 - expreriment + 1e-6))
  colnames(expreriment) <- colnames(results_ko)[1:104]
  rownames(expreriment) <- rows
  expreriment <- as.data.frame(cbind(expreriment,model_no=i)) %>% rownames_to_column('node')
  left_join(expreriment,annot) %>% select(condition,all_of(colnames(results_ko)[1:104]),model_no)
}
XKO <- do.call(rbind,XKO)
Xact <- foreach(i = seq(0,49)) %dopar% {
  expreriment <- as.matrix(results_act %>% filter(model_no==i) %>%
                             select(-type,-model_no) %>% column_to_rownames('condition'))
  rows <- rownames(expreriment)
  expreriment[expreriment>1] <- 1
  expreriment[expreriment<0] <- 0
  expreriment <- log((expreriment+1e-6)/(1 - expreriment + 1e-6))
  colnames(expreriment) <- colnames(results_act)[1:104]
  rownames(expreriment) <- rows
  expreriment <- as.data.frame(cbind(expreriment,model_no=i)) %>% rownames_to_column('node')
  left_join(expreriment,annot) %>% select(condition,all_of(colnames(results_act)[1:104]),model_no)
}
Xact <- do.call(rbind,Xact)

### Score LEMBAS results in terms of MAS and Fibrosis----------------
Yko <- as.data.frame(predict(plsr_model,XKO[,2:105]))
Yko$condition <- XKO$condition
Yko$model_no <- XKO$model_no
Yko <- Yko %>% gather('phenotype','score',-model_no,-condition)
Yko <- Yko %>% mutate(phenotype=ifelse(phenotype=='NAS','MAS','Fibrosis stage'))
# repeat for activation
Yact <- as.data.frame(predict(plsr_model,Xact[,2:105]))
Yact$condition <- Xact$condition
Yact$model_no <- Xact$model_no
Yact <- Yact %>% gather('phenotype','score',-model_no,-condition)
Yact <- Yact %>% mutate(phenotype=ifelse(phenotype=='NAS','MAS','Fibrosis stage'))

phenotypeKO_plot <- ggboxplot(Yko %>% filter(condition %in% c('untreated',
                                                              'TGFBR1','TGFBR2',
                                          'IFNAR1','IFNAR2')) %>%
            mutate(condition = factor(condition,levels = c('untreated',
                                                           'TGFBR1','TGFBR2',
                                                           'IFNAR1','IFNAR2'))),
          x='condition',y='score',color = 'condition',add='jitter')+
  stat_compare_means(ref.group = 'untreated',
                     label = 'p.signif',
                     method='wilcox.test',
                     paired = TRUE)+
  facet_wrap(~phenotype,scales = 'free_y')
phenotypeKO_plot <- add_theme(phenotypeKO_plot) + theme(legend.position = 'none',
                                                        element_text(angle=0))
print(phenotypeKO_plot)
ggsave(filename = "../figures/2 hour/phenotypeKO_plot.png", plot = phenotypeKO_plot, units = "cm", width = 16, height = 12)

phenotypeACT_plot <- ggboxplot(Yact %>% filter(condition %in% c('untreated',
                                                                'TGFBR1','TGFBR2',
                                                              'IFNAR1','IFNAR2')) %>%
                                mutate(condition = factor(condition,levels = c('untreated',
                                                                               'TGFBR1','TGFBR2',
                                                                               'IFNAR1','IFNAR2'))),
                              x='condition',y='score',color = 'condition',add='jitter')+
  stat_compare_means(ref.group = 'untreated',
                     label = 'p.signif',
                     method='wilcox.test',
                     paired = TRUE)+
  facet_wrap(~phenotype,scales = 'free_y')
phenotypeACT_plot <- add_theme(phenotypeACT_plot) + theme(legend.position = 'none',
                                                          element_text(angle=0))
print(phenotypeACT_plot)
ggsave(filename = "../figures/2 hour/phenotypeACT_plot.png", plot = phenotypeACT_plot, units = "cm", width = 16, height = 12)

Yall <- rbind(Yact %>% mutate(stimuli='activation'),
              Yko %>% mutate(stimuli='inhibition'))
Yall <- Yall %>% mutate(stimuli = ifelse(condition=='untreated','untreated',stimuli))
Yall <- distinct(Yall)
Yall<-  Yall %>% mutate(stimuli =factor(stimuli,levels = c('inhibition','activation','untreated')))
# Calculate the mean of 'untreated' for each phenotype
control_mu_mas <- mean(Yall$score[Yall$condition=='untreated' & Yall$phenotype=='MAS'])
control_mu_fibrosis <- mean(Yall$score[Yall$condition=='untreated' & Yall$phenotype!='MAS'])
untreated_means <- Yall %>%
  filter(condition == 'untreated') %>%
  group_by(phenotype) %>%
  summarise(mean_untreated = mean(score)) %>% mutate(stimuli='untreated')
phenotype_plot <- ggboxplot(Yall %>% filter(condition %in% c('untreated','TGFBR1','TGFBR2',
                                                                'IFNAR1','IFNAR2')) %>%
                                 mutate(condition = factor(condition,levels = c('untreated','TGFBR1','TGFBR2',
                                                                                'IFNAR1','IFNAR2'))),
                               x='condition',y='score',color = 'stimuli',add='jitter')+
  stat_compare_means(aes(group = stimuli),
                     method='wilcox.test',
                     paired=TRUE,
                     label = 'p.signif')+
  theme(panel.grid = element_line(),
          text = element_text(family = 'Arial',size=16),
        axis.title.x = element_blank(),
        axis.text.x = element_text(angle = 45))+
  facet_wrap(~phenotype,scales = 'free_y')+
  geom_hline(data = untreated_means, 
             aes(yintercept = mean_untreated,color=stimuli), 
             linetype = "dashed",
             lwd = 1)
print(phenotype_plot)
ggsave(filename = "../figures/2 hour/phenotype_plot.png", plot = phenotype_plot, 
       units = "cm", width = 16, height = 12,
       dpi = 600)

### Differential
Yall_control <- Yall %>% filter(condition=='untreated') %>%
  mutate(score = ifelse(phenotype=='MAS',score - control_mu_mas,score -control_mu_fibrosis))
### activation - inhibition
Yall_diff <- left_join(Yact %>% filter(condition!='untreated'),
                       Yko  %>% filter(condition!='untreated'),
                       by=c('condition','phenotype','model_no'))
Yall_diff <- Yall_diff %>% mutate(score = score.x-score.y) %>% select(-score.x,-score.y) %>% 
  mutate(stimuli = 'stimulated')
Yall_diff <- rbind(Yall_diff,Yall_control)
stats_res <- Yall_diff %>% group_by(condition,phenotype) %>% 
  summarise(p_value = wilcox.test(score, mu = 0)$p.value) %>%
  mutate(p_adjusted = p.adjust(p_value, method = "BH")) %>%
  ungroup()
Yall_diff <- left_join(Yall_diff,stats_res)
pvals_df <- Yall_diff %>%
  filter(condition %in% c('untreated', 'TGFBR1', 'TGFBR2', 'IFNAR1', 'IFNAR2')) %>%
  mutate(group1 = condition, group2 = condition, 
         y.position = max(score) + 0.05)%>% filter(group1 %in% c('untreated','TGFBR1','TGFBR2',
                                                                 'IFNAR1','IFNAR2')) %>%
  filter(group2 %in% c('untreated','TGFBR1','TGFBR2',
                       'IFNAR1','IFNAR2')) 
pvals_df <- pvals_df %>% select(condition,phenotype,p_value,p_adjusted,group1,group2,y.position) %>%
  unique()
pvals_df <- pvals_df %>% mutate(plab = ifelse(p_adjusted<=0.0001,'****',
                                              ifelse(p_adjusted<=0.001,'***',
                                                     ifelse(p_adjusted<=0.01,'**',
                                                            ifelse(p_adjusted<=0.05,'*',
                                                                   '')))))
phenotype_plot <- ggboxplot(Yall_diff %>% filter(condition %in% c('untreated','TGFBR1','TGFBR2',
                                                             'IFNAR1','IFNAR2')) %>%
                              mutate(condition = factor(condition,levels = c('untreated','TGFBR1','TGFBR2',
                                                                             'IFNAR1','IFNAR2'))),
                            x='condition',y='score',color = 'stimuli',add='jitter')+
  ylab(expression(Delta*'score'))+
  theme(panel.grid = element_line(),
        text = element_text(family = 'Arial',size=16),
        axis.title.x = element_blank(),
        axis.text.x = element_text(angle = 45))+
  facet_wrap(~phenotype,scales = 'free_y')+
  geom_hline(yintercept = 0, 
             linetype = "dashed",
             lwd = 1)+
  stat_pvalue_manual(pvals_df,
                     x = "group1",
                     label = "plab",
                     y.position = "y.position",
                     tip.length = 0,
                     size = 4)
print(phenotype_plot)
ggsave(filename = "../figures/2 hour/differetial_phenotype_plot.png", plot = phenotype_plot, 
       units = "cm", width = 16, height = 12,
       dpi = 600)

### Show what happens to fibrosis and MAS when hitting known:
### IFNa reponse members,
### IFNg reponse members,
### TGFb reponse members,
### JAK-STAT reponse members,
### IFNa reponse members
library(EGSEAdata)
library(AnnotationDbi)
library(org.Hs.eg.db)
library(topGO)
library(GO.db)
egsea.data(species = "human",returnInfo = TRUE)
msig_overview <- t(as.data.frame(msigdb[[1]]))
for (i in 2:(length(msigdb)-1)) {
  msig_overview <- rbind(msig_overview,t(as.data.frame(msigdb[[i]])))
}
msig_overview <- as.data.frame(msig_overview)
rownames(msig_overview) <- msig_overview$STANDARD_NAME
hallmarks <- msig_overview %>% filter(CATEGORY_CODE=='H')
hallmarks <- hallmarks %>%
  filter(grepl('INTERFERON',STANDARD_NAME) | grepl('TGF_BETA',STANDARD_NAME)| grepl('JAK_STAT',STANDARD_NAME) | grepl('EPITHELIAL_MESENCHYMAL_TRANSITION',STANDARD_NAME))
df_msig <- as.data.frame(do.call(rbind,msigdb))
inds <- which(df_msig$STANDARD_NAME %in% hallmarks$STANDARD_NAME)
genesets <- c()
h <- 1
for (ind in inds){
  set <- msigdb[[ind]]
  nm <- set["STANDARD_NAME"]
  genesets <- c(genesets,str_split(set["MEMBERS_EZID"],pattern = ","))
  names(genesets)[h] <- nm
  h <- h+1
}
df_hallmakrs <- as.data.frame(do.call(rbind,genesets)) %>%
  rownames_to_column('geneset') %>% gather('value','entrez_id',-geneset)
df_hallmakrs <- df_hallmakrs %>% dplyr::select(-value) %>% unique() %>% 
    group_by(geneset) %>% mutate(num_genes = n_distinct(entrez_id)) %>% ungroup()
entrez_ids <- mapIds(org.Hs.eg.db, keys = unique(df_hallmakrs$entrez_id), column = "SYMBOL", keytype = "ENTREZID")
inds <- which(!is.na(entrez_ids))
entrez_ids <- entrez_ids[inds]
entrez_ids <- data.frame(entrez_ids)
entrez_ids <- entrez_ids %>% rownames_to_column('entrez_id')
df_hallmakrs <- left_join(df_hallmakrs,entrez_ids)
df_hallmakrs <- df_hallmakrs%>% mutate(geneset=substr(geneset, nchar('HALLMARK_')+1, nchar(geneset)))
df_hallmakrs <- df_hallmakrs %>% mutate(geneset=str_replace_all(geneset,'_',' '))
df_hallmakrs <- df_hallmakrs %>% dplyr::select(geneset,c('gene'='entrez_ids'))%>%
  filter(!is.na(gene)) %>% unique()
df_hallmakrs <- df_hallmakrs %>% filter(gene %in% Yko$condition)
### add manually IFNs and IRFs in INF response pathway
df_hallmakrs <- rbind(df_hallmakrs %>% 
  mutate(geneset = ifelse(geneset=='INTERFERON GAMMA RESPONSE' & grepl('IFNA',gene),
                          'INTERFERON ALPHA RESPONSE',geneset)) %>%
  mutate(geneset = ifelse(geneset=='IL6 JAK STAT3 SIGNALING' & grepl('IFNA',gene),
                            'INTERFERON ALPHA RESPONSE',geneset)) %>%
  mutate(geneset = ifelse(geneset=='IL6 JAK STAT3 SIGNALING' & grepl('IFNG',gene),
                            'INTERFERON GAMMA RESPONSE',geneset)) %>%
  mutate(geneset = ifelse(geneset=='INTERFERON ALPHA RESPONSE' & grepl('IFNG',gene),
                          'INTERFERON GAMMA RESPONSE',geneset)),
  data.frame(gene = c('IFNGR2','IFNGR1','IFNAR1','IFNAR2','TGFBR2',
                      'JAK1','JAK2','STAT1','STAT2','IRF1','IRF9',
                      'JAK1','TYK2','STAT1','STAT2','IRF9',
                      'JAK1','JAK2'),
             geneset = c('INTERFERON GAMMA RESPONSE','INTERFERON GAMMA RESPONSE',
                      'INTERFERON ALPHA RESPONSE','INTERFERON ALPHA RESPONSE',
                      'TGF BETA SIGNALING',
                      'INTERFERON GAMMA RESPONSE','INTERFERON GAMMA RESPONSE','INTERFERON GAMMA RESPONSE',
                      'INTERFERON GAMMA RESPONSE','INTERFERON GAMMA RESPONSE','INTERFERON GAMMA RESPONSE',
                      'INTERFERON ALPHA RESPONSE','INTERFERON ALPHA RESPONSE','INTERFERON ALPHA RESPONSE',
                      'INTERFERON ALPHA RESPONSE','INTERFERON ALPHA RESPONSE',
                      'IL6 JAK STAT3 SIGNALING','IL6 JAK STAT3 SIGNALING'))) %>%
  unique()
Yko_filt <- Yko %>% filter(condition %in% df_hallmakrs$gene)
# Yko_filt <- Yko_filt %>% group_by(condition,phenotype) %>% mutate(score = mean(score)) %>%
#   dplyr::select(condition,phenotype,score) %>% unique() %>%
#   spread('phenotype','score')
# Yko_filt <- Yko_filt %>% mutate(MAS = MAS/8,`Fibrosis stage`=`Fibrosis stage`/4)
# dend <- hclust(dist(Yko_filt))
# ordered_matrix <- Yko_filt[dend$order,]
Yko_filt <- left_join(Yko_filt,df_hallmakrs,by=c('condition'='gene'))

# ggboxplot(Yko_filt %>% filter(phenotype=='MAS'),x='condition',y='score',color = 'geneset') +
#   facet_wrap(~geneset,scale = 'free') +
#   theme(legend.position = 'none')

scatterplot <- ggplot(Yko_filt %>% spread('phenotype','score') %>% 
         mutate(MAS = MAS-control_mu_mas, `Fibrosis stage` = `Fibrosis stage` - control_mu_fibrosis ) %>%
         group_by(condition) %>% mutate(MAS = mean(MAS),`Fibrosis stage` = mean(`Fibrosis stage`)) %>%
         ungroup() %>% dplyr::select(-model_no) %>% unique() %>%
         # mutate(geneset = ifelse(!(abs(MAS)>=1 |abs(`Fibrosis stage`)>=1),NA,geneset)) %>%
         mutate(label = ifelse(MAS>=1 & `Fibrosis stage`>=0.5,condition,NA)) %>%
         mutate(label = ifelse(MAS<=-1 & `Fibrosis stage`<=-0.5,condition,label)),
       aes(x=MAS,y=`Fibrosis stage`,color = geneset)) +
  geom_point()+
  geom_text_repel(aes(label=label,color=geneset),max.overlaps = 80,size=5)+
  ylab(expression('average '*Delta*'Fibrosis stage')) + xlab(expression('average '*Delta*'MAS'))
scatterplot <- add_theme(scatterplot) + theme(legend.position = 'right',
                                              text = element_text(size=22))
print(scatterplot)
ggsave(filename = "../figures/2 hour/phenotype_change_scatterplot.png", plot = scatterplot, units = "cm", width = 28, height = 16)

### Show change in:
### IRF1, IRF3, IRF7, IRF9, STAT1, STAT2, JAK1, JAK2 for IFNg after KO of IFNGR1/2
### IRF9, IRF3, IRF7, STAT1, STAT2, TYK2, JAK1 for IFNa after KO of IFNAR1/2
### STAT2, SMAD1, SMAD2, SMAD4 , TGFB1
XKO_diff <- foreach(i = seq(0,49)) %dopar% {
  expreriment <- as.matrix(XKO %>% filter(model_no==i) %>% filter(condition!='untreated') %>%
                             dplyr::select(-model_no) %>% column_to_rownames('condition'))
  rows <- rownames(expreriment)
  control <- as.matrix(XKO %>% filter(model_no==i) %>% filter(condition=='untreated') %>%
                         dplyr::select(-model_no) %>% column_to_rownames('condition'))
  expreriment <- apply(expreriment,1,function(x) x-control)
  expreriment <- t(expreriment)
  colnames(expreriment) <- colnames(results_ko)[1:104]
  rownames(expreriment) <- rows
  as.data.frame(cbind(expreriment,model_no=i)) %>% rownames_to_column('node')
}
XKO_diff <- do.call(rbind,XKO_diff)
XKO_diff_filt <- XKO_diff %>% dplyr::select(node,model_no,
                                            all_of(c('IRF1','IRF9', 'STAT1', 'STAT2',
                                                     'SMAD3', 'SMAD4',
                                                     "NFKB1","NFKB2")))
XKO_diff_filt <- XKO_diff_filt %>% filter(node %in% c('JAK1','JAK2',
                                                      'TGFBR1','TGFBR2',
                                                      'IFNAR1','IFNAR2',
                                                      'IFNGR1','IFNGR2'))


ifng_plot <- ggboxplot(XKO_diff_filt %>% filter(node %in% c('IRF1', 'IRF3', 'IRF7','IRF9', 
                                                            'STAT1', 'STAT2', #'JAK1', 'JAK2',
                                               'IFNGR1','IFNGR2')) %>%
            dplyr::select(node,model_no,all_of(c('IRF1', 'IRF9', 'STAT2'))) %>% #, 'STAT1'
            gather('TF','activity',-node,-model_no),
          x = 'node',y='activity',add='jitter')+
  ggtitle('IFNg response') + xlab('KO of signaling node')+
  geom_hline(yintercept = 0,linetype = 'dashed',lwd=0.8,color='black')+
  # scale_y_break(c(-4, -8))+
  scale_y_continuous(limits = c(-3,NA))+
  facet_wrap(~TF)
ifng_plot <- add_theme(ifng_plot) + theme(panel.border = element_blank(),
                                          axis.line.y = element_line(color = "black"),
                                          plot.title = element_text(hjust = 0.5),
                                          text = element_text(size=16),
                                          axis.text.x = element_text(angle=45,vjust = 0.5),
                                          axis.title.x = element_blank())
print(ifng_plot)
ggsave(filename = "../figures/2 hour/ifng_effect_plot.pdf", plot = ifng_plot, 
       device = cairo_pdf,
       units = "cm", width = 12, height = 12)


ifna_plot <- ggboxplot(XKO_diff_filt %>% filter(node %in% c('IRF9', 'STAT1', 'STAT2', #'TYK2', 'JAK1',
                                                            'IFNAR1','IFNAR2')) %>%
                         dplyr::select(node,model_no,all_of(c('IRF1', 'IRF9', 'STAT2'))) %>% #, 'STAT1'
                         gather('TF','activity',-node,-model_no),
                       x = 'node',y='activity',add='jitter')+
  ggtitle('IFNa response') + xlab('KO of signaling node')+
  geom_hline(yintercept = 0,linetype = 'dashed',lwd=0.8,color='black')+
  scale_y_break(c(-2.5, -14))+
  facet_wrap(~TF)
ifna_plot <- add_theme(ifna_plot) + theme(panel.border = element_blank(),
                                          axis.line.y = element_line(color = "black"),
                                          plot.title = element_text(hjust = 0.5),
                                          text = element_text(size=16),
                                          axis.text.x = element_text(angle=45,vjust = 0.5),
                                          axis.title.x = element_blank())
print(ifna_plot)
ggsave(filename = "../figures/2 hour/ifna_effect_plot.pdf", plot = ifna_plot, 
       device = cairo_pdf,
       units = "cm", width = 12, height = 12)

tgfb_plot <- ggboxplot(XKO_diff_filt %>% filter(node %in% c('TGFBR1','TGFBR2')) %>%
                         dplyr::select(node,model_no,all_of(c('SMAD3', 'SMAD4'))) %>%
                         gather('TF','activity',-node,-model_no),
                       x = 'node',y='activity',add='jitter')+
  ggtitle('TGFb response') + xlab('KO of signaling node')+
  geom_hline(yintercept = 0,linetype = 'dashed',lwd=0.8,color='black')+
  scale_y_break(c(-3.5, -11))+
  facet_wrap(~TF,ncol = 5)
tgfb_plot <- add_theme(tgfb_plot) + theme(panel.border = element_blank(),
                                          axis.line.y = element_line(color = "black"),
                                          plot.title = element_text(hjust = 0.5),
                                          text = element_text(size=16),
                                          axis.text.x = element_text(angle=45,vjust = 0.5),
                                          axis.title.x = element_blank())
print(tgfb_plot)
ggsave(filename = "../figures/2 hour/tgfb_effect_plot.pdf", plot = tgfb_plot, 
       device = cairo_pdf,
       units = "cm", width = 20, height = 10)
