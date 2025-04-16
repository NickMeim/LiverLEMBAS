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

### Load data------------------------
# Load TFs results
results_tfs <- data.table::fread('../results/allmodels/signal_exploration/TFs_all_conditions.csv') %>% 
  column_to_rownames('V1')
# Load output nodes
results_outs <- data.table::fread('../results/allmodels/signal_exploration/Nodes_output_conditions.csv') %>% 
  column_to_rownames('V1')
# Load output nodes
results_ins <- data.table::fread('../results/allmodels/signal_exploration/Nodes_input_conditions.csv') %>% 
  column_to_rownames('V1')

#Load annotation file to replace with gene names
annot <- read.delim('../../../Artificial-Signaling-Network/DrugsANNSignaling/data/uniprot-reviewed_yes+AND+organism__Homo+sapiens+(Human)+[9606]_.tab')
annot <- annot %>% select(c('uniprot'='Entry'),c('name'='Gene.names...primary..')) %>% unique()
annot <- annot %>% mutate(name = ifelse(name=='' | name==' ' | is.na(name),uniprot,name))

### Load human data and score samples with PLSR
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
human_annot <- data.frame(name=colnames(TF_human))
human_annot <- left_join(human_annot,annot)
## keep only TFs in LEMBAS
print(all(human_annot$name==colnames(TF_human)))
colnames(TF_human)  <- human_annot$uniprot
TF_human <- TF_human[,colnames(results_tfs)[1:104]]
# TF_human <- 1/(1+exp(-TF_human))
## train one model with all the data
plsr_model <- opls(x = TF_human, 
                   y = Yh,
                   predI = 8,
                   crossvalI = 1,
                   scaleC = "center",
                   fig.pdfC = "none",
                   info.txtC = "none")
x <- results_tfs[,1:104]
x[x>1] <- 1
x[x<0] <- 0
x <- log((x+1e-6)/(1 - x + 1e-6))
hist(as.matrix(x))
Ysimulation <- predict(plsr_model,x)
Ysimulation <- as.data.frame(Ysimulation)
Ysimulation <- cbind(results_tfs %>% select(model_no,node,condition),Ysimulation)
Ysimulation <- left_join(Ysimulation,annot,by=c('node'='uniprot'))

## re-annotate data
results_tfs <- results_tfs %>% gather('TF','activity',-model_no,-node,-condition)
results_tfs <- left_join(results_tfs,annot,by=c('node'='uniprot')) %>% select(-node) %>% 
  select(model_no,c('node'='name'),condition,TF,activity)
results_tfs <- left_join(results_tfs,annot,by=c('TF'='uniprot')) %>% select(-TF) %>% 
  select(model_no,node,condition,c('TF'='name'),activity)

results_outs <- left_join(results_outs,annot,by=c('node'='uniprot')) %>% select(-node) %>% 
  select(model_no,c('node'='name'),condition,outlet,activity)
results_outs <- left_join(results_outs,annot,by=c('outlet'='uniprot')) %>% select(-outlet) %>% 
  select(model_no,node,condition,c('outlet'='name'),activity)

results_ins <- left_join(results_ins,annot,by=c('node'='uniprot')) %>% select(-node) %>% 
  select(model_no,c('node'='name'),condition,inlet,activity)
results_ins <- left_join(results_ins,annot,by=c('inlet'='uniprot')) %>% select(-inlet) %>% 
  select(model_no,node,condition,c('inlet'='name'),activity)

#### Visualize convergence of activity of the node itself-----------------
ggplot(results_outs %>% filter(outlet %in% unique(results_tfs$node)) %>%
         filter(outlet==node) %>% 
         group_by(node,condition) %>% mutate(mu=mean(activity)) %>% mutate(se=sd(activity)/sqrt(50)) %>% ungroup() %>%
         select(node,condition,mu,se) %>% unique(),
       aes(x=condition,y=mu,color=node))+
  geom_point()+
  geom_line()+
  geom_errorbar(aes(ymin=mu-se,ymax=mu+se),width=10)+
  ylab('activity') + xlab('input signal') + labs(color='input node')

#### Visualize convergence of activity of the output nodes-----------------
pout <- ggplot(results_outs %>% filter(!(outlet %in% unique(results_tfs$node))) %>% #filter(node=='IFNAR1')%>%
         group_by(node,outlet,condition) %>% 
         mutate(mu=mean(activity)) %>% mutate(se=sd(activity)/sqrt(50)) %>% ungroup() %>%
         select(node,outlet,condition,activity,mu,se) %>% unique(),
       aes(x=condition,y=activity,color=outlet))+
  geom_smooth(se = TRUE,alpha=0.2)+
  # geom_point()+
  # geom_line()+
  # geom_errorbar(aes(ymin=mu-se,ymax=mu+se),width=10)+
  ylab('activity') + xlab('input signal') + labs(color='outlet')+
  facet_wrap(~node)
pout <- add_theme(pout)
print(pout)
ggsave(filename = "../figures/outlets_convergence.pdf", 
       plot = pout, 
       device = cairo_pdf,
       units = "cm", width = 16, height = 12)
ggsave(filename = "../figures/outlets_convergence.png", 
       plot = pout, 
       units = "cm", width = 16, height = 12)

#### Visualize convergence of activity of input nodes------------------------
pin <- ggplot(results_ins %>% filter(!(inlet %in% unique(results_tfs$node))) %>%
                group_by(node,inlet,condition) %>% 
                mutate(mu=mean(activity)) %>% mutate(se=sd(activity)/sqrt(50)) %>% ungroup() %>%
                select(node,inlet,condition,activity,mu,se) %>% unique(),
              aes(x=condition,y=activity,color=inlet))+
  geom_smooth(se = TRUE,alpha=0.2)+
  # geom_point()+
  # geom_line()+
  # geom_errorbar(aes(ymin=mu-se,ymax=mu+se),width=10)+
  ylab('activity') + xlab('input signal') + labs(color='inlet')+
  facet_wrap(~node)
pin <- add_theme(pin)
print(pin)
ggsave(filename = "../figures/inlets_convergence.pdf", 
       plot = pin, 
       device = cairo_pdf,
       units = "cm", width = 16, height = 12)
ggsave(filename = "../figures/inlets_convergence.png", 
       plot = pin, 
       units = "cm", width = 16, height = 12)

### Visualize the convergence of TFs----------------
ptfs <- ggplot(results_tfs %>% 
                group_by(node,TF,condition) %>% 
                mutate(mu=mean(activity)) %>% mutate(se=sd(activity)/sqrt(50)) %>% ungroup() %>%
                select(node,TF,condition,activity,mu,se) %>% unique(), #%>% group_by(TF) %>% mutate(min_a=min(mu)) %>% ungroup() %>% filter(min_a<0 | TF %in% c('STAT1','STAT2','SMAD3')),
              aes(x=condition,y=activity,color=TF))+
  geom_smooth(se = TRUE,alpha=0.2)+
  # geom_point()+
  # geom_line()+
  # geom_errorbar(aes(ymin=mu-se,ymax=mu+se),width=10)+
  ylab('activity') + xlab('input signal') + labs(color='TF')+
  facet_wrap(~node)
ptfs <- add_theme(ptfs)
ptfs <- ptfs + theme(legend.position = 'none')
print(ptfs)
ggsave(filename = "../figures/TFs_convergence.pdf", 
       plot = ptfs, 
       device = cairo_pdf,
       units = "cm", width = 16, height = 12)
ggsave(filename = "../figures/TFs_convergence.png", 
       plot = ptfs, 
       units = "cm", width = 16, height = 12)

### Visualize convergence of scored phenotype------------
Yc <- Ysimulation %>% gather('phenotype','score',-model_no,-condition,-node,-name) %>%
  filter(condition==0) %>% filter(name=='IFNAR1') %>%
  select(phenotype,score,model_no) %>% unique()
for (i in 1:(length(unique(Ysimulation$condition))-1)){
  tmp <- Ysimulation %>% gather('phenotype','score',-model_no,-condition,-node,-name) %>%
    filter(condition==0) %>% filter(name=='IFNAR1') %>%
    select(phenotype,score,model_no) %>% unique() 
  Yc <- rbind(Yc,tmp)
}
Yc <- Yc %>%
  mutate(condition = rep(unique(Ysimulation$condition),100))
Yc <- Yc %>% select(model_no,phenotype,score,condition) %>% 
  group_by(phenotype) %>%
  mutate(mu = mean(score)) %>% mutate(se = sd(score)/sqrt(50)) %>%
  ungroup() %>%
  select(phenotype,mu,se,condition) %>% 
  unique()
pY <- ggplot(Ysimulation %>% gather('phenotype','score',-model_no,-condition,-node,-name),
             aes(x=condition,y=score,color=name))+
  geom_smooth(alpha=0.2)+
  geom_ribbon(data=Yc %>% mutate(`shaded area` = 'untreated'),
              aes(x = condition,
                  ymin = mu - se, 
                  ymax = mu + se,
                  fill = `shaded area`), 
              alpha = 0.2,
              linetype = "dashed",inherit.aes = FALSE,show.legend = TRUE) +
  facet_wrap(~phenotype,scales = 'free_y')
pY <- add_theme(pY)
print(pY)
ggsave(filename = "../figures/Ysimulated_convergence.png", 
       plot = pY, 
       units = "cm", width = 16, height = 12)


### See if any drug can be considered a IFNa,g or TGFb equivalent---------------
### Get drugs and untreated signaling nodes and keep receptors
signaling_nodes <- data.table::fread('../results/allmodels/predictions/SignalingNodes_all_conditions.csv')%>%
  select(-V1) %>% 
  filter(node %in% c('P36897','P37173',
                     'P17181','P48551',
                     'P15260','P38484'))
## for 2 hour erythromycin ethylsuccinate works but not for 8 hour
# selected_drugs <- signaling_nodes %>% group_by(condition,node) %>% mutate(mu_act = mean(activity)) %>% ungroup() %>%
#   group_by(node) %>% mutate(max_act=max(mu_act)) %>%
#   ungroup() %>% filter(mu_act==max_act) %>%
#   separate(condition,into = c('drug','dose','time'),sep = '\\+') %>%
#   select(drug,dose,time,node,max_act) %>% unique()
# selected_drugs <- left_join(selected_drugs,annot,by=c('node'='uniprot')) %>%
#   select(-node) %>% select(drug,dose,time,c('node'='name'),max_act)
# signaling_nodes <- signaling_nodes %>%
  # separate(condition,into = c('drug','dose','time'),sep = '\\+',remove = FALSE) %>%
  # filter(condition=='untreated' | drug %in% selected_drugs$drug)
signaling_nodes <- signaling_nodes %>%
  separate(condition,into = c('drug','dose','time'),sep = '\\+',remove = FALSE) %>%
  filter(condition=='untreated' | drug %in% c('phenylbutazone','erythromycin ethylsuccinate','benzbromarone'))
signaling_nodes <- left_join(signaling_nodes,annot,by=c('node'='uniprot')) %>%
  select(-node) %>% select(condition,model_no,c('node'='name'),activity)
signaling_nodes <- signaling_nodes %>%
  separate(condition,into = c('drug','dose','time'),sep = '\\+',remove = FALSE)
## Load also untreated signaling nodes
untreated_signaling <- data.table::fread('../results/allmodels/InSilicoKOs/Nodes_subset_conditions.csv') %>%
  select(-V1,-type) %>% filter(condition=='untreated') %>% gather('node','activity',-condition,-model_no) %>%
  filter(node %in% c('P36897','P37173',
                     'P17181','P48551',
                     'P15260','P38484'))
untreated_signaling <- left_join(untreated_signaling,annot,by=c('node'='uniprot')) %>%
  select(-node) %>% select(condition,model_no,c('node'='name'),activity)
untreated_signaling <- untreated_signaling %>%
  mutate(dose='untreated') %>%
  mutate(time='2 hour') %>%
  mutate(drug='untreated') %>%
  select(all_of(colnames(signaling_nodes)))
all_signaling <- rbind(signaling_nodes,untreated_signaling)

### Visualize for IFNAR1
# drugs <- selected_drugs %>% filter(node=='IFNAR1')
# ifnar1_df <- all_signaling %>% 
#   filter(drug %in% c(drugs$drug,'untreated')) %>% filter(grepl('IFNA',node)) %>% #%>% filter(node=='IFNAR1') %>%
#   mutate(node = paste0(node,' : ',drugs$drug))
ifnar1_df <- all_signaling %>% 
  filter(drug %in% c('phenylbutazone','untreated')) %>% filter(grepl('IFNA',node)) %>% #%>% filter(node=='IFNAR1') %>%
  mutate(node = paste0(node,' : ','phenylbutazone'))
ifnar1_df$dose <- factor(ifnar1_df$dose,
                         levels = c('untreated','Control','Low','Middle','High'))
ggboxplot(ifnar1_df,
          x='dose',y='activity',add='jitter') +
  facet_wrap(~node,ncol = 1) +
  scale_y_continuous(limits = c(NA,1)) +
  geom_hline(yintercept = 0.5,linetype='dashed',color='black',lwd=1)
ggsave('../figures/ifnaR_drug_activator.png',
       width = 16,
       height = 12,
       units = 'cm',
       dpi=600)

### Visualize for IFNGR1
# drugs <- selected_drugs %>% filter(node=='IFNGR1')
# ifngr1_df <- all_signaling %>% 
#   filter(drug %in% c(drugs$drug,'untreated')) %>% filter(grepl('IFNG',node)) %>% #%>% filter(node=='IFNGR1') %>%
#   mutate(node = paste0(node,' : ',drugs$drug))
ifngr1_df <- all_signaling %>% 
  filter(drug %in% c('benzbromarone','untreated')) %>% filter(grepl('IFNG',node)) %>% #%>% filter(node=='IFNGR1') %>%
  mutate(node = paste0(node,' : ','benzbromarone'))
ifngr1_df$dose <- factor(ifngr1_df$dose,
                         levels = c('untreated','Control','Low','Middle','High'))
ggboxplot(ifngr1_df,
          x='dose',y='activity',add='jitter') +
  facet_wrap(~node,ncol = 1) +
  scale_y_continuous(limits = c(NA,1)) +
  geom_hline(yintercept = 0.5,linetype='dashed',color='black',lwd=1)
# ggsave('../figures/ifngR_drug_activator.png',
#        width = 16,
#        height = 12,
#        units = 'cm',
#        dpi=600)

### Visualize for TGFBR1
# drugs <- selected_drugs %>% filter(node=='TGFBR1')
# tgfbr1_df <- all_signaling %>% 
#   filter(drug %in% c(drugs$drug,'untreated')) %>% filter(grepl('TGFBR',node)) %>% #filter(node=='TGFBR1') %>%
#   mutate(node = paste0(node,' : ',drugs$drug))
tgfbr1_df <- all_signaling %>% 
  filter(drug %in% c('erythromycin ethylsuccinate','untreated')) %>% filter(grepl('TGFBR',node)) %>% #filter(node=='TGFBR1') %>%
  mutate(node = paste0(node,' : ','erythromycin ethylsuccinate'))
tgfbr1_df$dose <- factor(tgfbr1_df$dose,
                         levels = c('untreated','Control','Low','Middle','High'))
ggboxplot(tgfbr1_df,
          x='dose',y='activity',add='jitter') +
  facet_wrap(~node,ncol = 1) +
  scale_y_continuous(limits = c(NA,1)) +
  geom_hline(yintercept = 0.5,linetype='dashed',color='black',lwd=1)
ggsave('../figures/tgfbR_drug_activator.png',
       width = 16,
       height = 12,
       units = 'cm',
       dpi=600)
# ### Visualize for IFNA
# drugs <- selected_drugs %>% filter(grepl('IFNA',receptor))
# ifna_df <- all_results %>% 
#   filter(condition %in% c(drugs$receptor,'untreated') | grepl(drugs$drug[1],condition) | grepl(drugs$drug[2],condition))
# ggboxplot(ifna_df %>% filter(TF %in% c('IRF1','IRF9','STAT1','STAT2')),
#           x='condition',y='activity',add='jitter') +
#   facet_wrap(~TF,ncol = 1)
# 
# ### Visualize for IFNG
# drugs <- selected_drugs %>% filter(grepl('IFNG',receptor))
# ifng_df <- all_results %>% 
#   filter(condition %in% c(drugs$receptor,'untreated') | grepl(drugs$drug[1],condition) | grepl(drugs$drug[2],condition))
# ggboxplot(ifng_df %>% filter(TF %in% c('IRF1','IRF9','STAT2')),
#           x='condition',y='activity',add='jitter') +
#   facet_wrap(~TF,ncol = 1)
# 
# ### Visualize for TGFB
# drugs <- selected_drugs %>% filter(grepl('TGFB',receptor))
# tgfb_df <- all_results %>% 
#   filter(condition %in% c(drugs$receptor,'untreated') | grepl(drugs$drug[1],condition) | grepl(drugs$drug[2],condition))
# ggboxplot(tgfb_df %>% filter(TF %in% c('SMAD3','SMAD4')),
#           x='condition',y='activity',add='jitter') +
#   facet_wrap(~TF,ncol = 1)


### Simulation + drugs---------------------------
results_drugs_outs <- data.table::fread('../results/allmodels/signal_exploration/Nodes_output_drugs_conditions.csv') %>% 
  column_to_rownames('V1')
results_drugs_outs <- results_drugs_outs %>% 
  separate('stimuli',into=c('node','drug','dose','time'),sep = '\\+')
results_drugs_outs <- left_join(results_drugs_outs,annot,by=c('node'='uniprot')) %>% select(-node) %>% 
  select(model_no,drug,dose,time,c('node'='name'),condition,outlet,activity)
results_drugs_outs <- left_join(results_drugs_outs,annot,by=c('outlet'='uniprot')) %>% select(-outlet) %>% 
  select(model_no,drug,dose,time,node,condition,c('outlet'='name'),activity)

# pout2 <- ggplot(results_drugs_outs %>% filter(!(outlet %in% unique(results_tfs$node))) %>% #filter(node=='IFNAR1')%>%
#                  group_by(node,outlet,condition) %>% 
#                  mutate(mu=mean(activity)) %>% mutate(se=sd(activity)/sqrt(50)) %>% ungroup() %>%
#                  select(drug,node,outlet,condition,activity,mu,se) %>% unique(),
#                aes(x=condition,y=activity,color=outlet))+
#   geom_smooth(se = TRUE,alpha=0.2)+
#   ylab('activity') + xlab('input signal') + labs(color='outlet')+
#   facet_wrap(vars(drug,node))
# pout2 <- add_theme(pout2)
# print(pout2)

# IFNAR1
p_ifnar1 <- ggboxplot(results_drugs_outs %>% filter(condition==(-3040) | condition==0)  %>%
                mutate(condition=ifelse(condition==0,drug,paste0('KO ',node,'+',drug))) %>%
                  filter(node=='IFNAR1')%>%
                 select(condition,node,outlet,activity,model_no) %>% unique(),
               x='condition',y='activity',color = 'outlet',add='jitter')+
  ylab('activity') + labs(color='outlet')+
  facet_wrap(~outlet,scales = 'free_y')
p_ifnar1 <- add_theme(p_ifnar1) + theme(legend.position = 'none',axis.text.x = element_text(angle = 20))
print(p_ifnar1)
ggsave(filename = "../figures/drug_ifnar1_ko.png",
       plot = p_ifnar1,
       units = "cm", width = 16, height = 12)

# IFNAR2
p_ifnar2 <- ggboxplot(results_drugs_outs %>% filter(condition==(-3040) | condition==0)  %>%
                        mutate(condition=ifelse(condition==0,drug,paste0('KO ',node,'+',drug))) %>%
                        filter(node=='IFNAR2')%>%
                        select(condition,node,outlet,activity,model_no) %>% unique(),
                      x='condition',y='activity',color = 'outlet',add='jitter')+
  ylab('activity') + labs(color='outlet')+
  facet_wrap(~outlet,scales = 'free_y')
p_ifnar2 <- add_theme(p_ifnar2) + theme(legend.position = 'none',axis.text.x = element_text(angle = 20))
print(p_ifnar2)
ggsave(filename = "../figures/drug_ifnar2_ko.png",
       plot = p_ifnar2,
       units = "cm", width = 16, height = 12)


# IFNGR1
p_ifngr1 <- ggboxplot(results_drugs_outs %>% filter(condition==(-3040) | condition==0)  %>%
                        mutate(condition=ifelse(condition==0,drug,paste0('KO ',node,'+',drug))) %>%
                        filter(node=='IFNGR1')%>%
                        select(condition,node,outlet,activity,model_no) %>% unique(),
                      x='condition',y='activity',color = 'outlet',add='jitter')+
  ylab('activity') + labs(color='outlet')+
  facet_wrap(~outlet,scales = 'free_y')
p_ifngr1 <- add_theme(p_ifngr1) + theme(legend.position = 'none')
print(p_ifngr1)
# ggsave(filename = "../figures/drug_ifngr1_ko.png", 
#        plot = p_ifngr1, 
#        units = "cm", width = 16, height = 12)

# IFNGR2
p_ifngr2 <- ggboxplot(results_drugs_outs %>% filter(condition==(-3040) | condition==0)  %>%
                        mutate(condition=ifelse(condition==0,drug,paste0('KO ',node,'+',drug))) %>%
                        filter(node=='IFNGR2')%>%
                        select(condition,node,outlet,activity,model_no) %>% unique(),
                      x='condition',y='activity',color = 'outlet',add='jitter')+
  ylab('activity') + labs(color='outlet')+
  facet_wrap(~outlet,scales = 'free_y')
p_ifngr2 <- add_theme(p_ifngr2) + theme(legend.position = 'none')
print(p_ifngr2)
# ggsave(filename = "../figures/drug_ifngr2_ko.png", 
#        plot = p_ifngr2, 
#        units = "cm", width = 16, height = 12)

# TGFBR1
p_tgfbr1 <- ggboxplot(results_drugs_outs %>% filter(condition==(-3040) | condition==0)  %>%
                        mutate(condition=ifelse(condition==0,'drug','KO+drug')) %>%
                        filter(node=='TGFBR1')%>%
                        select(condition,node,outlet,activity,model_no) %>% unique(),
                      x='condition',y='activity',color = 'outlet',add='jitter')+
  ylab('activity') + labs(color='outlet')+
  facet_wrap(~outlet,scales = 'free_y')
p_tgfbr1 <- add_theme(p_tgfbr1) + theme(legend.position = 'none',axis.text.x = element_text(angle = 0))
print(p_tgfbr1)
ggsave(filename = "../figures/drug_tgfbr1_ko.png",
       plot = p_tgfbr1,
       units = "cm", width = 16, height = 12)

# TGFBR2
p_tgfbr2 <- ggboxplot(results_drugs_outs %>% filter(condition==(-3040) | condition==0)  %>%
                        mutate(condition=ifelse(condition==0,drug,paste0('KO ',node,'+',drug))) %>%
                        filter(node=='TGFBR2')%>%
                        select(condition,node,outlet,activity,model_no) %>% unique(),
                      x='condition',y='activity',color = 'outlet',add='jitter')+
  ylab('activity') + labs(color='outlet')+
  facet_wrap(~outlet,scales = 'free_y')
p_tgfbr2 <- add_theme(p_tgfbr2) + theme(legend.position = 'none')
print(p_tgfbr2)
# ggsave(filename = "../figures/drug_tgfbr2_ko.png", 
#        plot = p_tgfbr2, 
#        units = "cm", width = 16, height = 12)


# ### Check how similar the disconnection looks to the KO-----------------------
# disconnect_resutls_tfs <- data.table::fread('../results/allmodels/signal_exploration/TFs_disconnected_node_conditions.csv') %>%
#   select(-V1)
# disconnect_resutls_outs <- data.table::fread('../results/allmodels/signal_exploration/Nodes_output_disconnected_node_conditions.csv') %>%
#   select(-V1)
# #Load annotation file to replace with gene names
# annot <- read.delim('../../../Artificial-Signaling-Network/DrugsANNSignaling/data/uniprot-reviewed_yes+AND+organism__Homo+sapiens+(Human)+[9606]_.tab')
# annot <- annot %>% select(c('uniprot'='Entry'),c('name'='Gene.names...primary..')) %>% unique()
# annot <- annot %>% mutate(name = ifelse(name=='' | name==' ' | is.na(name),uniprot,name))
# ### Get input for plsr
# x <- disconnect_resutls_tfs %>% spread('TF','activity')
# x <- x[,4:107]
# x[x>1] <- 1
# x[x<0] <- 0
# x <- log((x+1e-6)/(1 - x + 1e-6))
# x <- x[,colnames(TF_human)]
# hist(as.matrix(x))
# 
# ## re-annotate data
# disconnect_resutls_tfs <- left_join(disconnect_resutls_tfs,annot,by=c('disconnected node'='uniprot')) %>% select(-`disconnected node`) %>% 
#   select(model_no,c('disconnected node'='name'),stimuli,TF,activity)
# disconnect_resutls_tfs <- left_join(disconnect_resutls_tfs,annot,by=c('TF'='uniprot')) %>% select(-TF) %>% 
#   select(model_no,`disconnected node`,stimuli,c('TF'='name'),activity)
# 
# disconnect_resutls_outs <- left_join(disconnect_resutls_outs,annot,by=c('disconnected node'='uniprot')) %>% select(-`disconnected node`) %>% 
#   select(model_no,c('disconnected node'='name'),stimuli,outlet,activity)
# disconnect_resutls_outs <- left_join(disconnect_resutls_outs,annot,by=c('outlet'='uniprot')) %>% select(-outlet) %>% 
#  select(model_no,`disconnected node`,stimuli,c('outlet'='name'),activity)
# 
# ## phenotype
# Ydisconnected <- predict(plsr_model,x)
# Ydisconnected <- as.data.frame(Ydisconnected)
# Ydisconnected <- cbind(disconnect_resutls_tfs %>% spread('TF','activity') %>%
#                          select(model_no,`disconnected node`,stimuli),
#                        Ydisconnected)
# 
# # from results_tfs and results_outs keep only condition == -3000
# results_tfs_filt <- results_tfs %>% filter(condition == (-3040) | condition ==0)
# results_tfs_filt <- distinct(results_tfs_filt %>% 
#   select(model_no,c('disconnected node'='node'),c('stimuli'='condition'),TF,activity)) %>%
#   mutate(stimuli=ifelse(stimuli==0,'pure untreated','computational KO'))
# 
# results_outs_filt <- results_outs %>% filter(condition == (-3040) | condition ==0)
# results_outs_filt <- distinct(results_outs_filt %>% 
#                                select(model_no,c('disconnected node'='node'),c('stimuli'='condition'),outlet,activity)) %>%
#   mutate(stimuli=ifelse(stimuli==0,'pure untreated','computational KO'))
# 
# Ysimulation_filt <- distinct(Ysimulation %>% filter(condition == (-3040) | condition ==0) %>%
#   select(model_no,c('disconnected node'='name'),c('stimuli'='condition'),NAS,fibrosis)) %>%
#   mutate(stimuli=ifelse(stimuli==0,'pure untreated','computational KO'))
# 
# ## Visualize predicted phenotype
# p_pheno <- ggboxplot(distinct(rbind(Ysimulation_filt,Ydisconnected)),
#                        x='stimuli',y='NAS',color='stimuli',
#                      add='jitter')+
#   ylab('activity') +
#   facet_wrap(~`disconnected node`)
# p_pheno <- add_theme(p_pheno) + theme(axis.title.x = element_blank(),
#                                           axis.text.x = element_blank())
# print(p_pheno)

# ## visualize outlets
# pout_disc <- ggboxplot(rbind(results_outs_filt,disconnect_resutls_outs) %>%
#                          filter(stimuli!='computational KO') %>%
#                          filter(`disconnected node`=='TGFBR1'),
#                        x='stimuli',y='activity',color='stimuli')+
#   ylab('activity') + labs(color='outlet')+
#   scale_y_continuous(c(NA,1.1),breaks = seq(0,1,0.1))+
#   facet_wrap(~outlet,scales='free_x')
# pout_disc <- add_theme(pout_disc) + theme(axis.title.x = element_blank(),
#                                           axis.text.x = element_blank())
# print(pout_disc)
