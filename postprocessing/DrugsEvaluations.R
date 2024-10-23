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
source('functions_translation.R')
library(ropls)
library(caret)

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

#Load annotation file to replace with gene names
annot <- read.delim('../../../Artificial-Signaling-Network/DrugsANNSignaling/data/uniprot-reviewed_yes+AND+organism__Homo+sapiens+(Human)+[9606]_.tab')
annot <- annot %>% select(c('TF'='Entry'),c('TF_symbol'='Gene.names...primary..')) %>% unique()
annot <- annot %>% filter(TF %in% TFdrugs$TF)
# annot <- annot %>% column_to_rownames('node')
names_annot <- annot$TF
TFdrugs <- left_join(TFdrugs,annot) %>% select(-TF)
TFdrugs <- TFdrugs %>% select(drug,dose,model_no,c('TF'='TF_symbol'),activity)

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
TF_human <- TF_human[,unique(TFdrugs$TF)]
# TF_human <- 1/(1+exp(-TF_human))

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

### Predict drugs phenotypes-----------
Ydrugs <- foreach(i = seq(0,49)) %dopar% {
  expreriment <- as.matrix(TFdrugs %>% filter(model_no==i) %>% 
                             mutate(condition = paste0(drug,'+',dose)) %>%
                             select(-drug,-dose,-model_no) %>% 
                             spread('TF','activity') %>% 
                             column_to_rownames('condition'))
  expreriment <- expreriment[,colnames(TF_human)]
  expreriment <- scale(expreriment)
  # control <- as.matrix(left_join(TFdrugs_control,annot) %>% select(-TF) %>%
  #                        filter(model_no==i) %>% 
  #                        select(-model_no) %>% 
  #                        spread('TF_symbol','control_activity') %>% 
  #                        column_to_rownames('drug'))
  # control <- control[,colnames(TF_human)]
  # control <- scale(control,center = TRUE,scale = FALSE)
  ## predict
  yhat <- as.data.frame(predict(plsr_model,expreriment)) %>% mutate(model_no=i) %>% 
    rownames_to_column('condition') %>%
    separate(condition,into=c('drug','dose'),sep="\\+")   
  # ## repeat for control
  # ycontrol <- as.data.frame(predict(plsr_model,control)) %>% mutate(model_no=i)%>% 
  #   rownames_to_column('drug')
  # yhat <- left_join(yhat,ycontrol,by=c('drug','model_no'))
  # yhat <- yhat %>% mutate(NAS=NAS.x-NAS.y , fibrosis = fibrosis.x-fibrosis.y)
  yhat %>% select(drug,dose,model_no,NAS,fibrosis)
}
Ydrugs <- do.call(rbind,Ydrugs)
Ydrugs <- Ydrugs %>% gather('phenotype','score',-model_no,-drug,-dose)
Ydrugs <- Ydrugs %>% mutate(phenotype=ifelse(phenotype=='NAS','MAS','Fibrosis stage'))
Ydrugs <- Ydrugs %>% group_by(drug,phenotype) %>% mutate(max_score= max(score)) %>% ungroup() %>%
  group_by(drug,dose,phenotype) %>% mutate(mu = mean(score),se=sd(score)/sqrt(50)) %>% ungroup()
candidates <- Ydrugs %>% select(drug,phenotype,max_score) %>% unique() %>%
  spread(phenotype,max_score)
fibrosis_top <- candidates$drug[order(-candidates$`Fibrosis stage`)][1:10]
mas_top <- candidates$drug[order(-candidates$MAS)][1:10]

Ydrugs$dose <- factor(Ydrugs$dose,levels = c('Low','Middle','High'))
ggboxplot(Ydrugs %>% filter(drug %in% c(fibrosis_top,mas_top)) %>% filter(phenotype=='MAS') %>%
         select(drug,dose,score,mu,se) %>% unique(),
       x='dose',y='score',add='jitter')+
  scale_y_continuous(limits = c(0,8))+
  facet_wrap(~drug)

ggplot(Ydrugs %>% filter(drug %in% c(fibrosis_top,mas_top)) %>% filter(phenotype=='MAS') %>%
            select(drug,dose,score,mu,se) %>% unique() %>%
         group_by(drug) %>% mutate(counts = n_distinct(dose)) %>% ungroup() %>%
         filter(counts >2),
          aes(x=dose,y=score))+
  geom_smooth(aes(group=drug),se=TRUE,color='black')+
  geom_point(aes(x=dose,y=mu))+
  scale_y_continuous(limits = c(0,8))+
  facet_wrap(~drug)
