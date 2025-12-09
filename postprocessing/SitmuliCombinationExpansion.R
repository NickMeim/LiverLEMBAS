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
options(future.globals.maxSize = 2 * 1024^3) # 2 GiB

## Simulate combination of TGFBR1 and IFNAR1---------------
results_ko <- data.table::fread('../results/allmodels/InSilicoKOs/TFs_all_conditions.csv') %>% 
  column_to_rownames('V1')
results_ko <- results_ko %>% filter(type!="KO+control") %>% filter(type!='control')
simulation <- data.table::fread('../results/allmodels/combo_simulation_expanded/TFs_all_conditions_expanded.csv') %>% 
  select(-V1)

#Load annotation file to replace with gene names
annot <- read.delim('../../../Artificial-Signaling-Network/DrugsANNSignaling/data/uniprot-reviewed_yes+AND+organism__Homo+sapiens+(Human)+[9606]_.tab')
annot <- annot %>% select(c('node'='Entry'),c('TF'='Gene.names...primary..')) %>% unique()
annot <- annot %>% filter(node %in% colnames(simulation))
annot <- left_join(data.frame(node = colnames(simulation)[1:104]),annot)
print(all(colnames(simulation)[1:104]==annot$node))
colnames(simulation)[1:104] <- annot$TF

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
options(future.globals.maxSize = 2 * 1024^3) # 2 GiB
Yinsilico <- foreach(i = seq(0,49)) %dopar% {
  stimuli <- simulation %>% filter(model_no==i) %>% filter(type!='untreated') %>% select(IFNa,TGFb)
  as.numeric(simulation$condition[simulation$model_no==i & simulation$type!='untreated'])
  expreriment <- as.matrix(simulation %>% filter(model_no==i) %>% filter(type!='untreated') %>%
                             select(-type,-model_no,-IFNa,-TGFb))
  expreriment[expreriment>1] <-1
  expreriment[expreriment<0] <-0
  expreriment <- log((expreriment+1e-6)/(1 - expreriment + 1e-6))
  colnames(expreriment) <- colnames(simulation)[1:104]
  ## predict
  yhat <- as.data.frame(predict(plsr_model,expreriment)) %>% mutate(model_no=i)
  yhat <- cbind(yhat,stimuli)
  yhat
}
Yinsilico <- do.call(rbind,Yinsilico)
control <- as.matrix(simulation %>% filter(type=='untreated') %>%
                       select(-type,-model_no,-IFNa,-TGFb))
control[control>1] <-1
control[control<0] <-0
control <- log((control+1e-6)/(1 - control + 1e-6))
colnames(control) <- colnames(simulation)[1:104]
Ycontrol <- predict(plsr_model,control)
mean_control_mas <- mean(Ycontrol[,1]) 
se_control_mas <- sd(Ycontrol[,1])/sqrt(50)
mean_control_fib <- mean(Ycontrol[,2]) 
se_control_fib <- mean(Ycontrol[,2]) /sqrt(50)
## save a centered version
Ycontrol <- as.data.frame(Ycontrol) %>% mutate(model_no=seq(0,49,1))
Yinsilico_centered <- left_join(Ycontrol,Yinsilico,by = 'model_no')
Yinsilico_centered <- Yinsilico_centered %>% mutate(NAS = NAS.y - NAS.x)
Yinsilico_centered <- Yinsilico_centered %>% mutate(fibrosis = fibrosis.y-fibrosis.x)
Yinsilico_centered <- Yinsilico_centered %>% 
  select(c("IFNAR1"="IFNa"),c("TGFBR1"="TGFb"),model_no,NAS,fibrosis)
saveRDS(Yinsilico_centered,'../results/allmodels/combo_simulation_expanded/Yinsilico_centered.rds')

# Yinsilico_centered2 <- Yinsilico_centered%>% gather('phenotype','score',-model_no,-IFNAR1,-TGFBR1) %>%
#   group_by(phenotype,IFNAR1,TGFBR1) %>% mutate(mu = mean(score)) %>% mutate(se=sd(score)/sqrt(50))

Yinsilico <- Yinsilico %>% gather('phenotype','score',-model_no,-IFNa,-TGFb) %>%
  group_by(phenotype,IFNa,TGFb) %>% mutate(mu = mean(score)) %>% mutate(se=sd(score)/sqrt(50))

### Visualize simulation
Yall <- Yinsilico %>%
  mutate(phenotype = ifelse(phenotype=='NAS','MAS','Fibrosis stage'))
Yall <- Yall %>% mutate(IFNa=as.numeric(IFNa)) %>% mutate(TGFb=as.numeric(TGFb))
mas_plot <- ggplot(Yall %>% filter(phenotype=='MAS') %>% select(IFNa,TGFb,mu,se,phenotype) %>% unique(),
                   aes(x=IFNa,y=TGFb,fill=mu))+
  geom_tile()+
  scale_fill_gradient2(midpoint = mean_control_mas)+
  labs(fill='MAS',x = 'stimulation of IFNAR1',y='stimulation of TGFBR1')+
  scale_x_continuous(n.breaks = 10)+
  scale_y_continuous(n.breaks = 10)+
  geom_hline(yintercept = 0,lwd=1,color='black')+
  geom_vline(xintercept = 0,lwd=1,color='black')
mas_plot <- add_theme(mas_plot) + 
  theme(legend.position = 'top',text = element_text(size=20),
        panel.border = element_blank(),
        panel.grid = element_blank())
print(mas_plot)
ggsave(filename = "../figures/expanded_ifna_tgfb_combo_mas_plot.pdf", plot = mas_plot, 
       device = cairo_pdf,
       units = "cm", width = 16, height = 12)

fib_plot <- ggplot(Yall %>% filter(phenotype!='MAS') %>% select(IFNa,TGFb,mu,se,phenotype) %>% unique(),
                   aes(x=IFNa,y=TGFb,fill=mu))+
  geom_tile()+
  scale_fill_gradient2(midpoint = mean_control_fib)+
  labs(fill='Fibrosis stage',x = 'stimulation of IFNAR1',y='stimulation of TGFBR1')+
  scale_x_continuous(n.breaks = 10)+
  scale_y_continuous(n.breaks = 10)+
  geom_hline(yintercept = 0,lwd=1,color='black')+
  geom_vline(xintercept = 0,lwd=1,color='black')
fib_plot <- add_theme(fib_plot) +
  theme(legend.position = 'top',text = element_text(size=20),
        panel.border = element_blank(),
        panel.grid = element_blank())
print(fib_plot)
ggsave(filename = "../figures/expanded_ifna_tgfb_combo_fibrosis_plot.pdf", plot = fib_plot, 
       device = cairo_pdf,
       units = "cm", width = 16, height = 12)
