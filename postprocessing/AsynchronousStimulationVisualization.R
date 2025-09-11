#### NEED TO CHANGE. THIS IS AN OLD TEMPLATE
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
simulation <- data.table::fread('../results/chained_async/TFs_chained_async_avg_then_chain.csv')
tt <- simulation %>% select(experiment,timepoint,description)

#Load annotation file to replace with gene names
annot <- read.delim('../../../Artificial-Signaling-Network/DrugsANNSignaling/data/uniprot-reviewed_yes+AND+organism__Homo+sapiens+(Human)+[9606]_.tab')
annot <- annot %>% select(c('node'='Entry'),c('TF'='Gene.names...primary..')) %>% unique()
annot <- annot %>% filter(node %in% colnames(simulation))
annot <- left_join(data.frame(node = colnames(simulation)[7:110]),annot)
print(all(colnames(simulation)[7:110]==annot$node))
colnames(simulation)[7:110] <- annot$TF

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
TF_human <- TF_human[,colnames(simulation)[7:110]]
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
options(future.globals.maxSize = 2 * 1024^3) # 2 GiB
stimuli <- simulation %>% select(-all_of(colnames(TF_human)))
expreriment <- as.matrix(simulation %>% select(all_of(colnames(TF_human))))
expreriment[expreriment>1] <-1
expreriment[expreriment<0] <-0
expreriment <- log((expreriment+1e-6)/(1 - expreriment + 1e-6))
yhat <- as.data.frame(predict(plsr_model,expreriment))
Yinsilico <- cbind(stimuli,yhat)



# control <- as.matrix(simulation %>% filter(type=='untreated') %>%
#                        select(-type,-model_no,-IFNa,-TGFb))
# control[control>1] <-1
# control[control<0] <-0
# control <- log((control+1e-6)/(1 - control + 1e-6))
# colnames(control) <- colnames(simulation)[1:104]
# Ycontrol <- predict(plsr_model,control)
# mean_control_mas <- mean(Ycontrol[,1]) 
# se_control_mas <- sd(Ycontrol[,1])/sqrt(50)
# mean_control_fib <- mean(Ycontrol[,2]) 
# se_control_fib <- mean(Ycontrol[,2]) /sqrt(50)
# Yinsilico <- Yinsilico %>% gather('phenotype','score',-model_no,-IFNa,-TGFb) %>%
#   group_by(phenotype,IFNa,TGFb) %>% mutate(mu = mean(score)) %>% mutate(se=sd(score)/sqrt(50))

### Visualize simulation
Yinsilico <- Yinsilico %>% filter(description!='empty')
Yall <- distinct(Yinsilico %>% 
                   mutate(T_at_0h = ifelse(is.na(T_at_0h),0,T_at_0h)) %>%
                   mutate(T_at_2h = ifelse(is.na(T_at_2h),0,T_at_2h)) %>%
                   mutate(T_at_8h = ifelse(is.na(T_at_8h),0,T_at_8h)) %>%
                   mutate(T_at_24h = ifelse(is.na(T_at_24h),0,T_at_24h)) %>%
                   mutate(I_at_0h = ifelse(is.na(T_at_0h),0,T_at_0h)) %>%
                   mutate(I_at_2h = ifelse(is.na(T_at_2h),0,T_at_2h)) %>%
                   mutate(I_at_8h = ifelse(is.na(T_at_8h),0,T_at_8h)) %>%
                   mutate(I_at_24h = ifelse(is.na(T_at_24h),0,T_at_24h)) %>%
                   mutate(TGFBR1 = T_at_0h+T_at_2h+T_at_8h+T_at_24h) %>%
                   mutate(IFNAR1 = I_at_0h+I_at_2h+I_at_8h+I_at_24h) %>%
                   select(experiment,timepoint,description,IFNAR1,TGFBR1,NAS,fibrosis))
mas_plot <- ggplot(Yall,aes(x=TGFBR1,y=IFNAR1,fill=NAS))+
  geom_tile()+
  scale_fill_gradient2(midpoint = 3)+
   labs(fill='MAS',x = 'stimulation of IFNAR1',y='stimulation of TGFBR1')+
  scale_x_continuous(n.breaks = 10)+
  scale_y_continuous(n.breaks = 10)+
  geom_hline(yintercept = 0,lwd=1,color='black')+
  geom_vline(xintercept = 0,lwd=1,color='black') + 
  facet_wrap(vars(experiment,timepoint))
mas_plot <- add_theme(mas_plot) + 
  theme(legend.position = 'top',text = element_text(size=20),
        panel.border = element_blank(),
        panel.grid = element_blank())
print(mas_plot)


fib_plot <-ggplot(Yall,aes(x=TGFBR1,y=IFNAR1,fill=fibrosis))+
  geom_tile()+
  scale_fill_gradient2(midpoint = 3)+
  labs(fill='MAS',x = 'stimulation of IFNAR1',y='stimulation of TGFBR1')+
  scale_x_continuous(n.breaks = 10)+
  scale_y_continuous(n.breaks = 10)+
  geom_hline(yintercept = 0,lwd=1,color='black')+
  geom_vline(xintercept = 0,lwd=1,color='black') + 
  facet_wrap(vars(experiment,timepoint))
fib_plot <- add_theme(fib_plot) +
  theme(legend.position = 'top',text = element_text(size=20),
        panel.border = element_blank(),
        panel.grid = element_blank())
print(fib_plot)
