library(tidyverse)
library(forcats)
library(doFuture)
library(ropls)
library(patchwork)
source("plotting_functions.R")
source('functions_translation.R')
# options(future.globals.maxSize = 2 * 1024^3) # 2 GiB

## Simulate combination of TGFBR1 and IFNAR1---------------
simulation <- data.table::fread('../results/async_experiments_tidy/async_stimulation_results_no_carry_on.csv')
tt <- simulation %>% select(all_of(colnames(simulation)[105:110]))

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
stimuli <- simulation %>% select(-all_of(colnames(TF_human)))
expreriment <- as.matrix(simulation %>% select(all_of(colnames(TF_human))))
expreriment[expreriment>1] <-1
expreriment[expreriment<0] <-0
expreriment <- log((expreriment+1e-6)/(1 - expreriment + 1e-6))
yhat <- as.data.frame(predict(plsr_model,expreriment))
Yinsilico <- cbind(stimuli,yhat)
gc()

### Visualize simulation
timepoint <- '24h'
mas_plot <- ggplot(Yinsilico %>% filter(measured_at==timepoint),
                   aes(x=IFNa_strength,y=TGFb_strength,fill=NAS))+
  geom_tile()+
  scale_fill_gradient2(midpoint = mean(as.matrix((Yinsilico %>% 
                                                    filter(measured_at==timepoint) %>% 
                                                    filter(TGFb_strength==0)%>% 
                                                    filter(IFNa_strength==0) %>% 
                                                    filter(TGFb_applied_at==IFNa_applied_at) %>% 
                                                    select(NAS)))))+
   labs(fill='MAS',x = 'stimulation of IFNAR1',y='stimulation of TGFBR1')+
  scale_x_continuous(n.breaks = 10)+
  scale_y_continuous(n.breaks = 10)+
  geom_hline(yintercept = 0,lwd=1,color='black')+
  geom_vline(xintercept = 0,lwd=1,color='black') + 
  facet_grid(paste0('TGFB at ',TGFb_applied_at) ~ paste0('IFNA at ',IFNa_applied_at))
mas_plot <- add_theme(mas_plot) + 
  theme(legend.position = 'top',text = element_text(size=20),
        panel.border = element_blank(),
        panel.grid = element_blank(),
        )
# print(mas_plot)
ggsave(filename = paste0('../results/async_experiments_tidy/plots/asychronous_MAS_measured_at_',
                         timepoint,'.png'),
       plot=mas_plot,
       width = 16,
       height = 10,
       units = 'in',
       dpi=600)

fib_plot <-ggplot(Yinsilico %>% filter(measured_at==timepoint),
                  aes(x=IFNa_strength,y=TGFb_strength,fill=fibrosis))+
  geom_tile()+
  scale_fill_gradient2(midpoint = mean(as.matrix((Yinsilico %>% 
                                                    filter(measured_at==timepoint) %>% 
                                                    filter(TGFb_strength==0)%>% 
                                                    filter(IFNa_strength==0) %>% 
                                                    filter(TGFb_applied_at==IFNa_applied_at) %>% 
                                                    select(fibrosis)))))+
  labs(fill='fibrosis',x= 'stimulation of IFNAR1',y='stimulation of TGFBR1')+
  scale_x_continuous(n.breaks = 10)+
  scale_y_continuous(n.breaks = 10)+
  geom_hline(yintercept = 0,lwd=1,color='black')+
  geom_vline(xintercept = 0,lwd=1,color='black') + 
  facet_grid(paste0('TGFB at ',TGFb_applied_at) ~ paste0('IFNA at ',IFNa_applied_at))
fib_plot <- add_theme(fib_plot) +
  theme(legend.position = 'top',text = element_text(size=20),
        panel.border = element_blank(),
        panel.grid = element_blank())
# print(fib_plot)
ggsave(filename = paste0('../results/async_experiments_tidy/plots/asychronous_fibrosis_measured_at_',
                         timepoint,'.png'),
       plot=fib_plot,
       width = 16,
       height = 10,
       units = 'in',
       dpi=600)
