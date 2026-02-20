library(tidyverse)
library(forcats)
library(doFuture)
library(ropls)
library(patchwork)
# parallel: set number of workers
cores <- 16
registerDoFuture()
plan(multisession,workers = cores)
source("../postprocessing/plotting_functions.R")
source('../postprocessing/functions_translation.R')
options(future.globals.maxSize = 2 * 1024^3) # 2 GiB
library(rstatix)
library(plotly)
library(htmlwidgets)
library(igraph)
library(MESS)
library(scales)

# figure settings
pub_font <- "Arial"


## Simulate grid combination of TGFBR1 and IFNAR1 plus inhibition of certain nodes---------------
simulation <- data.table::fread('../results/insilico_grid_combinations_ko/insilico_grid_multi_ko_results.csv')
Ystim_no_ko <- readRDS('../results/allmodels/combo_simulation_expanded/Yinsilico.rds')
Ycontrol <- readRDS('../results/allmodels/combo_simulation_grid_multiKO/Ycontrol.rds')

#Load annotation file to replace with gene names
annot <- read.delim('../../../Artificial-Signaling-Network/DrugsANNSignaling/data/uniprot-reviewed_yes+AND+organism__Homo+sapiens+(Human)+[9606]_.tab')
annot <- annot %>% select(c('node'='Entry'),c('TF'='Gene.names...primary..')) %>% unique()
annot <- annot %>% filter(node %in% colnames(simulation))
annot <- left_join(data.frame(node = colnames(simulation)[5:108]),annot)
print(all(colnames(simulation)[5:108]==annot$node))
colnames(simulation)[5:108] <- annot$TF

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
TF_human <- TF_human[,colnames(simulation)[5:108]]
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
options(future.globals.maxSize = 2 * 1024^3) # 2 GiB
Yinsilico <- foreach(i = seq(1,50)) %dopar% {
  stimuli <- simulation %>% filter(Model_No==i) %>% 
    select(c('IFNa'='IFNAR1_Stim'),c('TGFb'='TGFBR1_Stim'),c('type'='Scenario'))
  expreriment <- as.matrix(simulation %>% filter(Model_No==i) %>% 
                             select(-Scenario,-Model_No,-IFNAR1_Stim,-TGFBR1_Stim))
  expreriment[expreriment>1] <-1
  expreriment[expreriment<0] <-0
  expreriment <- log((expreriment+1e-6)/(1 - expreriment + 1e-6))
  colnames(expreriment) <- colnames(simulation)[1:104]
  ## predict
  yhat <- as.data.frame(predict(plsr_model,expreriment)) %>% mutate(Model_No=i)
  yhat <- cbind(yhat,stimuli)
  yhat
}
Yinsilico <- do.call(rbind,Yinsilico)

mean_control_mas <- mean(Ycontrol[,1]) 
se_control_mas <- sd(Ycontrol[,1])/sqrt(50)
mean_control_fib <- mean(Ycontrol[,2]) 
se_control_fib <- mean(Ycontrol[,2]) /sqrt(50)

## Combine
Yinsilico <- Yinsilico %>% gather('phenotype','score',-Model_No,-IFNa,-TGFb,-type) %>%
  group_by(phenotype,type,IFNa,TGFb) %>% mutate(mu = mean(score)) %>% mutate(se=sd(score)/sqrt(50))
Yinsilico <- Yinsilico %>%
  mutate(phenotype = ifelse(phenotype=='NAS','MAS','Fibrosis stage'))
Yinsilico <- Yinsilico %>% mutate(IFNa=as.numeric(IFNa)) %>% mutate(TGFb=as.numeric(TGFb))

Yall <- rbind(Yinsilico,
              Ystim_no_ko %>% mutate(type='no KO') %>%  mutate(phenotype = ifelse(phenotype=='NAS','MAS','Fibrosis stage')) %>%
                mutate(IFNa=as.numeric(IFNa)) %>% mutate(TGFb=as.numeric(TGFb)) %>%
                select(c('Model_No'='model_no'),IFNa,TGFb,type,phenotype,score,mu,se))
Yall$type <- factor(Yall$type,
                    levels = c('no KO','TP63_KO','SMAD3_KO','TP63_SMAD3_KO',
                               'STAT2_KO','STAT3_KO','STAT2_STAT3_KO',
                               'STAT2_TP63_SMAD3_KO','STAT3_STAT2_TP63_SMAD3_KO'))

### Visualize---------------
# We have Yall, Yinsilico, Ycontrol, mean and se of control
mas_plot <- ggplot(Yall %>% filter(phenotype=='MAS') %>% select(IFNa,TGFb,mu,se,phenotype,type) %>% unique(),
                   aes(x=IFNa,y=TGFb,fill=mu))+
  geom_tile()+
  scale_fill_gradient2(midpoint = mean_control_mas)+
  labs(fill='MAS',x = 'stimulation of IFNAR1',y='stimulation of TGFBR1')+
  scale_x_continuous(n.breaks = 10)+
  scale_y_continuous(n.breaks = 10)+
  geom_hline(yintercept = 0,lwd=1,color='black')+
  geom_vline(xintercept = 0,lwd=1,color='black') + 
  facet_wrap(~type)
mas_plot <- add_theme(mas_plot) + 
  theme(legend.position = 'top',text = element_text(size=20),
        panel.border = element_blank(),
        panel.grid = element_blank()) +
  theme(
    text = element_text(family = pub_font, size = 20),
    plot.title = element_text(family = pub_font),
    axis.title = element_text(family = pub_font),
    axis.text = element_text(family = pub_font),
    legend.title = element_text(family = pub_font),
    legend.text = element_text(family = pub_font),
    strip.text = element_text(family = pub_font)
  )
print(mas_plot)
# save
ggsave(
  filename = "../figures/combinatorialKOs_mas_plot_8hr.pdf",
  plot = mas_plot,
  device = cairo_pdf,
  width = 320,
  height = 200,
  units = "mm",
  bg = "white",
  family = pub_font,
  dpi = 600
)
ggsave(
  filename = "../figures/combinatorialKOs_mas_plot_8hr.png",
  plot = mas_plot,
  device = ragg::agg_png,
  width = 320,
  height = 200,
  units = "mm",
  dpi = 600,
  bg = "white"
)


fib_plot <- ggplot(Yall %>% filter(phenotype!='MAS') %>% select(IFNa,TGFb,mu,se,phenotype,type) %>% unique(),
                   aes(x=IFNa,y=TGFb,fill=mu))+
  geom_tile()+
  scale_fill_gradient2(midpoint = mean_control_fib)+
  labs(fill='Fibrosis stage',x = 'stimulation of IFNAR1',y='stimulation of TGFBR1')+
  scale_x_continuous(n.breaks = 10)+
  scale_y_continuous(n.breaks = 10)+
  geom_hline(yintercept = 0,lwd=1,color='black')+
  geom_vline(xintercept = 0,lwd=1,color='black') + 
  facet_wrap(~type)
fib_plot <- add_theme(fib_plot) + 
  theme(legend.position = 'top',text = element_text(size=20),
        panel.border = element_blank(),
        panel.grid = element_blank())+
  theme(
    text = element_text(family = pub_font, size = 20),
    plot.title = element_text(family = pub_font),
    axis.title = element_text(family = pub_font),
    axis.text = element_text(family = pub_font),
    legend.title = element_text(family = pub_font),
    legend.text = element_text(family = pub_font),
    strip.text = element_text(family = pub_font)
  )
print(fib_plot)
# save
ggsave(
  filename = "../figures/combinatorialKOs_fibrosis_plot_8hr.pdf",
  plot = fib_plot,
  device = cairo_pdf,
  width = 320,
  height = 200,
  units = "mm",
  bg = "white",
  family = pub_font,
  dpi = 600
)
ggsave(
  filename = "../figures/combinatorialKOs_fibrosis_plot_8hr.png",
  plot = fib_plot,
  device = ragg::agg_png,
  width = 320,
  height = 200,
  units = "mm",
  dpi = 600,
  bg = "white"
)