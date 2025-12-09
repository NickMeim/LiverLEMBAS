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


##### I MUST GET NET FOR MAS TOO
##### I MUST MASK TFS NOT IN THE MODEL

## Simulate combination of TGFBR1 and IFNAR1---------------
results_ko <- data.table::fread('results/TrimmedSimulation_8 hour.csv') %>% 
  column_to_rownames('V1')
control_trimmed <- data.table::fread('results/TrimmedSimulation_control_8 hour.csv') %>% 
  select(-V1)

#Load annotation file to replace with gene names
annot <- read.delim('../../../Artificial-Signaling-Network/DrugsANNSignaling/data/uniprot-reviewed_yes+AND+organism__Homo+sapiens+(Human)+[9606]_.tab')
annot <- annot %>% select(c('node'='Entry'),c('TF'='Gene.names...primary..')) %>% unique()
annot <- annot %>% filter(node %in% colnames(control_trimmed))
annot <- left_join(data.frame(node = colnames(control_trimmed)[1:104]),annot)
print(all(colnames(control_trimmed)[1:104]==annot$node))
colnames(control_trimmed)[1:104] <- annot$TF
print(all(colnames(results_ko)[1:104]==annot$node))
colnames(results_ko)[1:104] <- annot$TF

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
TF_human <- TF_human[,colnames(control_trimmed)[1:104]]
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
Yinsilico <- foreach(i = seq(0,49)) %dopar% {
  stimuli <- results_ko %>% filter(model_no==i) %>% filter(type!='untreated') %>% select(IFNAR1,TGFBR1)
  experiment <- as.matrix(results_ko %>% filter(model_no==i) %>% filter(type!='untreated') %>%
                             select(-type,-model_no,-IFNAR1,-TGFBR1))
  experiment[experiment>1] <-1
  experiment[experiment<0] <-0
  experiment <- log((experiment+1e-6)/(1 - experiment + 1e-6))
  colnames(experiment) <- colnames(control_trimmed)[1:104]
  zeroed <- !(colnames(experiment) %in% c('TP63','SMAD3','STAT2','GLI2','POU2F1','HIF1A','FOXO1','SMAD4','TWIST1'))
  experiment[,zeroed] <- 0
  ## predict
  yhat <- as.data.frame(predict(plsr_model,experiment)) %>% mutate(model_no=i)
  yhat <- cbind(yhat,stimuli)
  yhat
}
Yinsilico <- do.call(rbind,Yinsilico)
gc()
control <- as.matrix(control_trimmed %>% filter(type=='untreated') %>%
                       select(-type,-model_no))
control[control>1] <-1
control[control<0] <-0
control <- log((control+1e-6)/(1 - control + 1e-6))
colnames(control) <- colnames(control_trimmed)[1:104]
zeroed <- !(colnames(control) %in% c('TP63','SMAD3','STAT2','GLI2','POU2F1','HIF1A','FOXO1','SMAD4','TWIST1'))
control[,zeroed] <- 0
Ycontrol <- predict(plsr_model,control)
mean_control_mas <- mean(Ycontrol[,1]) 
se_control_mas <- sd(Ycontrol[,1])/sqrt(50)
mean_control_fib <- mean(Ycontrol[,2]) 
se_control_fib <- mean(Ycontrol[,2]) /sqrt(50)
Ycontrol <- as.data.frame(Ycontrol) %>% mutate(model_no = seq(0,49,1))
Yinsilico <- left_join(Ycontrol,Yinsilico,by = 'model_no')
Yinsilico <- Yinsilico %>% mutate(NAS = NAS.y - NAS.x)
Yinsilico <- Yinsilico %>% mutate(fibrosis = fibrosis.y-fibrosis.x)
Yinsilico <- Yinsilico %>% select(IFNAR1,TGFBR1,model_no,NAS,fibrosis)
# Yinsilico$NAS <- Yinsilico$NAS - mean_control_mas
# Yinsilico$fibrosis <- Yinsilico$fibrosis - mean_control_fib
Yinsilico <- Yinsilico %>% gather('phenotype','score',-model_no,-IFNAR1,-TGFBR1) %>%
  group_by(phenotype,IFNAR1,TGFBR1) %>% mutate(mu = mean(score)) %>% mutate(se=sd(score)/sqrt(50))

### Visualize simulation
Yall <- Yinsilico %>%
  mutate(phenotype = ifelse(phenotype=='NAS','MAS','Fibrosis stage'))
Yall <- Yall %>% mutate(IFNAR1=as.numeric(IFNAR1)) %>% mutate(TGFBR1=as.numeric(TGFBR1))
mas_plot <- ggplot(Yall %>% filter(phenotype=='MAS') %>% select(IFNAR1,TGFBR1,mu,se,phenotype) %>% unique(),
                   aes(x=IFNAR1,y=TGFBR1,fill=mu))+
  geom_tile()+
  scale_fill_gradient2(midpoint = 0)+
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
ggsave(filename = "../figures/trimmedNetSimulationGrid_ifna_tgfb_combo_mas_plot.pdf", plot = mas_plot, 
       device = cairo_pdf,
       units = "cm", width = 16, height = 12)

fib_plot <- ggplot(Yall %>% filter(phenotype!='MAS') %>% select(IFNAR1,TGFBR1,mu,se,phenotype) %>% unique(),
                   aes(x=IFNAR1,y=TGFBR1,fill=mu))+
  geom_tile()+
  scale_fill_gradient2(midpoint = 0)+
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
ggsave(filename = "../figures/trimmedNetSimulationGrid_ifna_tgfb_combo_fibrosis_plot.pdf", plot = fib_plot, 
       device = cairo_pdf,
       units = "cm", width = 16, height = 12)

### Compare DMAS and Dfibrosis from the mean with those from the whole network----------
gc()
Ycentered_full <- readRDS('../results/allmodels/combo_simulation_expanded/Yinsilico_centered.rds')
Ycentered_full <- Ycentered_full %>% gather('phenotype','score_full',-model_no,-IFNAR1,-TGFBR1) %>%
  group_by(phenotype,IFNAR1,TGFBR1) %>% mutate(mu_full = mean(score_full)) %>% mutate(se_full=sd(score_full)/sqrt(50))%>%
    ungroup()
Ycentered_full <- Ycentered_full %>% mutate(IFNAR1=as.numeric(IFNAR1)) %>% mutate(TGFBR1=as.numeric(TGFBR1))
df_compare <- left_join(Ycentered_full,Yinsilico)
df_plot <- distinct(df_compare %>% select(-score,-score_full,-model_no))
ccc_lin <- function(x, y, na.rm = TRUE) {
  if (na.rm) {
    ok <- is.finite(x) & is.finite(y)
    x <- x[ok]; y <- y[ok]
  }
  mx <- mean(x); my <- mean(y)
  sx <- sd(x);  sy <- sd(y)
  r  <- cor(x, y)
  
  ccc <- (2 * r * sx * sy) / (sx^2 + sy^2 + (mx - my)^2)
  
  list(ccc = ccc, r = r, mx = mx, my = my, sx = sx, sy = sy, n = length(x))
}
xvar <- "mu_full"
yvar <- "mu"

ccc_by_task <- df_plot %>%
  group_by(phenotype) %>%
  summarise(
    ccc = ccc_lin(.data[[xvar]], .data[[yvar]])$ccc,
    r   = cor(.data[[xvar]], .data[[yvar]], use = "complete.obs"),
    n   = sum(is.finite(.data[[xvar]]) & is.finite(.data[[yvar]])),
    .groups = "drop"
  ) %>%
  mutate(label = sprintf("CCC = %.2f\nR = %.2f\nn = %d", ccc, r, n))

ggplot(df_plot, aes(x = .data[[xvar]], y = .data[[yvar]])) +
  geom_point(alpha = 0.2, size = 1) +
  geom_abline(slope = 1, intercept = 0, linetype = 2, linewidth = 1) +
  facet_wrap(~ phenotype) +
  geom_text(
    data = ccc_by_task,
    aes(label = label),
    x = -Inf, y = Inf,
    inherit.aes = FALSE,
    hjust = -0.1, vjust = 1.1,
    size = 5
  ) +
  labs(x = "prediction with full net",
       y = "prediction with trimmed net") +
  theme_bw()
ggsave('../figures/trimmed_vs_full_simulation_8hours.png',
       width = 12,
       height = 9,
       units = 'cm',
       dpi=600)

Ybinary <- df_plot %>% mutate(mu=1*(mu>0)) %>%
  mutate(mu_full=1*(mu_full>0))
conf_nas <- caret::confusionMatrix(data = factor(Ybinary$mu[Ybinary$phenotype=='NAS'],
                                             levels = c(0,1)),
                              reference= factor(Ybinary$mu_full[Ybinary$phenotype=='NAS'],
                                                levels = c(0,1)))
print(conf_nas)

conf_fibrosis <- caret::confusionMatrix(data = factor(Ybinary$mu[Ybinary$phenotype!='NAS'],
                                                 levels = c(0,1)),
                                   reference= factor(Ybinary$mu_full[Ybinary$phenotype!='NAS'],
                                                     levels = c(0,1)))
print(conf_fibrosis)
