library(tidyverse)
library(dorothea)
library(ggplot2)
library(ggpubr)
library(ggsignif)
library(patchwork)
library(ropls)
source("plotting_functions.R")
source('functions_translation.R')

### Load bulk RNA counts (already CPM normalized)--------------
zip_path <- '../data/RNAseq/4HJ57S_results.zip'
file_in_zip <- "4HJ57S-expression-matrix.tsv"
sample_map <- c(
  "4HJ57S_1_cpm"  = "DMSO_1",
  "4HJ57S_2_cpm"  = "DMSO_2",
  "4HJ57S_3_cpm"  = "IFNa_1",
  "4HJ57S_4_cpm"  = "IFNa_2",
  "4HJ57S_5_cpm"  = "TGFb_1",
  "4HJ57S_6_cpm"  = "TGFb_2",
  "4HJ57S_7_cpm"  = "Both_1",
  "4HJ57S_8_cpm"  = "Both_2",
  "4HJ57S_9_cpm"  = "DPP_1",
  "4HJ57S_10_cpm" = "DPP_2",
  "4HJ57S_11_cpm" = "SIS3_1",
  "4HJ57S_12_cpm" = "SIS3_2",
  "4HJ57S_13_cpm" = "DEU_1",
  "4HJ57S_14_cpm" = "DEU_2",
  "4HJ57S_15_cpm" = "NSC_1",
  "4HJ57S_16_cpm" = "NSC_2"
)
# Base R
cpm_counts <- read.delim(
  file = unz(zip_path, file_in_zip),
  header = TRUE,
  sep = "\t",
  check.names = FALSE
)

cpm_counts <- cpm_counts %>% mutate(gene_name=ifelse(gene_name=="",gene_id,gene_name))

# Case-insensitive matching (matches "cpm", "CPM", etc.)
cols_to_keep <- grep("gene_name|gene_id|gene_biotype|cpm", colnames(cpm_counts), ignore.case = TRUE, value = TRUE)
# Subset columns
cpm_counts <- cpm_counts[, cols_to_keep, drop = FALSE]

cpm_counts <- aggregate(cpm_counts[,4:ncol(cpm_counts)],by = list(cpm_counts$gene_name),FUN = sum) %>%
  column_to_rownames('Group.1')
colnames(cpm_counts) <- ifelse(
  colnames(cpm_counts) %in% names(sample_map),
  sample_map[colnames(cpm_counts)],
  colnames(cpm_counts)
)
log_cpm <- log10(cpm_counts+1)

# Get standarized, aggregated and normalized gene expression
gex <- t(log_cpm - rowMeans(log_cpm))

### Infer TF activity----------------
dorotheaData = read.table('../data/dorothea.tsv', sep = "\t", header=TRUE)
confidenceFilter = is.element(dorotheaData$confidence, c('A', 'B'))
dorotheaData = dorotheaData[confidenceFilter,]
# Estimate TF activities
settings = list(verbose = TRUE, minsize = 5)
TF_activities = run_viper(t(gex), dorotheaData, options =  settings)
tfs <- rownames(TF_activities)
tf_mat <- TF_activities
TF_activities <- as.data.frame(TF_activities) %>% rownames_to_column('TF') %>% gather(key='condition',value='activity',-TF)
TF_activities <- TF_activities %>% separate('condition',into=c('condition','repID'))


### Visualize TP63, SMAD3, STAT2, STAT3----------
ggdotplot(TF_activities %>% filter(TF %in% c('TP63', 
                                             'SMAD3', 
                                             'STAT2', 
                                             'STAT3')),
          x='condition',y='activity',fill='condition') +
  facet_wrap(~TF,scales = 'free_y')+ 
  theme(legend.position = 'none')+
  geom_hline(yintercept = 0,linetype='dashed',lwd=1)
ggsave(filename = '../figures/RNA_TFact_validation.png',width = 9,height = 9, units = 'in')


### Load previous data and use PLSR to score phenotype-----------
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

### Intersection of current and previous gene expression
common_genes <- intersect(colnames(Xh),rownames(log_cpm))

# TF_human <- run_viper(t(Xh), dorotheaData, options =  settings)
# TF_human <-t(TF_human)
# TF_human <- TF_human[,tfs]

## train one model with all the data
plsr_model <- opls(x = Xh, #TF_human, 
                   y = Yh,
                   predI = 8,
                   crossvalI = 1,
                   scaleC = "center",
                   fig.pdfC = "none",
                   info.txtC = "none")
# yhat <- as.data.frame(predict(plsr_model,t(tf_mat)[,tfs]))
yhat <- as.data.frame(predict(plsr_model,gex[,common_genes]))
yhat <- yhat %>% rownames_to_column('condition') %>% separate('condition',into=c('condition','repID'))
yhat <- yhat %>% gather('phenotype','score',-condition,-repID)

ggdotplot(yhat,
          x='condition',y='score',fill='condition') +
  facet_wrap(~phenotype,scales = 'free_y')+ 
  theme(legend.position = 'none')+
  geom_hline(yintercept = 0,linetype='dashed',lwd=1)
ggsave(filename = '../figures/RNA_PhenoScore_validation.png',width = 9,height = 9, units = 'in')
