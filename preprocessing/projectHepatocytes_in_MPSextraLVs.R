library(tidyverse)
library(LIV2Trans)
library(AnnotationDbi)
library(org.Hs.eg.db)
# Function for loading datasets and appending them to a list
load_datasets <- function(dataset_names, dir_data){
  # Look at files in data directory
  files <- dir(dir_data)
  # Trim to datasets
  files <- files[grepl("dataset.RData", files)]
  # Loop for dataset names given and load one by one, appending to a list.
  # dataset names don't have to match perfectly
  data_list <- list(NULL)
  if (length(files) > 0){ # Loop if any match was found
    for (name in dataset_names){
      if (any(grepl(name, files, ignore.case = TRUE))){ # Avoid loading bad matches
        load(file = paste0(dir_data, files[grep(name, files, ignore.case = TRUE)]))
        print(paste0("Loading file: ", paste0(dir_data, files[grep(name, files, ignore.case = TRUE)])))
        data_list[[name]]$counts <- data
        data_list[[name]]$metadata <- metadata
        data_list[[name]]$genes <- rownames(data)
        if (exists("exp_factors")){
          data_list[[name]]$exp_factors <- exp_factors
        }
      }
    }
    # Remove redundant first element
    data_list[[1]] <- NULL
  }
  
  return(data_list)
  
} # End function

### Load the all the data to be used and preprocess-----------------------------------------
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
Yh <- as.matrix(data_list[[ref_dataset]]$metadata  %>% 
                  dplyr::select(nas_score,Fibrosis_stage)) #keep both Fibrosis and NAS
colnames(Yh) <- c('NAS','fibrosis')
Xh <- data_list[[ref_dataset]]$data_center %>% t()
# Get matrices of target dataset
Xm <- data_list[[target_dataset]]$data_center %>% t()
# Get Wm as the PC space of the MPS data when averaging tech replicates to capture variance due to experimental factors
Wm <- data_list[[target_dataset]]$Wm_group %>% as.matrix()

### Load model results from MPS
Wh <- readRDS(paste0('../../FattyLiverModeling/results/Wh_',tolower(ref_dataset),'.rds'))
plsr_model <- readRDS(paste0('../../FattyLiverModeling/results/PLSR_model_',tolower(ref_dataset),'.rds'))
Bh <- t(plsr_model@weightMN) %*% plsr_model@coefficientMN
phi <- Wh %*% Bh
Wm_opt <- analytical_solution_opt(y=Yh,
                                  W_invitro = Wm,
                                  phi = phi)

### Load microarray hepatocytes data-------------------------------------
hepatocytes_metadata <- read.delim('../data/E-MTAB-798.sdrf.txt')
pheno <- hepatocytes_metadata %>%
  mutate(dose = ifelse(grepl('Low', Description), 'Low',
                       ifelse(grepl('Middle', Description), 'Middle',
                              ifelse(grepl('High', Description), 'High', 'Control')))) %>%
  mutate(condition = ifelse(grepl('Control', dose), 'control', 'treated')) %>%
  mutate(time = paste0(Factor.Value.time., ' ', Unit.time.unit.)) %>%
  dplyr::select(c('drug' = 'Factor.Value.cohort.'), condition, dose,
                c('dose_value'='Factor.Value.dose.'), time, Array.Data.File) %>%
  column_to_rownames('Array.Data.File')
pheno <- pheno %>% mutate(dose_value= ifelse(is.na(dose_value),0,dose_value))
data_replicates <- pheno %>% mutate(replicate_id = paste(drug,dose,time,sep = "+")) %>%
  dplyr::select(replicate_id) %>% rownames_to_column('samples')
onesample_exps <- data_replicates %>%  group_by(replicate_id) %>% mutate(nsample = n()) %>% ungroup()
onesample_exps <- onesample_exps %>% filter(nsample==1)
data_replicates <- data_replicates %>% filter(!(samples %in% onesample_exps$samples))
hepatocytes <- readRDS('preprocessed_data/gex_normalized_aggregated_microarray.rds')
hepatocytes <- hepatocytes[,which(colnames(hepatocytes) %in% data_replicates$samples)]
hepatocytes <- t(hepatocytes)
hepatocytes <- left_join(as.data.frame(hepatocytes) %>% rownames_to_column('samples'),
                         data_replicates %>% dplyr::select(replicate_id,samples))
hepatocytes <- aggregate(hepatocytes[,2:12646],by=list(hepatocytes$replicate_id),FUN=median)
hepatocytes <- as.matrix(hepatocytes %>% column_to_rownames('Group.1'))

hepatocytes_standarized <- hepatocytes - colMeans(hepatocytes)
hepatocytes_standarized <- hepatocytes_standarized[,which(colnames(hepatocytes_standarized) %in% colnames(Xh))]
addZeros <- matrix(0,
                   ncol = length(which(!(colnames(Xh) %in% colnames(hepatocytes_standarized)))),
                   nrow=nrow(hepatocytes))
rownames(addZeros) <- rownames(hepatocytes_standarized)
colnames(addZeros) <- colnames(Xh)[which(!(colnames(Xh) %in% colnames(hepatocytes_standarized)))]
hepatocytes_standarized <- cbind(hepatocytes_standarized,addZeros)
hepatocytes_standarized <- hepatocytes_standarized[,colnames(Xh)]

### Project MPS. hepatocytes and human samples in the extra LV space---------------
Zm <- as.data.frame(Xm %*% Wm_opt) %>% mutate(dataset='MPS')
Zh <- as.data.frame(Xh %*% Wm_opt) %>% mutate(dataset='human')
Zhepatocytes <- as.data.frame(hepatocytes_standarized %*% Wm_opt) %>% mutate(dataset='hepatocytes')
Z <- rbind(Zm,Zh,Zhepatocytes)
Z$dataset <- factor(Z$dataset,levels = c('MPS','hepatocytes','human'))
ggplot(Z,aes(x=V1,y=V2,fill = dataset))+
  geom_point(shape=21,size=2,alpha=0.5,colour='black')+
  scale_fill_viridis_d(option = "plasma") + 
  xlab('extra LV 1') + ylab('extra LV 2') +
  theme_pubr(base_size = 30,base_family = 'Arial')+
  theme(text = element_text(size=30,family = 'Arial'),
        plot.title = element_text(hjust = 0.5),
        axis.line = element_line(linewidth = 4),
        legend.position = 'right')
Yhepa <- predict(plsr_model,hepatocytes_standarized)
hist(Yhepa[,1],40)
hist(Yhepa[,2],40)
