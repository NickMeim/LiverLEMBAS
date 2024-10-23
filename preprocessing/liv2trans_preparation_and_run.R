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
dataset_names <- c("Govaere")
ref_dataset <- "Govaere"
# Load
data_list <- load_datasets(dataset_names, dir_data = '../../FattyLiverModeling/data/')
# Run PCA
tmp <- process_datasets(data_list, filter_variance = F)
data_list <- tmp$data_list
plt_list <- tmp$plt_list
# Define matrices of interest
Yh <- as.matrix(data_list[[ref_dataset]]$metadata  %>% dplyr::select(nas_score,Fibrosis_stage)) #keep both Fibrosis and NAS
colnames(Yh) <- c('NAS','fibrosis')
Xh <- data_list[[ref_dataset]]$data_center %>% t()

#### Load gex and save a replicate merged standarized GEX file to train a TF to Genes model--------------------
metadata <- read.delim('../data/E-MTAB-798.sdrf.txt')
pheno <- metadata %>%
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
TrimmedTFs <- read.delim('../preprocessing/preprocessed_data/TrainingValidationData/Hepatocytes_input_conditions.tsv',
                         sep = '\t') %>% column_to_rownames('X')
data_replicates <- data_replicates %>% filter(replicate_id %in% rownames(TrimmedTFs))
gex_rma <- readRDS('preprocessed_data/gex_normalized_aggregated_microarray.rds')
gex_rma <- gex_rma[,which(colnames(gex_rma) %in% data_replicates$samples)]
gex_rma <- t(gex_rma)
gex_rma <- left_join(as.data.frame(gex_rma) %>% rownames_to_column('samples'),data_replicates %>% dplyr::select(replicate_id,samples))
### Load also Wm to keep only genes present
WMPS <- readRDS('../../FattyLiverModeling/results/Wm_kostrzewski_total.rds')
genes_mps <- intersect(rownames(WMPS),colnames(gex_rma))
genes_govaere_hepatocytes <- intersect(colnames(Xh),colnames(gex_rma))
## aggregate and standardize Gex
gex_rma <- aggregate(gex_rma[,2:12646],by=list(gex_rma$replicate_id),FUN=median)
gex_rma <- as.matrix(gex_rma %>% column_to_rownames('Group.1'))
gex_rma <- gex_rma[rownames(TrimmedTFs),]
gex_rma_standar <- gex_rma - colMeans(gex_rma)

## save Gene expression for analysis with human data
## and for analysis with already trained MPS
# data.table::fwrite(as.data.frame(gex_rma_standar[,genes_govaere_hepatocytes]),
#         'preprocessed_data/GeneExpressionHepatocytes.csv',
#         row.names = TRUE,col.names = TRUE)
# data.table::fwrite(as.data.frame(gex_rma_standar[,genes_mps]),
#                    'preprocessed_data/GeneExpressionHepatocytesMPS.csv',
#                    row.names = TRUE,col.names = TRUE)
# data.table::fwrite(as.data.frame(Xh[,genes_govaere_hepatocytes]),
#                    'preprocessed_data/X_Govaere_hepatocytes.csv',
#                    row.names = TRUE,col.names = TRUE)

### Run LIV2TRANS approach-------------------
Xm <- gex_rma_standar[,genes_govaere_hepatocytes]
Xh <- Xh[,genes_govaere_hepatocytes]
pca <- prcomp(Xm,center = TRUE,scale. = FALSE)
Wm <- pca$rotation
Wm <- Wm[,-ncol(Wm)]
gc()

### Run cross validation
# liv2trans_CV_results <- liv2trans_run_CV(Xh,Yh,Xm,Wm,num_folds = 5,LV_number = 8)
# ggboxplot(liv2trans_CV_results,x='input',y='r',color='input',add='jitter')+
#   scale_y_continuous(n.breaks = 10)+
#   facet_wrap(~set)+
#   stat_compare_means(comparisons = list(c('human features truncated via the original in vitro model',
#                                           'human features truncated via the optimized in vitro model'),
#                                         c('human features truncated via the optimized in vitro model',
#                                           'shuffled features')),
#                      method = 'wilcox',
#                      label.y = c(0.85,0.85))+
#   theme(text=element_text(family = 'Arial',size=20),
#         legend.position = 'bottom',
#         axis.text.x = element_blank(),
#         panel.grid.major = element_line())
# ggsave('../results/hepatocytes_liv2trans_cv.png',
#        width = 18,
#        height = 9,
#        units = 'in',
#        dpi = 600)

### Run main function with all the data
liv2trans_results <- liv2trans_run(Xh,Yh,Xm,Wm,LV_number = 8)
W_translatable <- liv2trans_results$W_translatable
W_opt <- liv2trans_results$W_opt
W_tot <- liv2trans_results$W_tot

### Get coefficients of PLSR model and the most important genes
plsr_model <- liv2trans_results$model
# Get Wh of PLSR
Wh <- matrix(data = 0, ncol = ncol(plsr_model@weightMN), nrow = ncol(Xh))
rownames(Wh) <- colnames(Xh)
colnames(Wh) <- colnames(plsr_model@weightMN)
for (ii in 1:nrow(plsr_model@weightMN)){
  Wh[rownames(plsr_model@weightMN)[ii], ] <- plsr_model@weightMN[ii,]
}
# Get regression coefficients
Bh <- t(plsr_model@weightMN) %*% plsr_model@coefficientMN
phi <- Wh %*% Bh

## visualize projected human samples
Zh <- Xh %*% W_opt
Zh <- cbind(Zh,Yh)
Zh <- as.data.frame(Zh) %>% gather('phenotype','Score',-V1,-V2)
colnames(Zh)[1:2] <- c('V1','V2')
ggplot(Zh %>% mutate(phenotype=ifelse(phenotype=='fibrosis','Fibrosis stage','MAS')) %>%
         group_by(phenotype) %>% mutate(normed_score=Score/max(Score)),
       aes(x=V1,y=V2,color=normed_score)) +
  geom_point(size=5)+
  scale_color_viridis_c()+
  labs(color = 'Score')+
  xlab('extra LV 1') + ylab('extra LV 2') +
  theme_pubr(base_size = 30,base_family = 'Arial')+
  theme(text = element_text(size=30,family = 'Arial'),
        plot.title = element_text(hjust = 0.5),
        axis.line = element_line(linewidth = 4),
        legend.position = 'right')+
  facet_wrap(~phenotype)

## visualize projected hepatocytes samples
Zm <- Xm %*% W_opt
Zm <- as.data.frame(Zm)
colnames(Zm)[1:2] <- c('V1','V2')
Zm <- Zm %>% mutate(data = 'hepatocytes')
Zm <- rbind(Zm,Zh %>% dplyr::select(V1,V2) %>% unique() %>% mutate(data='human'))
ggplot(Zm,aes(x=V1,y=V2,color=data)) +
  geom_point(size=3)+
  scale_color_viridis_d()+
  xlab('extra LV 1') + ylab('extra LV 2') +
  theme_pubr(base_size = 30,base_family = 'Arial')+
  theme(text = element_text(size=30,family = 'Arial'),
        plot.title = element_text(hjust = 0.5),
        axis.line = element_line(linewidth = 4))
### Visualize pathway activity-------------------------
pathway_interpretation_results <- pathway_activity_interpretation(W_opt,Wm)
ggsave('../results/liv2trans_extra1_paths.png',
       plot = pathway_interpretation_results$plots[[1]],
       dpi = 600)
ggsave('../results/liv2trans_extra2_paths.png',
       plot = pathway_interpretation_results$plots[[2]],
       dpi = 600)

### Visualize pathway activity for existing translatable components TCs
pathway_interpretation_TCs_results <- pathway_activity_interpretation(W_translatable,Wm,lim = 8)
ggsave('../results/liv2trans_TC1_paths.png',
       plot = pathway_interpretation_TCs_results$plots[[1]] + ggtitle('TC 1'),
       dpi = 600)
ggsave('../results/liv2trans_TC2_paths.png',
       plot = pathway_interpretation_TCs_results$plots[[2]]+ ggtitle('TC 2'),
       dpi = 600)
# png('hepg2_LVextra1.png',width = 9,height = 6,units = 'in',res = 300)
# print(pathway_interpretation_results$plots[[1]])
# dev.off()

### Visualize Hallmark genesets enrichment-------------------------
entrez_ids <- mapIds(org.Hs.eg.db, keys = rownames(W_opt), column = "ENTREZID", keytype = "SYMBOL")
entrez_ids <- unname(entrez_ids)
inds <- which(!is.na(entrez_ids))
entrez_ids <- entrez_ids[inds]
meas <- as.matrix(W_opt[inds,])
rownames(meas) <- entrez_ids
hallmarks_interpretation_results <- HallmarksFastenrichment(signature_ids = colnames(W_opt),
                                                            gene_ids = rownames(meas),
                                                            measurements = meas,
                                                            axis_lab = 'LV extra',
                                                            n_permutations=10000)
ggsave('../results/liv2trans_extra1_hallmarks.png',
       plot = hallmarks_interpretation_results$plots[[1]],
       dpi = 600)
ggsave('../results/liv2trans_extra2_hallmakrs.png',
       plot = hallmarks_interpretation_results$plots[[2]],
       dpi = 600)
### Visualize Hallmark genesets enrichment for existing translatable components TCs
entrez_ids <- mapIds(org.Hs.eg.db, keys = rownames(W_translatable), column = "ENTREZID", keytype = "SYMBOL")
entrez_ids <- unname(entrez_ids)
inds <- which(!is.na(entrez_ids))
entrez_ids <- entrez_ids[inds]
meas <- as.matrix(W_translatable[inds,])
rownames(meas) <- entrez_ids
hallmarks_interpretation_TCs_results <- HallmarksFastenrichment(signature_ids = colnames(W_translatable),
                                                                gene_ids = rownames(meas),
                                                                measurements = meas,
                                                                axis_lab = 'TC',
                                                                n_permutations=10000)
ggsave('../results/liv2trans_TC1_hallmarks.png',
       plot = hallmarks_interpretation_TCs_results$plots[[1]]+ ggtitle('TC 1'),
       dpi = 600)
ggsave('../results/liv2trans_TC2_hallmakrs.png',
       plot = hallmarks_interpretation_TCs_results$plots[[2]]+ ggtitle('TC 2'),
       dpi = 600)

### Get top 100 genes in TCs and extras and PLSR weights---------------------------
gextra <- unique(c(rownames(W_opt)[order(-abs(W_opt[,1]))[1:100]],
                   rownames(W_opt)[order(-abs(W_opt[,2]))[1:100]]))
gTC <- unique(c(rownames(W_translatable)[order(-abs(W_translatable[,1]))[1:100]],
                rownames(W_translatable)[order(-abs(W_translatable[,2]))[1:100]]))
gPLSR <- unique(c(rownames(phi)[order(-abs(phi[,1]))[1:100]],
                  rownames(phi)[order(-abs(phi[,2]))[1:100]]))
important_genes <- unique(c(gextra,gTC,gPLSR))
saveRDS(important_genes,'preprocessed_data/Hepatocytes_human_important_genes.rds')

# dorotheaData = read.table('../data/dorothea.tsv', sep = "\t", header=TRUE)
# confidenceFilter = is.element(dorotheaData$confidence, c('A', 'B'))
# dorotheaData = dorotheaData[confidenceFilter,]
