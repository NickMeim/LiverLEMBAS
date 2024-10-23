library(dorothea)
library(tidyverse)
library(hgu133a.db)
library(AnnotationDbi)
library(affy)
# library(ArrayExpress)
# library(limma)
library(doFuture)

# parallel set number of workers
cores <- 15
registerDoFuture()
plan(multisession,workers = cores)

### Load raw data and metadata---------------------------------
metadata <- read.delim('../data/E-MTAB-798.sdrf.txt')
# metadata <- metadata %>% filter(Factor.Value.time.==timepoint)
available_files <- list.files('../../../../Downloads/E-MAT-790/raw_data',pattern = '.CEL')
metadata <- metadata %>% filter(Array.Data.File %in% available_files)


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
# Load the raw CEL files corresponding to the drug
raw <- ReadAffy(celfile.path = "../../../../Downloads/E-MAT-790/raw_data",
                filenames = rownames(pheno),
                phenoData = pheno)

# Normalize the data using RMA------------------------------------------------
exprs_rma <- affy::rma(raw, normalize = TRUE, background = TRUE)

# get the normalized gene expression and aggregate probes into genes
gex_rma <- exprs(exprs_rma)
# gex_rma <- gset[["GSE31534_series_matrix.txt.gz"]]@assayData[["exprs"]]
anno <- AnnotationDbi::select(hgu133a.db,
                              keys = (rownames(gex_rma)),
                              columns = c("SYMBOL"),
                              keytype = "PROBEID")
anno <- subset(anno , !is.na(SYMBOL))
annog <- group_by(anno, PROBEID)
anno_summarized <-
  dplyr::summarize(annog, no_of_matches = n_distinct(SYMBOL))
anno_filtered <- filter(anno_summarized, no_of_matches == 1)
ids <- (rownames(gex_rma) %in% anno_filtered$PROBEID)
gex_rma <- subset(gex_rma,ids)
gex_rma <- as.data.frame(gex_rma) %>%
  rownames_to_column("PROBEID")
gex_rma <- left_join(gex_rma,anno)
gex_rma <- gex_rma  %>% dplyr::select(-PROBEID)
gex_rma <- aggregate(gex_rma[,1:(ncol(gex_rma)-1)],by=list(gex_rma$SYMBOL),FUN=median)
gex_rma <- gex_rma %>% column_to_rownames('Group.1')
# saveRDS(gex_rma,'preprocessed_data/gex_normalized_aggregated_microarray.rds')

# Calculate TF activity------------------------------------------------------------
dorotheaData = read.table('../data/dorothea.tsv', sep = "\t", header=TRUE)
confidenceFilter = is.element(dorotheaData$confidence, c('A', 'B'))
dorotheaData = dorotheaData[confidenceFilter,]

#Standarize gene expression
gex_standarized <- scale(t(gex_rma),scale = F)
hist(gex_standarized)

# Estimate TF activities
settings = list(verbose = TRUE, minsize = 5)
TF_activities = run_viper(t(gex_standarized), dorotheaData, options =  settings)

# Save results
annot <- read.csv('../data/uniprot-reviewed_yes+AND+organism__Homo+sapiens+(Human)+[9606]_.tab', sep='\t')
annot <- annot %>% dplyr::select(Gene.names...primary..,Entry)
tfs <- data.frame("Gene.names...primary.."=rownames(TF_activities))
tfs <- left_join(tfs,annot)
print(all(rownames(TF_activities)==tfs$Gene.names...primary..))
rownames(TF_activities) <- tfs$Entry
TF_activities <- t(TF_activities)
hist(1/(1+exp(-TF_activities)),main='Inferred TF activities',xlab='activity')
TF_activities <- 1/(1+exp(-TF_activities))
# saveRDS(TF_activities, 'preprocessed_data/TF_activities/TF_acitivities_hepatocytes_microarray.rds')

# group_by(replicate_id) %>% mutate(nsample = n()) %>% ungroup()
data_replicates <- pheno %>% mutate(replicate_id = paste(drug,dose,time,sep = "+")) %>%
  dplyr::select(replicate_id) %>% rownames_to_column('samples')
onesample_exps <- data_replicates %>%  group_by(replicate_id) %>% mutate(nsample = n()) %>% ungroup()
onesample_exps <- onesample_exps %>% filter(nsample==1)
data_replicates <- data_replicates %>% filter(!(samples %in% onesample_exps$samples))
data_replicates <- left_join(data_replicates,as.data.frame(TF_activities) %>% rownames_to_column('samples'))

# # Filter high variance TFs---------------------------
# TFs_vars <- aggregate(data_replicates[,3:ncol(data_replicates)],by=list(data_replicates$replicate_id),FUN=var)
# ### Perform random permutations per TF and 
# # determine with a KS-test if the distribution of random variances is different
# randomVariance <- function(data,ind){
#   RandomVar <- aggregate(data[sample(1:nrow(data)),2+ind],
#                          by=list(data$replicate_id),
#                          FUN=var)
#   return(RandomVar[,2:ncol(RandomVar)])
# }
# TFs_random_all <- data.frame()
# for (i in 1:100){
#   TFs_random <- foreach(j = seq(1,ncol(TF_activities))) %dopar% {
#     randomVariance(data_replicates ,
#                    ind = j)
#   }
#   TFs_random <-do.call(cbind,TFs_random)
#   TFs_random_all <- rbind(TFs_random_all,TFs_random)
#   if (i%%10==0){
#     print(paste0('Finished iteration: ',i))
#   }
# }
# saveRDS(TFs_random_all,'preprocessed_data/TF_activities/TFs_random_variances_all_microarray.rds')
# 
# filter_tfs <- NULL
# ks_tests <- NULL
# difference_means <- NULL
# k <- 1
# for (j in 1:ncol(TFs_random_all)){
#   test <- ks.test(TFs_vars[,j+1],TFs_random_all[,j],alternative = "g") # test if vars are stochastically less than random
#   ks_tests[[j]] <- test 
#   difference_means[j] <- mean(TFs_vars[,j+1])- mean(TFs_random_all[,j])
#   if (test$p.value>0.05){
#     filter_tfs[k] <- colnames(TFs_vars)[j] 
#   }
# }
# saveRDS(filter_tfs,'preprocessed_data/TF_activities/filtered_tfs_from_repls_microarray.rds')
# saveRDS(difference_means,'preprocessed_data/TF_activities/difference_means_of_variances_microarray.rds')
# saveRDS(ks_tests,'preprocessed_data/TF_activities/ks_tests_repls_microarray.rds')
# print(paste0('Filtered TFs:',filter_tfs))
# hist(difference_means,main = 'Mean variace of a TF - mean variance of a TF in random data')
# 
# TF_activities <- TF_activities[,which(!(colnames(TF_activities) %in% filter_tfs))]

### Filter samples----------------
replicatesCor <- function(data,s){
  x <- data %>% filter(replicate_id==s)
  x <- t(x[,3:ncol(data)])
  c <- cor(x)
  c <- c[lower.tri(c, diag = FALSE)]
  return(mean(c))
}
ids <- unique(data_replicates$replicate_id)

TFs_cor <- foreach(id = ids) %dopar% {
  replicatesCor(data_replicates ,
                s = id)
}

TFs_cor <-do.call(rbind,TFs_cor)
TFs_cor <- as.data.frame(TFs_cor)
TFs_cor <- TFs_cor %>% mutate(replicate_id=ids)
TFs_cor <- TFs_cor %>% dplyr::select(replicate_id,c('corr'='V1'))
TFs_cor <- left_join(TFs_cor,
                     pheno %>% mutate(replicate_id = paste(drug,dose,time,sep = "+")) %>%  
                       group_by(replicate_id) %>% mutate(nsample = n()) %>% ungroup() %>%
                       dplyr::select(replicate_id,nsample) %>% unique())

rep_size <- pheno %>% mutate(replicate_id = paste(drug,dose,time,sep = "+")) %>%  
  group_by(replicate_id) %>% mutate(nsample = n()) %>% ungroup() %>%
  dplyr::select(replicate_id,nsample) %>% unique() %>% filter(nsample>=2)
rep_size <- unique(rep_size$nsample)
rep_size <- rep_size[order(rep_size)]
random_corrs <- matrix(0,nrow=length(rep_size),ncol=1000)
for (i in 1:length(rep_size)){
  n <- rep_size[i]
  for (j in 1:1000){
    tfs <- data_replicates[sample(1:nrow(data_replicates),n),3:ncol(data_replicates)]
    c <- cor(t(tfs))
    c <- c[lower.tri(c, diag = FALSE)]
    random_corrs[i,j] <- mean(c)
  }
  print(paste0('Finished n: ',n))
}
rownames(random_corrs) <- rep_size


TFs_cor$p.value <- 1
for (i in 1:nrow(TFs_cor)){
  TFs_cor$p.value[i] <- sum(random_corrs[which(rownames(random_corrs)==TFs_cor$nsample[i]),]>=TFs_cor$corr[i])/1000
  if (i %% 1000 ==0){
    print(paste0('Finished ',i))
  }
}

TFs_cor <- TFs_cor %>% filter(p.value<0.05)

### Merge replicates------------------
data_merged <- pheno %>% mutate(replicate_id = paste(drug,dose,time,sep = "+")) %>%  
  filter((replicate_id %in% TFs_cor$replicate_id) | (replicate_id %in% onesample_exps$replicate_id)) %>%
  rownames_to_column('samples')

TF_activities <- as.data.frame(TF_activities) %>% rownames_to_column('samples')
TF_activities <- TF_activities %>% filter(samples %in% data_merged$samples)
TF_activities <- left_join(TF_activities,data_merged %>% dplyr::select(replicate_id,samples))
TF_activities <- TF_activities %>% dplyr::select(-samples)
ind <- which(colnames(TF_activities)=='replicate_id')
TF_activities <- aggregate(TF_activities[,1:118],by=list(TF_activities$replicate_id),FUN=median)
TF_activities <- TF_activities %>% column_to_rownames('Group.1')
TF_activities <- as.matrix(TF_activities)
hist(TF_activities)
write.table(TF_activities,
            file ='preprocessed_data/TF_activities/TF_acitivities_hepatocytes_microarray.tsv' , 
            quote=FALSE, sep = "\t", row.names = TRUE, col.names = NA)
