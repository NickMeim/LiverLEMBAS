# Load requeired packages
library(dorothea)
library(tidyverse)
library(doFuture)
# parallel: set number of workers
cores <- 15
registerDoFuture()
plan(multisession,workers = cores)

# Access command-line arguments
# args <- commandArgs(trailingOnly = TRUE)

# inputFile <- args[1]
# outputFile = args[2]
inputFile <- 'preprocessed_data/TF_activities/ligands_l1000_allgenes_lvl3_tfs.rds'
outputFile = "preprocessed_data/TF_activities/l1000_allgenes_lvl3_tfs.tsv"
# Read inferred TF activities
TF_activities <- readRDS(inputFile)

### First filter out TFS with consistently high variance across replicates in the samples 
### Then filter samples with low correlation between replicates.---------------
sigInfo <- read.delim('../../../L1000_2021_11_23/siginfo_beta.txt')
sigInfo <- sigInfo %>% 
  mutate(quality_replicates = ifelse(is_exemplar_sig==1 & qc_pass==1,1,0))
# Filter only experiments of drugs
sigInfo <- sigInfo %>% filter(pert_type=='trt_lig')
sigInfo <- sigInfo %>% filter(quality_replicates==1)
# Create identifier to signify duplicate
# signatures: meaning same drug, same dose,
# same time duration, same cell-type
sigInfo <- sigInfo %>% 
  mutate(duplIdentifier = paste0(cmap_name,"_",pert_idose,"_",pert_itime,"_",cell_iname))
sigInfo <- sigInfo %>% group_by(duplIdentifier) %>%
  mutate(dupl_counts = n()) %>% ungroup()
sigInfo <- sigInfo %>% filter(cell_iname=='HEPG2')
sigInfo <- sigInfo %>% filter(pert_itime=='2 h')
sigInfo <- sigInfo %>% mutate(inst_id=strsplit(distil_ids,"\\|")) %>% unnest(inst_id)
gc()

data_replicates <- sigInfo %>% filter(nsample>=2) %>% select(sig_id,inst_id) %>% unique()
data_replicates <- left_join(data_replicates,as.data.frame(TF_activities) %>% rownames_to_column('inst_id'))

# Filter high variance TFs---------------------------
TFs_vars <- aggregate(data_replicates[,3:ncol(data_replicates)],by=list(data_replicates$sig_id),FUN=var)
### Perform random permutations per TF and 
# determine with a KS-test if the distribution of random variances is different
randomVariance <- function(data,ind){
  RandomVar <- aggregate(data[sample(1:nrow(data)),2+ind],
                         by=list(data$sig_id),
                         FUN=var)
  return(RandomVar[,2:ncol(RandomVar)])
}
TFs_random_all <- data.frame()
for (i in 1:100){
  TFs_random <- foreach(j = seq(1,ncol(TF_activities))) %dopar% {
    randomVariance(data_replicates ,
                   ind = j)
  }
  TFs_random <-do.call(cbind,TFs_random)
  TFs_random_all <- rbind(TFs_random_all,TFs_random)
  if (i%%100==0){
    print(paste0('Finished iteration: ',i))
  }
}
saveRDS(TFs_random_all,'preprocessed_data/TF_activities/TFs_random_variances_all.rds')

filter_tfs <- NULL
ks_tests <- NULL
difference_means <- NULL
k <- 1
for (j in 1:ncol(TFs_random_all)){
  test <- ks.test(TFs_vars[,j+1],TFs_random_all[,j],alternative = "g") # test if vars are stochastically less than random
  ks_tests[[j]] <- test 
  difference_means[j] <- mean(TFs_vars[,j+1])- mean(TFs_random_all[,j])
  if (test$p.value>0.05){
    filter_tfs[k] <- colnames(TFs_vars)[j] 
  }
}
saveRDS(filter_tfs,'preprocessed_data/TF_activities/filtered_tfs_from_repls.rds')
saveRDS(difference_means,'preprocessed_data/TF_activities/difference_means_of_variances.rds')
saveRDS(ks_tests,'preprocessed_data/TF_activities/ks_tests_repls.rds')
print(paste0('Filtered TFs:',filter_tfs))
hist(difference_means,main = 'Mean variace of a TF - mean variance of a TF in random data')

TF_activities <- TF_activities[,which(!(colnames(TF_activities) %in% filter_tfs))]

### Filter samples----------------
replicatesCor <- function(data,s){
  x <- data %>% filter(sig_id==s)
  x <- t(x[,3:ncol(data)])
  c <- cor(x)
  c <- c[lower.tri(c, diag = FALSE)]
  return(mean(c))
}
sigs <- unique(data_replicates$sig_id)

TFs_cor <- foreach(sig = sigs) %dopar% {
  replicatesCor(data_replicates ,
                s = sig)
}

TFs_cor <-do.call(rbind,TFs_cor)
TFs_cor <- as.data.frame(TFs_cor)
TFs_cor <- TFs_cor %>% mutate(sig_id=sigs)
TFs_cor <- TFs_cor %>% select(sig_id,c('corr'='V1'))
TFs_cor <- left_join(TFs_cor,sigInfo %>% select(sig_id,nsample) %>% unique())

rep_size <- sigInfo %>% filter(nsample>=2)
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
  TFs_cor$p.value[i] <- sum(random_corrs[which(rownames(random_corrs)==TFs_cor$distil_nsample[i]),]>=TFs_cor$corr[i])/1000
  if (i %% 1000 ==0){
    print(paste0('Finished ',i))
  }
}

TFs_cor <- TFs_cor %>% filter(p.value<0.05)

### Merge replicates------------------
data_merged <- sigInfo %>% filter(sig_id %in% TFs_cor$sig_id | nsample==1) %>% select(-nsample) 

TF_activities <- as.data.frame(TF_activities) %>% rownames_to_column('inst_id')
TF_activities <- TF_activities %>% filter(inst_id %in% data_merged$inst_id)
TF_activities <- left_join(TF_activities,data_merged %>% select(sig_id,inst_id))
TF_activities <- TF_activities %>% select(-inst_id)
ind <- which(colnames(TF_activities)=='sig_id')
TF_activities <- aggregate(TF_activities[,1:114],by=list(TF_activities$sig_id),FUN=median)
TF_activities <- TF_activities %>% column_to_rownames('Group.1')
TF_activities <- as.matrix(TF_activities)
hist(TF_activities)
write.table(TF_activities, file = outputFile, quote=FALSE, sep = "\t", row.names = TRUE, col.names = NA)

### Make also ligand receptor file in long format----------------------------
TF_activities <- read.delim(outputFile,sep = "\t")
sigInfo <- read.delim('../../../L1000_2021_11_23/siginfo_beta.txt')
sigInfo <- sigInfo %>% filter(sig_id %in% TF_activities$X)
sigInfo <- sigInfo %>% 
mutate(cmap_name = ifelse(cmap_name=='IL12','IL12A',cmap_name)) %>%
  mutate(cmap_name = ifelse(cmap_name=='IL17','IL17A',cmap_name)) %>%
  mutate(cmap_name = ifelse(cmap_name=='IL17E','IL25',cmap_name)) %>%
  mutate(cmap_name = ifelse(cmap_name=='IL8','CXCL8',cmap_name)) %>%
  mutate(cmap_name = ifelse(cmap_name=='BMP11','GDF11',cmap_name)) %>%
  mutate(cmap_name = ifelse(cmap_name=='BMP13','GDF6',cmap_name)) %>%
  mutate(cmap_name = ifelse(cmap_name=='CCL3L1','CCL3L3',cmap_name))%>%
  mutate(cmap_name = ifelse(cmap_name=='FIGF','VEGFD',cmap_name)) %>%
  mutate(cmap_name = ifelse(cmap_name=='IFNA1','IFNA13',cmap_name)) %>%
  mutate(cmap_name = ifelse(cmap_name=='IL28A','IFNL2',cmap_name)) %>%
  mutate(cmap_name = ifelse(cmap_name=='IL29','IFNL1',cmap_name)) %>%
  mutate(cmap_name = ifelse(cmap_name=='KNG','KNG1',cmap_name))%>%
  mutate(cmap_name = ifelse(cmap_name=='LGALS7','LGALS7; LGALS7B',cmap_name)) %>%
  mutate(cmap_name = ifelse(cmap_name=='WISP2','CCN5',cmap_name)) %>%
  mutate(cmap_name = ifelse(cmap_name=='WISP3','CCN6 ',cmap_name)) %>%
  unique()
gc()
annot <- read.csv('../data/uniprot-reviewed_yes+AND+organism__Homo+sapiens+(Human)+[9606]_.tab', sep='\t')
annot <- annot %>% dplyr::select(c('ligand'='Gene.names...primary..'),c('source'='Entry')) %>% unique()
omniPath_RLs <- OmnipathR::curated_ligand_receptor_interactions()
omniPath_RLs <- rbind(omniPath_RLs %>% select(c('ligand'='source_genesymbol'),source),
                      omniPath_RLs %>% select(c('ligand'='target_genesymbol'),c('source'='target'))) %>%
  unique()
LRI <- rbind(left_join(sigInfo %>% select(c('ligand'='cmap_name')) %>% unique(),
                 omniPath_RLs) %>% unique() %>% filter(!is.na(source)),
             left_join(sigInfo %>% select(c('ligand'='cmap_name')) %>% unique(),
                       annot) %>% unique() %>% filter(!is.na(source))) %>%
  unique()
RLfull <- read.delim('preprocessed_data/PKN/RLFull.tsv',sep='\t') %>%
  select(source,target,direction,stimulation,inhibition) %>% unique()
omniPath_RLs <- OmnipathR::curated_ligand_receptor_interactions() %>% filter(consensus_direction==1) %>%
  select(source,target,c('direction'='consensus_direction'),
         c('stimulation'='is_stimulation'),c('inhibition'='is_inhibition')) %>% unique()
RLfull <- rbind(RLfull,omniPath_RLs) %>% unique()
LRI <- left_join(LRI,RLfull) 
LRI <- LRI %>% filter(!is.na(target)) %>% unique()
write.table(LRI %>% select(-ligand) %>% unique(),
            file ='preprocessed_data/PKN/RL_L1000.tsv' , 
            quote=FALSE, sep = "\t", row.names = FALSE)
sigInfo_filt <- sigInfo %>% filter(cmap_name %in% LRI$ligand)
TF_activities <- as.matrix(TF_activities %>% column_to_rownames('X'))
TF_activities <- TF_activities[sigInfo_filt$sig_id,]
# ### Compare 2h vs 4h of same ligand+dose---------------
# groups_row <- sigInfo %>% select(-inst_id) %>% unique() %>% 
#   column_to_rownames('sig_id') %>% select(pert_itime)
# x <- TF_activities[rownames(groups_row),]
# print(all(rownames(groups_row)==rownames(x)))
# pheatmap::pheatmap(x, 
#                    color = colorRampPalette(c("blue", "white", "red"))(100), 
#                    breaks = seq(0, 1, length.out = 100),
#                    main = "Clustered TF activities",
#                    annotation_row = groups_row,
#                    fontsize = 18,
#                    filename = 'clustering_of_timepoints.png',
#                    show_colnames = FALSE,  
#                    show_rownames = FALSE,
#                    height = 12,
#                    width = 14,
#                    units = 'in',
#                    res=600
# )
# 
# x <- x %>% rownames_to_column('sig_id') %>% gather('TF','activity',-sig_id)
# sig2 <- sigInfo %>% filter(pert_itime=='2 h') %>% select(sig_id,cmap_name,pert_idose) %>%
#   mutate(condition = paste0(cmap_name,"+",pert_idose))
# sig4 <- sigInfo %>% filter(pert_itime=='4 h')%>% select(sig_id,cmap_name,pert_idose)%>%
#   mutate(condition = paste0(cmap_name,"+",pert_idose))
# sig2 <- sig2 %>% filter(cmap_name %in% sig4$cmap_name) %>% filter(condition %in% sig4$condition)
# sig4 <- sig4 %>% filter(cmap_name %in% sig2$cmap_name) %>% filter(condition %in% sig2$condition)
# compared <- left_join(sig2,sig4,by=c("cmap_name","pert_idose","condition"))
# compared <- compared %>% unique()
# compared_x <- left_join(compared,x %>% select(c('sig_id.x'='sig_id'),c('TF.x'='TF'),c('activity.x'='activity')))
# compared_x <- left_join(compared_x,x %>% select(c('sig_id.y'='sig_id'),c('TF.y'='TF'),c('activity.y'='activity')))
# compared_x <- compared_x %>% filter(TF.x==TF.y)
# ggpubr::ggscatter(compared_x,x='activity.x',y='activity.y',cor.coef = TRUE)
