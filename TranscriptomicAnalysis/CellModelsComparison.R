library(tidyverse)
library(cmapR)
library(dorothea)
library(doFuture)

# parallel set number of workers
cores <- 15
registerDoFuture()
plan(multisession,workers = cores)

### Load my PHHs data----------------------
PHH_metadata <- read.table('../preprocessing/preprocessed_data/TrainingValidationData/Hepatocytes_inputdataInfo.tsv',
                           sep = '\t',header = TRUE)
phh_drugs <- unique(PHH_metadata$canonical_smiles)

## Load pre-processed L1000 data (landmark genes only)------

# Check L1000 documentation for information.
geneInfo <- read.delim('../../../L1000_2021_11_23/geneinfo_beta.txt')
# Keep only measured best inferred genes
geneInfo <-  geneInfo %>% filter(feature_space != "inferred")
# geneInfo <-  geneInfo %>% filter(feature_space =="landmark" | (gene_symbol %in% c('STAT2','TP63','TGFBR1')))
# Keep only protein-coding genes
geneInfo <- geneInfo %>% filter(gene_type=="protein-coding")

#Load signature info and split data to high quality replicates and low quality replicates
sigInfo <- read.delim('../../../L1000_2021_11_23/siginfo_beta.txt')
sigInfo <- sigInfo %>% 
  mutate(quality_replicates = ifelse(is_exemplar_sig==1 & qc_pass==1 & nsample>=3,1,0))
# Filter drugs
sigInfo <- sigInfo %>% filter(pert_type=='trt_cp')
sigInfo <- sigInfo %>% filter(quality_replicates==1)
sigInfo <- sigInfo %>% filter(cell_iname=='HEPG2')
drugInfo <- read.delim('../../../L1000_2021_11_23/compoundinfo_beta.txt')
drugInfo <- drugInfo %>% filter(pert_id %in% sigInfo$pert_id)
# cellInfo <- read.delim('../../../L1000_2021_11_23/cellinfo_beta.txt')
common_drugs <- phh_drugs[which(phh_drugs %in% drugInfo$canonical_smiles)]

## Load for L1000
parse_gctx_parallel <- function(path ,rid,cid){
  gctx_file <- parse_gctx(path ,rid = rid,cid = cid)
  return(gctx_file@mat)
}
# Path to raw data
ds_path <- '../../../L1000_2021_11_23/level5_beta_trt_cp_n720216x12328.gctx'

sigIds <- sigInfo$sig_id
sigList <-  split(sigIds, 
                  ceiling(seq_along(sigIds)/ceiling(length(sigIds)/cores)))
# Parse the data file in parallel
cmap_gctx <- foreach(sigs = sigList) %dopar% {
  parse_gctx_parallel(ds_path ,
                      rid = unique(as.character(geneInfo$gene_id)),
                      cid = sigs)
}
gc()
L1000_gene_expression <-do.call(cbind,cmap_gctx)
L1000_gene_expression <- left_join(as.data.frame(L1000_gene_expression) %>% rownames_to_column('gene_id'),
                                   geneInfo %>% select(gene_id,gene_symbol) %>% unique() %>%
                                     mutate(gene_id = as.character(gene_id)))
gex_standarized_l1000 <- L1000_gene_expression %>% column_to_rownames('gene_symbol') %>%
  select(-gene_id)
gex_standarized_l1000 <- gex_standarized_l1000 - rowMeans(gex_standarized_l1000)

## Keep from the L1000 only what exists in the PHHs
drugInfo <- drugInfo %>% filter(canonical_smiles %in% common_drugs)
sigInfo <- sigInfo %>% filter(pert_id %in% drugInfo$pert_id)
sigInfo <- left_join(sigInfo,drugInfo %>% select(pert_id,canonical_smiles) %>% unique())
                           

### Keep common in PHHs as well
PHH_metadata_filt <- PHH_metadata %>% filter(canonical_smiles %in% common_drugs)
dose_time_map <- left_join(sigInfo %>% select(canonical_smiles,pert_dose,pert_time),
                           PHH_metadata_filt %>% separate('time',into=c('time','unitTime')))
dose_time_map <- dose_time_map %>% mutate(time = as.numeric(time))
dose_time_map <- dose_time_map %>% mutate(abs_dose_diff = abs(dose-pert_dose))
dose_time_map <- dose_time_map %>% mutate(abs_time_diff = abs(time-pert_time))
dose_time_map <- dose_time_map %>% group_by(drug) %>%
  mutate(min_dose_diff = min(abs_dose_diff)) %>%
  ungroup() %>%
  filter(abs_dose_diff==min_dose_diff)
dose_time_map <- dose_time_map %>% group_by(drug) %>%
  mutate(min_time_diff = min(abs_time_diff)) %>%
  ungroup() %>%
  filter(abs_time_diff==min_time_diff)
dose_time_map <- dose_time_map %>% filter(abs_dose_diff<=20)
PHH_metadata_filt <- PHH_metadata_filt %>% filter(X %in% dose_time_map$X)
sigInfo_filt <- sigInfo %>% filter(canonical_smiles %in% dose_time_map$canonical_smiles)

### Load pre-processed gene expression--------------------

# L1000_gene_expression <- parse_gctx_parallel(ds_path ,
#                             rid = unique(as.character(geneInfo$gene_id)),
#                             cid = sigInfo$sig_id)
# L1000_gene_expression <- left_join(as.data.frame(L1000_gene_expression) %>% rownames_to_column('gene_id'),
#                   geneInfo %>% select(gene_id,gene_symbol) %>% unique() %>%
#                     mutate(gene_id = as.character(gene_id)))

## Infer TF activity
dorotheaData = read.table('../data/dorothea.tsv', sep = "\t", header=TRUE)
confidenceFilter = is.element(dorotheaData$confidence, c('A', 'B'))
dorotheaData = dorotheaData[confidenceFilter,]
# gex_standarized_l1000 <- L1000_gene_expression %>% column_to_rownames('gene_symbol') %>%
#   select(-gene_id)
# gex_standarized_l1000 <- gex_standarized_l1000 - rowMeans(gex_standarized_l1000)
# gex_standarized_l1000 <- gex_standarized_l1000
# Estimate TF activities
settings = list(verbose = TRUE, minsize = 5)
TF_activities_L1000 = run_viper(gex_standarized_l1000, dorotheaData, options =  settings)
TF_activities_L1000 <- 1/(1+exp(-TF_activities_L1000))
TF_activities_L1000 <- as.data.frame(TF_activities_L1000) %>% rownames_to_column('TF')
TF_activities_L1000 <- TF_activities_L1000 %>% gather('sig_id','activity_L1000',-TF)
# tt <- TF_activities_L1000 %>% filter(TF %in% c('STAT1','STAT2','STAT3','SMAD2','SMAD3','TP53','TP63'))
tt <- TF_activities_L1000 %>% filter(TF %in% c('STAT1','STAT2','STAT3','SMAD2','SMAD3','TP53','TP63')) %>%
  filter(sig_id %in% sigInfo_filt$sig_id)
# qq <- sigInfo %>% select(sig_id,cmap_name,pert_idose,pert_itime) %>%
#   filter(grepl('IFNA',cmap_name) | grepl('IFNG',cmap_name) | grepl('TGFB',cmap_name))
qq <- sigInfo %>% select(sig_id,cmap_name,pert_idose,pert_itime) %>%
  filter(sig_id %in% sigInfo_filt$sig_id)

x <- gex_standarized_l1000[,sigInfo_filt$sig_id]
x <- x %>% rownames_to_column('gene') %>%
  filter(gene %in% c('STAT1','STAT2','STAT3','SMAD2','SMAD3','TP53','TP63','IFNAR1','IFNAR2','IFNGR1','IFNGR2','TGFBR1','TGFBR2'))



TF_activities_L1000 <- TF_activities_L1000 %>% filter(sig_id %in% sigInfo_filt$sig_id)
TF_activities_L1000 <- left_join(TF_activities_L1000,sigInfo_filt %>% select(sig_id,cmap_name,canonical_smiles,pert_idose,pert_itime))
### Load PHHs microarray and TF activities
gex_rma <- readRDS('../preprocessing/preprocessed_data/gex_normalized_aggregated_microarray.rds')
#Standarize gene expression
gex_standarized_phh <- scale(t(gex_rma),scale = F)
# PHH_gene_expression <- 

# Load TF activities
TF_activities_phh <- read.table('../preprocessing/preprocessed_data/TF_activities/TF_acitivities_hepatocytes_microarray.tsv',
                                sep = "\t",header = TRUE)
TF_activities_phh <- TF_activities_phh %>% filter(X %in% PHH_metadata_filt$X) %>%
  gather('uniprotID','activity',-X)
annot <- read.csv('../data/uniprot-reviewed_yes+AND+organism__Homo+sapiens+(Human)+[9606]_.tab', sep='\t')
annot <- annot %>% dplyr::select(Gene.names...primary..,Entry)
annot <- annot %>% filter(Entry %in% TF_activities_phh$uniprotID)
colnames(annot) <- c('TF','uniprotID')
TF_activities_phh <- left_join(TF_activities_phh,annot) %>% select(-uniprotID)
TF_activities_phh <- left_join(TF_activities_phh,PHH_metadata_filt)

common_tfs <- intersect(TF_activities_phh$TF,TF_activities_L1000$TF)

### Visualize TF results--------------
allTF_results <- left_join(TF_activities_phh,TF_activities_L1000) %>%
  filter(TF %in% common_tfs)
allTF_results <- allTF_results %>% filter(TF %in% c('STAT1','STAT2','STAT3','SMAD2','SMAD3','TP53','TP63'))
allTF_results <- allTF_results %>% group_by(canonical_smiles) %>% 
  mutate(r = cor(activity,activity_L1000)) %>% 
  ungroup()
allTF_results <- allTF_results %>% group_by(TF) %>% mutate(r_per_TF = cor(activity,activity_L1000))%>% ungroup()

ggplot(allTF_results,aes(x=activity_L1000,y=activity))+
  geom_point()+
  geom_hline(yintercept = 0.4)+
  geom_hline(yintercept = 0.6)+
  geom_vline(xintercept = 0.4)+
  geom_vline(xintercept = 0.6)+
  facet_wrap(~TF)
