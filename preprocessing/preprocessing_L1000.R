library(tidyverse)
library(cmapR)
library(doFuture)

# parallel set number of workers
cores <- 15
registerDoFuture()
plan(multisession,workers = cores)

#### Load sigInfo and geneInfo of the new L1000 dataset----------------------------------------
# Check L1000 documentation for information.
geneInfo <- read.delim('../../../L1000_2021_11_23/geneinfo_beta.txt')
geneInfo <-  geneInfo %>% filter(feature_space != "inferred")
# Keep only protein-coding genes
geneInfo <- geneInfo %>% filter(gene_type=="protein-coding")

#Load signature info and split data to high quality replicates and low quality replicates
sigInfo <- read.delim('../../../L1000_2021_11_23/siginfo_beta.txt')

# Create a proxy for quality of replicates
# Keep only samples with at least 3 replicates and that satisfy specific conditions.
# Check the LINCS2020 Release Metadata Field Definitions.xlsx file for 
# a complete description of each argument. It can be accessed online 
# or in the data folder.
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

## Keep only liver cells (HEPG2)
sigInfo <- sigInfo %>% filter(cell_iname=='HEPG2')

p <- ggplot(sigInfo %>% group_by(pert_itime) %>% mutate(n_points=n()) %>% ungroup() %>%
              mutate(n_points=as.numeric(n_points)) %>%
              mutate(pert_itime=factor(pert_itime,
                                       levels = c('1 h','2 h','3 h',"4 h",
                                                  '6 h','24 h','48 h','72 h',
                                                  '96 h','120 h','144 h','168 h'))) %>% 
              dplyr::select(pert_itime,n_points) %>% unique(),
            aes(pert_itime,n_points)) +
  geom_bar(stat="identity") + scale_y_log10() +
  xlab('Perturbation time duration (h)') + ylab('Number of data points (log10 scale)')
print(p)
p <- ggplot(sigInfo %>% group_by(pert_itime) %>% mutate(n_points=n()) %>% ungroup() %>%
              mutate(n_points=as.numeric(n_points)) %>%
              mutate(pert_itime=factor(pert_itime,
                                       levels = c('1 h','2 h','3 h',"4 h",
                                                  '6 h','24 h','48 h','72 h',
                                                  '96 h','120 h','144 h','168 h'))) %>% 
              dplyr::select(pert_itime,tas) %>% unique(),
            aes(pert_itime,tas)) + geom_boxplot(outliers = FALSE)+ geom_jitter()+
  xlab('Perturbation time duration (h)') + ylab('Transcriptional Activity Score')
print(p)

## Keep 2h as training stimulation as it has more samples and a slightly higher TAS 
sigInfo <- sigInfo %>% filter(pert_itime=='2 h')
sigInfo <- sigInfo %>% mutate(inst_id=strsplit(distil_ids,"\\|")) %>% unnest(inst_id)

### Load Level-3 gene expression-----------------------------

# rid is the gene entrez_id to find the gene in the data
# cid is the sig_id, meaning the sampe id
# path is the path to the data
parse_gctx_parallel <- function(path ,rid,cid){
  gctx_file <- parse_gctx(path ,rid = rid,cid = cid)
  return(gctx_file@mat)
}
# Read GeX for ligands
# Split sig_ids to run in parallel
sampleIds <- unique(sigInfo$inst_id)
idList <-  split(sampleIds, 
                  ceiling(seq_along(sampleIds)/ceiling(length(sampleIds)/cores)))

# Parallelize parse_gctx function

# Path to raw data
ds_path <- '../../../L1000_2021_11_23/level3_beta_trt_misc_n26428x12328.gctx'

# Parse the data file in parallel
cmap_gctx <- foreach(sigs = idList) %dopar% {
  parse_gctx_parallel(ds_path ,
                      rid = unique(as.character(geneInfo$gene_id)),
                      cid = sigs)
}
gc()
cmap <-do.call(cbind,cmap_gctx)
saveRDS(cmap,'preprocessed_data/lvl3_ligands_l1000_gex.rds')

