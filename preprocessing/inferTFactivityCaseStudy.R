# Load requeired packages
library("dorothea")
library(tidyverse)

# Access command-line arguments
# args <- commandArgs(trailingOnly = TRUE)

# inputGeneExpr <- args[1]
# outputFile = args[2]
inputGeneExpr <- 'preprocessed_data/lvl3_ligands_l1000_gex.rds'
outputFile = "preprocessed_data/TF_activities/ligands_l1000_allgenes_lvl3_tfs.rds"
minNrOfGenes = 5
E <- readRDS(inputGeneExpr)
# E <- E[,1:15000]
gc()

dorotheaData = read.table('../data/dorothea.tsv', sep = "\t", header=TRUE)
confidenceFilter = is.element(dorotheaData$confidence, c('A', 'B'))
dorotheaData = dorotheaData[confidenceFilter,]

# Standarize gene expression before running viper to infer TF activity
E.standardized = E-rowMeans(E)
colnames(E.standardized) = colnames(E)
gc()
# Load gene information
geneInfo <- read.delim('../../../L1000_2021_11_23/geneinfo_beta.txt')
geneInfo <-  geneInfo %>% filter(feature_space != "inferred")
# Keep only protein-coding genes
geneInfo <- geneInfo %>% filter(gene_type=="protein-coding")
print(all(rownames(E.standardized)==geneInfo$gene_id))
rownames(E.standardized) <- geneInfo$gene_symbol
E <- NULL
gc()

# Estimate TF activities
settings = list(verbose = TRUE, minsize = minNrOfGenes)
TF_activities = run_viper(E.standardized, dorotheaData, options =  settings)

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
saveRDS(TF_activities, outputFile)

