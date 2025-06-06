library(tidyverse)
library(dbparser)
library(XML)

# dvobj <- parseDrugBank(db_path= "../../../Artificial-Signaling-Network/DrugsANNSignaling/data/full database.xml",
#                        drug_options= drug_node_options(),
#                        parse_salts= TRUE,
#                        parse_products= TRUE,
#                        references_options= references_node_options(),
#                        cett_options= cett_nodes_options())
# # saveRDS(dvobj,'dvobj.rds')
# snp_effects <- dvobj$drugs$snp_effects
# drug_interactions <- dvobj$drugs$drug_interactions
# snp_effects <- snp_effects %>% select(`protein-name`,`gene-symbol`,`uniprot-id`,`pubmed-id`,parent_key)
# drug_interactions <- drug_interactions %>% select(-description)
# drug_interactions <- left_join(drug_interactions,snp_effects)
# df_drugbank <- dvobj$drugs$general_information
# df_drugbank <- left_join(df_drugbank,drug_interactions,by=c('primary_key'='drugbank-id','name'='name'))
# df_drugbank$name <- tolower(df_drugbank$name)
# df_drugbank <- df_drugbank %>%
#   select(c('drugbank_id'='primary_key'),'name','cas_number',
#          c('target_symbol'='gene-symbol'),c('target_uniprot'='uniprot-id')) %>%
#   filter(!is.na(target_symbol) | !is.na(target_uniprot)) %>% unique()


### PharmacoGenomics
library(PharmacoGx)

# Load lethality/viability datasets----------------------
columns <- c('drugid','cid','inchikey','smiles')
available_datasets <- availablePSets()
available_datasets <- available_datasets %>% filter(!(`PSet Name` %in% c('GDSC_2020(v1-8.2)','GBM_scr2','GBM_scr3','PDTX_2019','BeatAML_2018','Tavor_2020')))
datasets <- available_datasets$`PSet Name`
names <- available_datasets$`Dataset Name`
print(datasets)
data_all <- data.frame()
# data_all <- readRDS('../results/viability_data_all.rds')
for (j in 1:length(datasets)){
  # message(paste0('Begun dataset: ',datasets[j]))
  dataset <- PharmacoGx::downloadPSet(datasets[j])
  cell_line_sensitivity <- dataset@sensitivity[["profiles"]]
  pert_info <-dataset@sensitivity[["info"]]
  drug_info <- dataset@drug
  if (any(!(columns %in% colnames(drug_info)))){
    cols_add <- columns[which(!(columns %in% colnames(drug_info)))]
    for (column in cols_add){
      drug_info[[column]] <- NA
    }
  }
  drug_info <- drug_info %>% select(all_of(columns)) %>% unique()
  drug_info <- drug_info %>% mutate(cid = as.character(cid))
  if ('ec50' %in% colnames(cell_line_sensitivity)){
    ind <- which(colnames(cell_line_sensitivity)=='ec50')
    colnames(cell_line_sensitivity)[ind] <- 'EC50'
  }
  ## Keep perturbations only for those drugs
  pert_info <- pert_info %>% filter(drugid %in% drug_info$drugid)
  pert_info <- pert_info %>% mutate(cellid=gsub('-','',cellid))
  pert_info <- pert_info %>% mutate(cellid=toupper(cellid))
  drug_info <- drug_info %>% filter(drugid %in% pert_info$drugid)
  pert_info <- pert_info %>% filter(drugid %in% drug_info$drugid)
  # filter sensitivity data
  cell_line_sensitivity <- cell_line_sensitivity[rownames(pert_info),]
  gc()
  if (!('exp_id' %in% colnames(pert_info))){
    pert_info <- pert_info %>% rownames_to_column('exp_id')
    rownames(pert_info) <- pert_info$exp_id
  }
  # filter to keep drugs that are in our data
  cell_line_sensitivity$EC50 <- log10(cell_line_sensitivity$EC50)
  pert_info <- pert_info %>% filter(drugid %in% drug_info$drugid)
  cell_line_sensitivity <- cell_line_sensitivity[rownames(pert_info),]
  cell_line_sensitivity <- cell_line_sensitivity %>% rownames_to_column('exp_id') %>%
    select(exp_id,EC50) %>% unique()
  # create first cell-line one-hot encoding matrix
  cell_info <- pert_info %>% select(exp_id,cellid) %>% unique()
  pert_info <- pert_info %>% select(exp_id,drugid,cellid)
  pert_info <- left_join(pert_info,drug_info %>% unique())
  pert_info <- left_join(pert_info,cell_info)
  
  # get all data
  data <- left_join(pert_info,cell_line_sensitivity)
  data <- data %>% group_by(drugid,cellid) %>% mutate(EC50=mean(EC50,na.rm=T)) %>% ungroup()
  data <- data %>% filter(!is.na(EC50))
  data <- distinct(data)
  
  # merge with other datasets
  data_all <- rbind(data_all,data %>% mutate(dataset = names[j]))
  # saveRDS(data_all,'../results/viability_data_all.rds')
  message(paste0('Finished dataset: ',datasets[j]))
}

