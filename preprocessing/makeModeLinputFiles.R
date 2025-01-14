library(tidyverse)

### Load trimmed and full data--------------
metadata <- read.delim('../data/E-MTAB-798.sdrf.txt')
drug_info_full <- readRDS('preprocessed_data/final_DTI_info_microarray.rds')
drug_info_full <- drug_info_full %>% select(name,canonical_smiles) %>% unique()
annot_full <- read.delim('../data/uniprot-reviewed_yes+AND+organism__Homo+sapiens+(Human)+[9606]_.tab')
annot_full <- annot_full %>% select(Entry,c('Target'=Gene.names...primary..)) %>% unique()
DTI_full <- read.delim('preprocessed_data/PKN/DTI_microarray.tsv',sep = '\t')
DTI_trimmed <- read.delim('preprocessed_data/PKN/DTI_microarray_trimmed.tsv',sep = '\t')
print(length(unique(DTI_trimmed$source)))
PKN <- read.delim('preprocessed_data/PKN/Hepatocytes-PKN-Model.tsv',sep = '\t')
TrimmedTFs <- read.delim('preprocessed_data/TF_activities/Trimmed_TFacitivities_hepatocytes_microarray.tsv',
                         sep = '\t') 
TrimmedTFs <- TrimmedTFs %>% column_to_rownames('X')
TrimmedTFs <- as.matrix(TrimmedTFs)

### Create the 4 important files to train DT-LEMBAS:-------------------
### 1) Y : TF_activities (already created)
### 2) X : samples x drugs, with column names the smiles and values log-transformed doses
### 3) mask : binary drug-target matrix (smiles x targets)
### 4) similarity CSV file with ECFP4 similarity which will be used by DT-LEMBAS

# Make DTI mask
drug_targets <- DTI_trimmed %>% mutate(value=1) %>% spread(target,value)
drug_targets <- drug_targets %>% column_to_rownames('source')
drug_targets[is.na(drug_targets)] <- 0
drug_targets <- as.matrix(drug_targets)
# Put the drugs missing with all their values zeros
drug_add <- matrix(0,
                   nrow = length(which(!(unique(DTI_full$source) %in% rownames(drug_targets)))) + 1,
                   ncol = ncol(drug_targets))
rownames(drug_add) <- c(unique(DTI_full$source[which(!(DTI_full$source %in% rownames(drug_targets)))]),
                        'Control')
colnames(drug_add) <- colnames(drug_targets)
drug_targets <- rbind(drug_targets,drug_add)
write.table(drug_targets,
            file ='preprocessed_data/PKN/Hepatocytes_DTI_mask.tsv' , 
            quote=FALSE, sep = "\t", row.names = TRUE,col.names = NA)

# Make input X matrix
inputdata <- metadata %>%
  mutate(dose = ifelse(grepl('Low', Description), 'Low',
                       ifelse(grepl('Middle', Description), 'Middle',
                              ifelse(grepl('High', Description), 'High', 'Control')))) %>%
  mutate(condition = ifelse(grepl('Control', dose), 'control', 'treated')) %>%
  mutate(time = paste0(Factor.Value.time., ' ', Unit.time.unit.)) %>%
  dplyr::select(c('drug' = 'Factor.Value.cohort.'), condition, dose,
                c('dose_value'='Factor.Value.dose.'), time, Array.Data.File) %>%
  column_to_rownames('Array.Data.File')
inputdata <- inputdata %>% mutate(dose_value= ifelse(is.na(dose_value),
                                                     median(metadata$Factor.Value.dose.,na.rm = TRUE),
                                                     dose_value))
inputdata <- inputdata %>% mutate(replicate_id = paste(drug,dose,time,sep = "+"))
inputdata <- inputdata %>% select(-condition,-dose) %>%
  select(c('condition'='replicate_id'),drug,c('dose'='dose_value'),time) %>% unique()
rownames(inputdata) <- NULL
inputdata <- distinct(inputdata)
inputdata <- inputdata %>% filter(condition %in% rownames(TrimmedTFs))
inputdata <- left_join(inputdata %>% mutate(drug=ifelse(grepl('Control',condition),'Control',drug)),
                       rbind(drug_info_full,
                             data.frame(name = c('Control'),
                                        canonical_smiles = c('Control'))) %>% 
                         select(c('drug'='name'),canonical_smiles))
write.table(inputdata %>% column_to_rownames('condition'),
            file ='preprocessed_data/PKN/Hepatocytes_inputdataInfo.tsv' , 
            quote=FALSE, sep = "\t", row.names = TRUE,col.names = NA)
Xinput <- inputdata %>% select(condition,canonical_smiles,dose) %>%
  mutate(log10Dose = log10(dose+1)) %>% select(-dose) %>%
  spread(canonical_smiles,log10Dose) %>% unique() %>% 
  column_to_rownames('condition')
Xinput[is.na(Xinput)] <- 0
Xinput <- as.matrix(Xinput)
Xinput <- Xinput[rownames(TrimmedTFs),rownames(drug_targets)]
write.table(as.data.frame(Xinput),
            file ='preprocessed_data/PKN/Hepatocytes_input_conditions.tsv' , 
            quote=FALSE, sep = "\t", row.names = TRUE,col.names = NA)

## Make similarity matrix for the model
similarities <- as.matrix(data.table::fread('preprocessed_data/ChemicalSims/all_similarities.csv') %>%
  column_to_rownames('V1'))

add_control_row <- matrix(0,nrow=1,ncol=ncol(similarities))
colnames(add_control_row) <- colnames(similarities)
rownames(add_control_row) <- c('Control')
similarities <- rbind(similarities,add_control_row)

add_control_col <- matrix(0,nrow=nrow(similarities),ncol=1)
rownames(add_control_col) <- rownames(similarities)
colnames(add_control_col) <- c('Control')
add_control_col['Control','Control'] <- 1
similarities <- cbind(similarities,add_control_col)

similarities <- similarities[rownames(drug_targets),rownames(drug_targets)]
data.table::fwrite(as.data.frame(similarities),
                   'preprocessed_data/ChemicalSims/microarray_similarities.csv',
                   row.names = TRUE)
