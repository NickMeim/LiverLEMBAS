library(tidyverse)
library(PharmacoGx)

### Load info data
inputInfo <- data.table::fread('preprocessed_data/TrainingValidationData/Hepatocytes_inputdataInfo.tsv',sep = '\t')
inputInfo <- inputInfo %>% 
  select(drug,dose,time,canonical_smiles) %>% 
  filter(drug!='Control') %>%
  unique()
