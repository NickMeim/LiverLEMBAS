library(tidyverse)

####  Load drug-bank data and load also the drugs that are tested in the hepatocytes ---------------
microarray_metadata <- read.delim('../data/E-MTAB-798.sdrf.txt')
# 2 drugs had wrong chembl id
microarray_metadata <- microarray_metadata %>%
  mutate(Term.Accession.Number=ifelse(Factor.Value.cohort.=='lornoxicam','CHEMBL1569487',
                                      ifelse(Factor.Value.cohort.=='ajmaline','CHEMBL5314404',
                                             Term.Accession.Number)))
TF_activities <- read.delim('preprocessed_data/TF_activities/TF_acitivities_hepatocytes_microarray.tsv',
                            sep = '\t') 
drugs_microarrayed <- TF_activities %>% select(X) %>% separate(X,into=c('drug','dose','time'),sep="\\+")
drugs_microarrayed <- unique(drugs_microarrayed$drug)
TF_activities <-1
gc()

### Load cmap from DT-LEMBAS paper (DOI: 10.1016/j.isci.2024.109509) as it contains mapping of drug names and smiles
cmap_drugs <- readRDS('../../../Artificial-Signaling-Network/DrugsANNSignaling/preprocessing/preprocessed_data/l1000_drugs_with_targets_all.rds')
annot <- read.delim('../data/uniprot-reviewed_yes+AND+organism__Homo+sapiens+(Human)+[9606]_.tab')
annot <- annot %>% select(Entry,c('Target'=Gene.names...primary..)) %>% unique()
cmap_drugs <-  cmap_drugs %>% mutate(Target=strsplit(Target,", ")) %>% unnest(Target)
cmap_drugs <- left_join(cmap_drugs,annot)
cmap_drugs <- cmap_drugs %>% filter(!is.na(Entry))
data <- readRDS("../../../Artificial-Signaling-Network/DrugsANNSignaling/preprocessing/preprocessed_data/all_cmap_sigs_with_pert_info.rds")
data <- data %>% filter(is_exemplar==1)  %>%
  filter(pert_itime %in% c("6 h"))
data <- data %>% dplyr::select(sig_id,pert_iname,pert_type,pert_idose,pert_itime,cell_id,pert_id,canonical_smiles) %>% unique()
cmap_ctrls <- data %>% filter(pert_type=='ctl_vehicle' | pert_type=='ctl_untrt') %>% unique()
cmap_ctrls <- cmap_ctrls %>% mutate(canonical_smiles = ifelse(pert_id=='DMSO','CS(C)=O',
                                                              ifelse(pert_id=='H2O','O',
                                                                     ifelse(pert_id=='PBS','Ctrl_PBS','Ctrl_UnTrt'))))
cmap_ctrls <- cmap_ctrls %>% mutate(Entry=ifelse(pert_id=='DMSO',c('Q01344', 'Q8WXI7', 'P01106'),'None')) %>% unnest(Entry)
cmap_ctrls <- cmap_ctrls  %>%  mutate(Target=ifelse(Entry=='Q01344','IL5RA',ifelse(Entry=='P01106','MYC',ifelse(Entry=='Q8WXI7','MUC16',Entry)))) 
cmap_ctrls <- cmap_ctrls %>% mutate(MOA=NA,Phase=NA,Disease.Area='control',pubchem_cid=NA)
cmap_drugs <- left_join(cmap_drugs,data)
cmap_ctrls <- cmap_ctrls %>% select(colnames(cmap_drugs))
cmap_drugs <- rbind(cmap_drugs,cmap_ctrls)
cmap_drugs <- cmap_drugs %>% filter(!is.na(Entry))
cmap_drugs <- cmap_drugs %>% filter(Entry!='')
cmap_drugs <- cmap_drugs %>% filter(Entry!=' ')
cmap_drugs_filtered <- cmap_drugs %>% filter(tolower(pert_iname) %in% c("aspirin","glibenclamide"))%>%
  select(canonical_smiles,c('name'='pert_iname'),c('targets'='Target')) %>% unique()
cmap_drugs_filtered <- cmap_drugs_filtered %>% 
  mutate(notes=ifelse(name=='glibenclamide',
                      'other name:glyburide,CHEMBL id:CHEMBL472',
                      'other name:acetylsalicylic acid,CHEMBL id:CHEMBL25')) %>%
  mutate(canonical_smiles = ifelse(name=='glibenclamide',
                                   'COc1ccc(Cl)cc1C(=O)NCCc1ccc(S(=O)(=O)NC(=O)NC2CCCCC2)cc1',
                                   'CC(=O)Oc1ccccc1C(=O)O'))

### Load drugbank from DT-LEMBAS paper (DOI: 10.1016/j.isci.2024.109509)
drug_targets <- readRDS('../preprocessing/preprocessed_data/drug_targets_space.rds')
df_drugbank <- readRDS('../data/df_drugbank.rds')
df_drugbank <- df_drugbank %>% 
  mutate(target_symbol=ifelse(target_symbol=='CP4F2','CYP4F2',target_symbol)) # wrong symbol
# drugs2forcekeep <- c('thioacetamide')
manual_entries <- rbind(data.frame(canonical_smiles = c('CC1=C(C(=CC=C1)NC2=CC(=NC(=N2)SCC(=O)O)Cl)C'),
                             name = c('WY-14643'),
                             targets = c('PPARA','PPARD','PPARG'),
                             notes= c('CHEMBL id:CHEMBL295416')),
                        data.frame(canonical_smiles = 'CC1=C(C(CCC1)(C)C)/C=C/C(=C/C=C/C(=C/CO)/C)/C',
                                   name = 'vitamin A',
                                   targets = c('RXRG','APOD','PTGDS',
                                               'RDH12','RDH5','RDH13',
                                               'ALDH1A2','ALDH1A1','LRAT',
                                               'DHRS3','ALDH1A3','RETSAT',
                                               'RDH11','DHRS4','RDH14',
                                               'RDH8','CYP26A1'),
                                   notes= c('Also known as retinol, with CHEMBL id:CHEMBL986')),
                        data.frame(canonical_smiles = 'CC(=N)S',
                                   name = 'thioacetamide',
                                   targets = c('CYP2E1'),
                                   notes= c('Interaction uncertain.TAA goes through a two-step bioactivation interceded by microsomal CYP2E1. CHEMBL id:CHEMBL38737')),
                        data.frame(canonical_smiles = 'O=C(OC[C@H]1O[C@@H](OC(=O)c2cc(O)c(O)c(OC(=O)c3cc(O)c(O)c(O)c3)c2)[C@H](OC(=O)c2cc(O)c(O)c(OC(=O)c3cc(O)c(O)c(O)c3)c2)[C@@H](OC(=O)c2cc(O)c(O)c(OC(=O)c3cc(O)c(O)c(O)c3)c2)[C@@H]1OC(=O)c1cc(O)c(O)c(OC(=O)c2cc(O)c(O)c(O)c2)c1)c1cc(O)c(O)c(OC(=O)c2cc(O)c(O)c(O)c2)c1',
                                   name = 'tannic acid',
                                   targets = c('ANO1','PTGS1','PTGS2'),
                                   notes= c('CHEMBL id:CHEMBL506247')),
                        data.frame(canonical_smiles = 'O=C(O)c1ccccc1Nc1ccccc1',
                                   name = 'phenylanthranilic acid',
                                   targets = c('TTR','NLRP3','PTGS1','PTGS2'),
                                   notes= c('CHEMBL id:CHEMBL23832')),
                        data.frame(canonical_smiles = 'NC1=NC(=O)C(c2ccccc2)O1',
                                   name = 'pemoline',
                                   targets = c('CAMP','MAOA'),
                                   notes= c('CHEMBL id:CHEMBL1177')),
                        data.frame(canonical_smiles = 'NC(=O)N/N=C/c1ccc([N+](=O)[O-])o1',
                                   name = 'nitrofurazone',
                                   targets = c('XDH'),
                                   notes= c('CHEMBL id:CHEMBL869')),
                        data.frame(canonical_smiles = 'CCN(CC)N=O',
                                   name = 'nitrosodiethylamine',
                                   targets = c('CYP2E1'),
                                   notes= c('CHEMBL id:CHEMBL164290')),
                        data.frame(canonical_smiles = 'O=C(O)c1cccnc1',
                                   name = 'nicotinic acid',
                                   targets = unique(c(df_drugbank$target_symbol[which(df_drugbank$name=='niacin')],
                                                      'CYP4F2')),
                                   notes= c('Other name:niacin, CHEMBL id:CHEMBL573')),
                        data.frame(canonical_smiles = 'S=C=Nc1cccc2ccccc12',
                                   name = 'naphthyl isothiocyanate',
                                   targets = c('CYP1A2','CYP2E1'),
                                   notes= c('CHEMBL id:CHEMBL1381098')),
                        data.frame(canonical_smiles = 'C[C@H]1C(=O)O[C@@H]2CCN3CC=C(COC(=O)[C@](C)(O)[C@]1(C)O)[C@H]23',
                                   name = 'monocrotaline',
                                   targets = c('BMPR2','CYP3A4','CYP2B6'),
                                   notes= c('CHEMBL id:CHEMBL521035')),
                        data.frame(canonical_smiles = 'Clc1c(Cl)c(Cl)c(Cl)c(Cl)c1Cl',
                                   name = 'hexachlorobenzene',
                                   targets = c('AHR'),
                                   notes= c('CHEMBL id:CHEMBL228514')),
                        data.frame(canonical_smiles = 'COc1ccc(Cl)cc1C(=O)NCCc1ccc(S(=O)(=O)NC(=O)NC2CCCCC2)cc1',
                                   name = 'glibenclamide',
                                   targets = unique(df_drugbank$target_symbol[which(df_drugbank$name=='glyburide')]),
                                   notes= c('other name:glyburide,CHEMBL id:CHEMBL472')),
                        data.frame(canonical_smiles = 'CCSCC[C@H](N)C(=O)O',
                                   name = 'ethionine',
                                   targets = c('unknown'),
                                   notes= c('CHEMBL id:CHEMBL203187')),
                        data.frame(canonical_smiles = 'CCOC(=O)CCC(=O)O[C@H]1[C@H](O[C@@H]2[C@@H](C)[C@H](O[C@H]3C[C@@](C)(OC)[C@@H](O)[C@H](C)O3)[C@@H](C)C(=O)O[C@H](CC)[C@@](C)(O)[C@H](O)[C@@H](C)C(=O)[C@H](C)C[C@@]2(C)O)O[C@H](C)C[C@@H]1N(C)C',
                                   name = 'erythromycin ethylsuccinate',
                                   targets = unique(df_drugbank$target_symbol[which(df_drugbank$name=='erythromycin')]),
                                   notes= c('Put erythromycin targets as initialization.CHEMBL id:CHEMBL1200688')),
                        data.frame(canonical_smiles = 'ClC(Cl)(Cl)Cl',
                                   name = 'carbon tetrachloride',
                                   targets = c('CYP2E1'),
                                   notes= c('CHEMBL id:CHEMBL44814')),
                        data.frame(canonical_smiles = 'CCOc1ccc(NC(=O)CC(C)O)cc1',
                                   name = 'bucetin',
                                   targets = unique(df_drugbank$target_symbol[df_drugbank$name=='phenacetin']),
                                   notes= c('Initialize targets with the structural related phenacetin. CHEMBL id:CHEMBL1697856')),
                        data.frame(canonical_smiles = 'Brc1ccccc1',
                                   name = 'bromobenzene',
                                   targets = c('GRP78','PDIA1','ERP29','PDIA6'),
                                   notes= c('CHEMBL id:CHEMBL16068')),
                        data.frame(canonical_smiles = 'CC(=O)Oc1ccccc1C(=O)O',
                                   name = 'aspirin',
                                   targets = unique(df_drugbank$target_symbol[which(df_drugbank$name=='acetylsalicylic acid')]),
                                   notes= c('other name:acetylsalicylic acid,CHEMBL id:CHEMBL25')),
                        data.frame(canonical_smiles = 'CC(=O)Nc1ccc2c(c1)Cc1ccccc1-2',
                                   name = 'acetamidofluorene',
                                   targets = c('CYP1A2','CYP2C9','AHR'),
                                   notes= c('CHEMBL id:CHEMBL311469')),
                        data.frame(canonical_smiles = 'C=CCO',
                                   name = 'allyl alcohol',
                                   targets = c('MGMT','ERCC1','ERCC4','ALDH2'),
                                   notes= c('CHEMBL id:CHEMBL234926')),
                        data.frame(canonical_smiles = 'CN(C)CCC=C1c2ccccc2COc2ccccc21',
                                   name = 'adapin',
                                   targets =unique(df_drugbank$target_symbol[which(df_drugbank$name=='doxepin')]),
                                   notes= c('other names:doxepin,CHEMBL id:CHEMBL1628227')))

df_drugbank <- df_drugbank %>% filter((tolower(name) %in% tolower(drugs_microarrayed)))
### Merge with CHEMBL
part1 <- read.delim('../data/chembl_part1.tsv')
part2 <- read.delim('../data/chembl_part2.tsv',header = FALSE)
colnames(part2) <- colnames(part1)
chembl_db <- rbind(part1%>% select(ChEMBL.ID,Smiles),
                   part2%>% select(ChEMBL.ID,Smiles))
part1 <- 1
part2 <- 1
chembl_db <- chembl_db %>% filter(!is.na(ChEMBL.ID)) %>% filter(ChEMBL.ID!='') %>%
  filter(ChEMBL.ID!=' ') %>% filter(ChEMBL.ID!='\t') %>%
  filter(!is.na(Smiles)) %>% filter(Smiles!='') %>%
  filter(Smiles!=' ') %>% filter(Smiles!='\t')
chembl_db <- distinct(chembl_db)
gc()
colnames(chembl_db) <- c('Term.Accession.Number','canonical_smiles')
chembl_db <- chembl_db %>% filter(Term.Accession.Number %in% microarray_metadata$Term.Accession.Number)
gc()

chembl_microarrayed <- left_join(chembl_db,
                                 microarray_metadata %>% 
                                   select(Term.Accession.Number,c('name'='Factor.Value.cohort.')) %>%
                                   unique())
all_drug_targets <- distinct(left_join(as.data.frame(drug_targets) %>% rownames_to_column('canonical_smiles') %>%
  gather('Entry','value',-canonical_smiles) %>% filter(value==1) %>% select(-value),
  annot) %>% select(canonical_smiles,c('targets'='Target')))
chembl_microarrayed <- left_join(chembl_microarrayed,all_drug_targets) %>%
  mutate(notes = paste0('CHEMBL id:',Term.Accession.Number)) %>% select(-Term.Accession.Number)
chembl_microarrayed <- chembl_microarrayed %>% filter(!is.na(targets))
microarray_drugs_info <- rbind(chembl_microarrayed,cmap_drugs_filtered,manual_entries) %>% unique()

## Now get just the drugbank plus chembl
df_drugbank_filtered <- df_drugbank %>% filter(!(tolower(name) %in% manual_entries$name)) %>%
  select(name,c('targets'='target_symbol')) %>% unique()
chembl_drugbank_microarrayed <- left_join(chembl_db,
                                             microarray_metadata %>% 
                                               select(Term.Accession.Number,c('name'='Factor.Value.cohort.')) %>%
                                               unique())
chembl_drugbank_microarrayed <- left_join(chembl_drugbank_microarrayed,df_drugbank_filtered) %>%
  mutate(notes = paste0('CHEMBL id:',Term.Accession.Number)) %>% select(-Term.Accession.Number)
chembl_drugbank_microarrayed <- chembl_drugbank_microarrayed %>% filter(!is.na(targets))
chembl_drugbank_microarrayed <- chembl_drugbank_microarrayed %>%
  filter(!(notes %in% microarray_drugs_info$notes))

print(length(unique(c(chembl_drugbank_microarrayed$name,microarray_drugs_info$name))))

final_DTI_info <- distinct(rbind(chembl_drugbank_microarrayed,microarray_drugs_info))
saveRDS(final_DTI_info,'preprocessed_data/final_DTI_info_microarray.rds')

## Augment PKN drug-target based on very high chemical similarity---------------
similarities <- data.table::fread('preprocessed_data/ChemicalSims/all_similarities.csv')
colnames(similarities)[1] <- 'drug.x'
similarities <- similarities %>% filter(drug.x %in% final_DTI_info$canonical_smiles)
similarities <- similarities %>% gather('drug.y','similarity',-drug.x) %>%
  filter(drug.x!=drug.y) %>% filter(similarity!=10) %>%
  unique()
### Just augment with target of drugs with similarity>=0.8
### and for ethionine ( CCSCC[C@H](N)C(=O)O ) get its highest similarity neighbor
### and the targets that also can be predicted by some protein-drug bunding model
similarities <- similarities %>% group_by(drug.x) %>% mutate(max_sim = max(similarity)) %>%
  ungroup() %>% filter(similarity>=0.8 | (drug.x=='CCSCC[C@H](N)C(=O)O' & similarity==max_sim))
augmented_drug_targets <- left_join(similarities %>% select(-similarity,-max_sim),
                                    as.data.frame(drug_targets) %>% rownames_to_column('drug.y') %>%
                                      gather('target','value',-drug.y) %>% 
                                      filter(value==1) %>% select(-value))
augmented_drug_targets <- augmented_drug_targets %>% select(c('source'='drug.x'),target) %>% unique()
## save for DT-LEMBAS preprocessing scirpts-----------------
DTI <- left_join(final_DTI_info %>% select(c('source'='canonical_smiles'),c('Target'='targets')),
                 annot %>% select(Target,c('target'='Entry'))) %>% unique()
DTI <- DTI %>% select(-Target) %>% filter(!is.na(target)) %>% unique()
DTI <- rbind(DTI,augmented_drug_targets) %>% unique()
print(length(unique(DTI$source)))
write.table(DTI,
            file ='preprocessed_data/PKN/DTI_microarray.tsv' , 
            quote=FALSE, sep = "\t", row.names = FALSE)
