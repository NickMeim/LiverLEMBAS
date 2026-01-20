library(tidyverse)
library(igraph)
source("../postprocessing/plotting_functions.R")
source('../postprocessing/functions_translation.R')

# Read the network file
network <- read.table("../preprocessing/preprocessed_data/PKN/Hepatocytes-PKN-Model.tsv", 
                      header = TRUE, 
                      sep = "\t", 
                      stringsAsFactors = FALSE)

# Read the annotation file
annotations <- read.table("../preprocessing/preprocessed_data/PKN/Hepatocytes-PKN-Annotation.tsv", 
                          header = TRUE, 
                          sep = "\t", 
                          stringsAsFactors = FALSE)

# Create a graph from the network (using source and target columns)
g <- graph_from_data_frame(network[, c("source", "target")], 
                           directed = TRUE)

# Define starting nodes
start_nodes <- c("P36897", "P17181")

# Find all reachable nodes from the starting nodes
reachable_nodes <- c()

for (start_node in start_nodes) {
  # Check if the node exists in the graph
  if (start_node %in% V(g)$name) {
    # Get all nodes reachable from this starting node
    reachable <- subcomponent(g, start_node, mode = "out")
    reachable_nodes <- c(reachable_nodes, V(g)$name[reachable])
  } else {
    warning(paste("Node", start_node, "not found in the network"))
  }
}

# Remove duplicates
reachable_nodes <- unique(reachable_nodes)

# Filter to get only TFs from the reachable nodes
# Select relevant columns: code (UniProt), name (symbol), and TF status
reachable_tfs <- annotations %>%
  filter(code %in% reachable_nodes & TF == "True") %>%
  select(code, name, TF) %>%
  arrange(name)  # Sort by name for easier reading

# Display results
cat("Starting nodes:", paste(start_nodes, collapse = ", "), "\n\n")
cat("Number of reachable nodes:", length(reachable_nodes), "\n")
cat("Number of reachable TFs:", nrow(reachable_tfs), "\n\n")
reachable_tfs <- reachable_tfs %>% select(name) %>% unique()

# Load human data, infer TF activity and train model to predict MAS, Fibrosis from TFs
## Load the all the data to be used
dataset_names <- c("Govaere", "Kostrzewski", "Wang", "Feaver")
ref_dataset <- "Govaere"
target_dataset <- "Kostrzewski"
# Load
data_list <- load_datasets(dataset_names, dir_data = '../../FattyLiverModeling/data/')
# Run PCA
tmp <- process_datasets(data_list, filter_variance = F)
data_list <- tmp$data_list
plt_list <- tmp$plt_list
# Define matrices of interest
Yh <- as.matrix(data_list[[ref_dataset]]$metadata  %>% select(nas_score,Fibrosis_stage)) #keep both Fibrosis and NAS
colnames(Yh) <- c('NAS','fibrosis')
Xh <- data_list[[ref_dataset]]$data_center %>% t()
if (ref_dataset=='Hoang'){
  sex_inferred <- data_list[[ref_dataset]]$metadata$sex
}else if (ref_dataset=='Pantano') {
  sex_inferred <- data_list[[ref_dataset]]$metadata$Sex
}else{
  sex_inferred <- apply(as.matrix(Xh[,c('RPS4Y1')]),2,sign) 
  sex_inferred <- 1*(sex_inferred>0)
  sex_inferred <- ifelse(sex_inferred==1,'male','female')
}
# Get matrices of target dataset
Xm <- data_list[[target_dataset]]$data_center %>% t()
# Get Wm as the PC space of the MPS data when averaging tech replicates to capture variance due to experimental factors
Wm <- data_list[[target_dataset]]$Wm_group %>% as.matrix()

# Infer TF activity
dorotheaData = read.table('../../FattyLiverModeling/data/dorothea.tsv', sep = "\t", header=TRUE)
confidenceFilter = is.element(dorotheaData$confidence, c('A', 'B'))
dorotheaData = dorotheaData[confidenceFilter,]
colnames(dorotheaData)[1] <- 'source'
# Estimate TF activities
TF_human <- decoupleR::run_viper(t(Xh), dorotheaData,minsize = 5,verbose = TRUE)
TF_human <- TF_human %>% select(source,condition,score) %>% spread(source,score) %>% 
  column_to_rownames('condition')
TF_human <- as.matrix(TF_human)
## keep only TFs in LEMBAS
## train one model with all the data
plsr_model <- ropls::opls(x = TF_human, 
                   y = Yh,
                   predI = 8,
                   crossvalI = 1,
                   scaleC = "center",
                   fig.pdfC = "none",
                   info.txtC = "none")
plot(Yh[,1],ropls::predict(plsr_model,TF_human)[,1])
plot(Yh[,2],ropls::predict(plsr_model,TF_human)[,2])
print(diag(cor(Yh,ropls::predict(plsr_model,TF_human))))
W <- plsr_model@coefficientMN
W <- as.data.frame(W) 
W <- W %>% rownames_to_column('name')

## Keep only reachable TFs that exist in the PLSR model
reachable_tfs <- reachable_tfs %>% filter(name %in% colnames(TF_human))

## Compare rankings
all_tfs <- data.frame(name = colnames(TF_human))
all_tfs <- left_join(all_tfs,annotations)
all_tfs <- all_tfs %>%
  mutate(is_reachable = code %in% reachable_nodes)

tf_data <- all_tfs %>%
  left_join(W, by = "name") %>%
  mutate(NAS=abs(NAS)) %>%
  mutate(fibrosis = abs(fibrosis))
tf_data <- tf_data %>%
  mutate(
    NAS_rank = rank(-NAS, na.last = "keep"),
    fibrosis_rank = rank(-fibrosis, na.last = "keep")
  )

tfs_to_annotate <- c("TP63","TP53", "STAT2","STAT3", "FOXO4","SMAD3",
                     "FOS","E2F3","FOSL1","HIF1A","POU2F1","SMAD4","FOXO1","TWIST1",
                     "ATF1","GLI2","FOXO3","GATA2","TFDP1","RELA","E2F1")

tf_data <- tf_data %>%
  mutate(to_annotate = name %in% tfs_to_annotate)
p1 <- ggplot(tf_data, aes(x = NAS_rank, y = NAS)) +
  # Plot all non-reachable TFs
  geom_point(data = tf_data %>% filter(!is_reachable), 
             aes(color = "Non-reachable"), 
             size = 3, alpha = 0.5) +
  # Plot reachable TFs
  geom_point(data = tf_data %>% filter(is_reachable), 
             aes(color = "Reachable"), 
             size = 3, alpha = 0.7) +
  # Annotate specific TFs
  geom_text_repel(data = tf_data %>% filter(to_annotate),
                  aes(label = name),
                  size = 5,
                  fontface = "bold",
                  box.padding = 0.5,
                  point.padding = 0.5,
                  segment.color = "black",
                  segment.size = 1,
                  max.overlaps = 15) +
  scale_color_manual(values = c("Reachable" = "#E41A1C", "Non-reachable" = "#377EB8")) +
  labs(title = "TF Ranking by MAS Importance",
       subtitle = "Reachable from IFNAR1 and/or TGFBR1",
       x = "Rank (1 = highest importance)",
       y = "Absolute Importance (NAS)",
       color = "TF Status") +
  theme_bw() +
  theme(legend.position = "top",
        plot.title = element_text(size = 14, face = "bold"),
        axis.title = element_text(size = 12),
        legend.text = element_text(size = 11))
print(p1)
ggsave("../figures/NAS_TF_ranking.png", p1, width = 10, height = 8, dpi = 600)

p2 <- ggplot(tf_data, aes(x = fibrosis_rank, y = fibrosis)) +
  # Plot all non-reachable TFs
  geom_point(data = tf_data %>% filter(!is_reachable), 
             aes(color = "Non-reachable"), 
             size = 3, alpha = 0.5) +
  # Plot reachable TFs
  geom_point(data = tf_data %>% filter(is_reachable), 
             aes(color = "Reachable"), 
             size = 3, alpha = 0.7) +
  # Annotate specific TFs
  geom_text_repel(data = tf_data %>% filter(to_annotate),
                  aes(label = name),
                  size = 5,
                  fontface = "bold",
                  box.padding = 0.5,
                  point.padding = 0.5,
                  segment.color = "black",
                  segment.size = 1,
                  max.overlaps = 15) +
  scale_color_manual(values = c("Reachable" = "#E41A1C", "Non-reachable" = "#377EB8")) +
  labs(title = "TF Ranking by Fibrosis Importance",
       subtitle = "Reachable from IFNAR1 and/or TGFBR1",
       x = "Rank (1 = highest importance)",
       y = "Absolute Importance (Fibrosis)",
       color = "TF Status") +
  theme_bw() +
  theme(legend.position = "top",
        plot.title = element_text(size = 14, face = "bold"),
        axis.title = element_text(size = 12),
        legend.text = element_text(size = 11))
print(p2)
ggsave("../figures/Fibrosis_TF_ranking.png", p2, width = 10, height = 8, dpi = 600)

