library(tidyverse)
library(igraph)

## Load the resulted network from the Partial Correlation pathway analysis
pps_net <- data.table::fread('PPS_results/8hours_edges_after_PPS_for_k_of_3.csv')

## Load fine with symbol-uniprot annotation
annot <- read.delim('../../../Artificial-Signaling-Network/DrugsANNSignaling/data/uniprot-reviewed_yes+AND+organism__Homo+sapiens+(Human)+[9606]_.tab')
annot <- annot %>% select(c('node'='Entry'),c('node_symbol'='Gene.names...primary..')) %>% unique()
annot <- annot %>% filter(node %in% c(pps_net$source,pps_net$target))
annot <- annot %>% unique()
annot <- rbind(annot,
               data.frame(node=c('fibrosis','NAS'),
                          node_symbol=c('fibrosis','MAS')))

## Merge so that I have symbols for my nodes
pps_net_2 <- left_join(annot %>% select(c('source'='node'),c('Source_symbol'='node_symbol')),
                       pps_net)
pps_net_2 <- left_join(annot %>% select(c('target'='node'),c('Target_symbol'='node_symbol')),
                       pps_net_2)

## Filter based on frequency (keep nodes appearing at least or more than a frequency threshold)
th <- 0.2 * 50
pps_net_2 <- pps_net_2 %>% filter(frequency>=th)

## Remove disconnected parts. I want fully connected paths from IFNAR1 and TGFBR1 to MAS and fibrosis
df2 <- pps_net_2 %>% mutate(.row_id = row_number())
g <- graph_from_data_frame(
  d = df2 %>% select(Source_symbol, Target_symbol, everything()),
  directed = TRUE
)

edges_between <- function(g, start, end) {
  # Returns original row ids (.row_id) for edges in the maximal subgraph that can connect start -> end
  if (!(start %in% V(g)$name) || !(end %in% V(g)$name)) return(integer(0))
  
  s <- match(start, V(g)$name)
  t <- match(end,   V(g)$name)
  
  # nodes reachable from start, and nodes that can reach end
  fwd <- subcomponent(g, s, mode = "out")
  bwd <- subcomponent(g, t, mode = "in")
  
  keep_nodes <- intersect(fwd, bwd)
  if (length(keep_nodes) == 0) return(integer(0))
  
  keep_names <- V(g)$name[keep_nodes]
  
  as_data_frame(g, what = "edges") %>%
    filter(from %in% keep_names, to %in% keep_names) %>%
    pull(.row_id) %>%
    unique()
}

keep_ids <- Reduce(union, list(
  edges_between(g, "IFNAR1", "fibrosis"),
  edges_between(g, "IFNAR1", "MAS"),
  edges_between(g, "TGFBR1", "fibrosis"),
  edges_between(g, "TGFBR1", "MAS")
))

df_kept <- df2 %>%
  filter(.row_id %in% keep_ids) %>%
  select(-.row_id)
df_kept <- df_kept %>% mutate(interaction=ifelse(avg_weight>0,"positive",
                                                 ifelse(avg_weight<0,"negative","unknown")))

## Save the frequecy filtered network
data.table::fwrite(df_kept,paste0('PPS_results/aggregated_edges_top1_8hours_freq',th,'.csv'))
