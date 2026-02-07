library(tidyverse)
library(DESeq2)
library(matrixStats)
library(irlba)   # fast PCA for large matrices
library(uwot)    # UMAP
library(decoupleR)

## Load data---------
metadata <- data.table::fread('ARCHS4_RESULTS/archs4_all_liver_related_samples.csv')
counts <- readRDS('ARCHS4_RESULTS/expression_transposed.rds') 
gc()

meta <- as.data.frame(metadata)
stopifnot("sample_id" %in% colnames(meta))
meta$sample_id <- unname(as.character(meta$sample_id))
rownames(meta) <- meta$sample_id
meta$sample_id <- NULL

if (!all(rownames(meta) %in% colnames(counts)) && all(rownames(meta) %in% rownames(counts))) {
  message("Transposing count matrix so that columns are samples...")
  counts <- t(counts)
}
rownames(counts) <- unname(as.character(rownames(counts)))
colnames(counts) <- unname(as.character(colnames(counts)))
rownames(counts) <- make.unique(rownames(counts))
counts <- round(counts)

dds <- DESeqDataSetFromMatrix(
  countData = counts,
  colData   = meta,
  design    = ~ 1
)
gc()

# -------------------------
# Step 1) Robust normalization (median-of-ratios size factors)
# -------------------------
dds <- estimateSizeFactors(dds, type = "poscounts")
norm_counts <- counts(dds, normalized = TRUE)
log_norm <- log2(norm_counts + 1)
gc()

# -------------------------
# Step 2) Filter low-expressed genes (avg log-scale threshold)
# -------------------------
keep <- rowSums(log_norm) >= 100
dds <- dds[keep, ]

# Recompute size factors after filtering
dds <- estimateSizeFactors(dds, type = "poscounts")
norm_counts <- counts(dds, normalized = TRUE)
# Log-transform for downstream single-sample activity inference
log_norm <- log2(norm_counts + 1)
gc()

## Make UMAP of expression of genes---------
set.seed(42)
n_var_genes <- 10000
vars <- matrixStats::rowVars(log_norm)
top_idx <- order(vars, decreasing = TRUE)[seq_len(min(n_var_genes, length(vars)))]

# samples x genes matrix for PCA/UMAP
X <- t(log_norm[top_idx, , drop = FALSE])
# X <- t(log_norm)

# truncated PCA (fast)
n_pcs <- 50
pca <- irlba::prcomp_irlba(X, n = n_pcs, center = TRUE, scale. = FALSE)

# UMAP on first k PCs (common choice: 20–50)
k_umap <- 30
um <- uwot::umap(pca$x[, seq_len(min(k_umap, ncol(pca$x))), drop = FALSE],
                 n_neighbors = 30,
                 min_dist = 0.3,
                 metric = "cosine",
                 verbose = TRUE)

um_df <- tibble(
  sample_id = rownames(pca$x),
  UMAP1 = um[, 1],
  UMAP2 = um[, 2]
)

## 2) Gather expression for your genes & plot in facets
genes <- c("TP63","TP53","SMAD3","STAT2","STAT3","ALB","CYP3A4","IFNAR1","TGFBR1")
present <- intersect(genes, rownames(log_norm))
missing <- setdiff(genes, rownames(log_norm))
if (length(missing) > 0) message("Missing genes (not found in rownames(log_norm)): ", paste(missing, collapse = ", "))

expr_long <- as_tibble(scale(t(log_norm[present, , drop = FALSE]),scale = FALSE,center = TRUE), 
                       rownames = "sample_id") %>%
  pivot_longer(cols = -sample_id, names_to = "gene", values_to = "expr")

plot_df <- cbind(um_df,expr_long)
# plot_df <- left_join(plot_df,metadata)

p <- ggplot(plot_df, aes(UMAP1, UMAP2, color = expr)) +
  geom_point(size = 0.4, alpha = 0.8) +
  facet_wrap(~ gene, ncol = 3, scales = "fixed") +
  # scale_color_viridis_c() +
  scale_color_gradient2()+
  coord_equal() +
  theme_bw(base_size = 11) +
  theme(
    panel.grid = element_blank(),
    strip.background = element_rect(fill = "grey95", color = NA)
  ) +
  # labs(color = "log2(norm+1)")
  labs(color = "Centered log2(norm+1)")

print(p)


### Drop some genes like we do with Jose for example
### Get only dorothea that I used
### run viper with similar parameters
### center my data

## Infer TF activity----------------------------
# 1) Get a signed TF-target network (CollecTRI) — default keeps complexes together
net <- decoupleR::get_collectri(organism = "human", split_complexes = FALSE)

# 2) Optional but recommended for speed/memory: keep only genes that appear as targets
targets <- unique(net$target)
mat <- log_norm[intersect(rownames(log_norm), targets), , drop = FALSE]

# 3) Run VIPER per sample
# method = "scale" (z-score per gene across samples) OR "rank" (rank-based, more robust)
acts <- decoupleR::run_viper(
  mat = mat,
  net = net,
  .source = "source",
  .target = "target",
  .mor    = "mor",
  minsize = 10,
  method  = "rank"   # try "scale" too;
)

# acts typically contains: statistic, source (TF), condition (sample), score, p_value
# Make a samples x TF matrix of activity scores
acts_wide <- acts %>%
  select(condition, source, score) %>%
  pivot_wider(names_from = source, values_from = score)

# If you want TFs x samples:
acts_mat <- acts_wide %>%
  column_to_rownames("condition") %>%
  as.matrix()

## Visualize in a UMAP performed on TFs activities
