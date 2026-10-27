library(tidyverse)
library(dorothea)
library(ggplot2)
library(ggpubr)
library(ggsignif)
library(patchwork)
library(ropls)
source("plotting_functions.R")
source('functions_translation.R')
library(edgeR)

### Load bulk RNA counts --------------
gene_annotation <- readRDS('../data/RNAseq/gene_annotation.rds')
custom_counts <- read.delim(
  "../data/RNAseq/SHGXH5_analysis/results/SHGXH5_counts_matrix.tsv",
  header = TRUE,
  sep = "\t",
  check.names = FALSE
)
cols_to_keep_custom <- grep("gene_name|gene_id|gene_biotype|SHGX", colnames(custom_counts), ignore.case = TRUE, value = TRUE)
# Remove sequencing order prefix from custom sample names
colnames(custom_counts) <- sub(
  "^SHGXH5_[0-9]+_",
  "",
  colnames(custom_counts)
)

### TMM normalization using egdeR ###
raw_counts <- as.matrix(
  custom_counts[, setdiff(colnames(custom_counts), "gene_id")]
)
storage.mode(raw_counts) <- "numeric"
rownames(raw_counts) <- custom_counts$gene_id

# 1. Define experimental groups from sample names
groups <- factor(
  sub("_[0-9]+$", "",
      sub("^SHGXH5_[0-9]+_", "", colnames(raw_counts)))
)
# 2. Create edgeR object
dge <- DGEList(
  counts = raw_counts,
  group = groups
)
# 3. Filter low-expression genes
keep <- filterByExpr(
  dge,
  group = groups,
  min.count = 10,
  min.total.count = 15
)
# 4. Apply filtering
dge <- dge[keep, , keep.lib.sizes = FALSE]
# 5. Extract filtered count matrix
raw_counts_filtered <- dge$counts
# 6. Summary
cat("Original genes:", nrow(raw_counts), "\n")
cat("Retained genes:", nrow(raw_counts_filtered), "\n")
cat("Removed genes:", sum(!keep), "\n")
cat("Percentage retained:",
    round(100 * mean(keep), 2), "%\n")

# Proceed with TMM normalization
dge_tmm <- calcNormFactors(dge, method = "TMM")
# dge_wsp <- calcNormFactors(
#   dge,
#   method = "TMMwsp"
# )
# Compute TMM-normalized CPM
cpm_tmm <- cpm(
  dge_tmm,
  log = FALSE,
  normalized.lib.sizes = TRUE
)
cpm_tmm <-  left_join(as.data.frame(cpm_tmm) %>% rownames_to_column('gene_id'),gene_annotation %>% select(-gene_biotype)) %>%
  mutate(gene_name=ifelse(gene_name=="",gene_id,gene_name))
cpm_tmm <- aggregate(cpm_tmm[,2:(ncol(cpm_tmm)-1)],by = list(cpm_tmm$gene_name),FUN = sum) %>%
  column_to_rownames('Group.1')
# cpm_wsp <- cpm(dge_wsp, log = FALSE,normalized.lib.sizes = TRUE)
# cpm_wsp <-  left_join(as.data.frame(cpm_wsp) %>% rownames_to_column('gene_id'),gene_annotation %>% select(-gene_biotype)) %>%
#   mutate(gene_name=ifelse(gene_name=="",gene_id,gene_name))
# cpm_wsp <- aggregate(cpm_wsp[,2:(ncol(cpm_wsp)-1)],by = list(cpm_wsp$gene_name),FUN = sum) %>%
#   column_to_rownames('Group.1')

logcpm_tmm <- log10(cpm_tmm + 1)

# Get standarized, aggregated and normalized gene expression
gex <- t(logcpm_tmm - rowMeans(logcpm_tmm))

## Do a replicate check
expression_matrices <- list(
  CPM_TMM = as.matrix(cpm_tmm),
  logCPM_TMM = as.matrix(logcpm_tmm)
)
# Ensure comparable gene IDs and samples
common_genes <- Reduce(
  intersect,
  lapply(expression_matrices, rownames)
)
common_samples <- Reduce(
  intersect,
  lapply(expression_matrices, colnames)
)
expression_matrices <- lapply(
  expression_matrices,
  function(m) {
    m <- m[common_genes, common_samples, drop = FALSE]
    storage.mode(m) <- "numeric"
    m
  }
)
# Extract condition by removing final replicate number
conditions <- sub("_[0-9]+$", "", common_samples)

# Group samples by condition
replicate_groups <- split(common_samples, conditions)
# Keep only conditions with at least 2 replicates
replicate_groups <- Filter(
  function(x) length(x) >= 2,
  replicate_groups
)
print(replicate_groups)
# Generate all replicate pairs within each condition
replicate_pairs <- do.call(
  rbind,
  lapply(names(replicate_groups), function(group) {
    
    pairs <- t(combn(replicate_groups[[group]], 2))
    
    data.frame(
      group = group,
      sample1 = pairs[, 1],
      sample2 = pairs[, 2]
    )
  })
)
rownames(replicate_pairs) <- NULL
print(replicate_pairs)

outdir <- "../data/RNAseq/SHGXH5_analysis/results"
dir.create(
  outdir,
  recursive = TRUE,
  showWarnings = FALSE
)
correlation_stats <- list()
k <- 1
set.seed(42)
pdf(
  file.path(outdir, "replicate_scatterplots_3transformations.pdf"),
  width = 17,
  height = 6
)
for (i in seq_len(nrow(replicate_pairs))) {
  
  group <- replicate_pairs$group[i]
  s1 <- replicate_pairs$sample1[i]
  s2 <- replicate_pairs$sample2[i]
  
  # Select expressed genes using CPM
  keep <- (
    expression_matrices$CPM_TMM[, s1] >= 1 |
      expression_matrices$CPM_TMM[, s2] >= 1
  )
  
  par(
    mfrow = c(1, 2),
    mar = c(5, 5, 4, 1),
    oma = c(0, 0, 2, 0)
  )
  
  for (type in names(expression_matrices)) {
    
    mat <- expression_matrices[[type]]
    
    x <- mat[keep, s1]
    y <- mat[keep, s2]
    
    valid <- is.finite(x) & is.finite(y)
    
    x <- x[valid]
    y <- y[valid]
    
    # Correlations calculated using ALL selected genes
    pearson <- if (
      length(x) >= 3 && sd(x) > 0 && sd(y) > 0
    ) cor(x, y, method = "pearson") else NA_real_
    
    spearman <- if (
      length(x) >= 3 && sd(x) > 0 && sd(y) > 0
    ) cor(x, y, method = "spearman") else NA_real_
    
    correlation_stats[[k]] <- data.frame(
      group = group,
      sample1 = s1,
      sample2 = s2,
      transformation = type,
      n_genes = length(x),
      pearson = pearson,
      spearman = spearman
    )
    
    k <- k + 1
    
    # Subsample points for plotting only
    # Correlations above still use every selected gene
    idx <- if (length(x) > 15000) {
      sample.int(length(x), 15000)
    } else {
      seq_along(x)
    }
    
    lim <- range(c(x, y), finite = TRUE)
    
    if (diff(lim) == 0) {
      lim <- lim + c(-1, 1)
    }
    
    plot(
      x[idx], y[idx],
      pch = 16,
      cex = 0.35,
      col = adjustcolor("steelblue", alpha.f = 0.2),
      xlab = s1,
      ylab = s2,
      main = type,
      xlim = lim,
      ylim = lim,
      asp = 1
    )
    
    abline(
      a = 0,
      b = 1,
      col = "red",
      lwd = 1.5
    )
    
    legend(
      "topleft",
      legend = c(
        sprintf("Pearson = %.3f", pearson),
        sprintf("Spearman = %.3f", spearman),
        paste("Genes =", length(x))
      ),
      bty = "n",
      cex = 0.85
    )
  }
  
  mtext(
    paste(group, "|", s1, "vs", s2),
    outer = TRUE,
    cex = 1.1,
    font = 2
  )
}
dev.off()
# Combine statistics
correlation_results <- do.call(
  rbind,
  correlation_stats
)
rownames(correlation_results) <- NULL

### Run PCA and see where each replicate falls

# remove TGFB2, both NCS2 and both SIS3
X <- gex[which(!(rownames(gex) %in% c('TGFb_2','Both_NSC_2','Both_SIS3_2'))),]
pca <- prcomp(
  X,
  center = TRUE,
  scale. = FALSE
)
# Percentage of total variance explained
variance_explained <- 100 * pca$sdev^2 / sum(pca$sdev^2)
# Extract PCA coordinates
pca_df <- as.data.frame(pca$x)
pca_df$sample <- rownames(pca_df)
# Remove SHGXH5 sequencing prefix, if present
pca_df$sample <- sub(
  "^SHGXH5_[0-9]+_", "",
  pca_df$sample
)
# Experimental condition
pca_df$condition <- sub(
  "_[0-9]+$", "",
  pca_df$sample
)
# Biological replicate number
pca_df$replicate <- sub(
  "^.*_([0-9]+)$", "\\1",
  pca_df$sample
)

p_pca <- ggplot(
  pca_df,
  aes(
    x = PC1,
    y = PC2,
    color = condition
  )
) +
  geom_point(
    size = 3.5,
    alpha = 0.9
  ) +
  geom_text_repel(
    aes(label = replicate),
    size = 3.5,
    max.overlaps = Inf,
    show.legend = FALSE
  ) +
  labs(
    title = "PCA of RNA-seq expression",
    subtitle = "TMM-normalized, log2-transformed, gene-centered",
    x = sprintf("PC1 (%.2f%% variance)", variance_explained[1]),
    y = sprintf("PC2 (%.2f%% variance)", variance_explained[2]),
    color = "Treatment"
  ) +
  coord_equal() +
  theme_classic(base_size = 13) +
  theme(
    legend.position = "right",
    plot.title = element_text(face = "bold")
  )

print(p_pca)


### Infer TF activity----------------
dorotheaData = read.table('../data/dorothea.tsv', sep = "\t", header=TRUE)
confidenceFilter = is.element(dorotheaData$confidence, c('A', 'B'))
dorotheaData = dorotheaData[confidenceFilter,]
# Estimate TF activities
settings = list(verbose = TRUE, minsize = 5)
TF_activities = run_viper(t(X), dorotheaData, options =  settings)
tfs <- rownames(TF_activities)
tf_mat <- TF_activities
TF_activities <- as.data.frame(TF_activities) %>% rownames_to_column('TF') %>% gather(key='condition',value='activity',-TF)
TF_activities <- TF_activities %>%
  separate('condition', into=c('condition','repID'),
           sep = "_(?=[0-9]+$)")

### replicate correlation
X_TFs <- t(tf_mat)
# Remove sequencing prefix, if present
sample_names <- sub(
  "^SHGXH5_[0-9]+_", "",
  rownames(X_TFs)
)
rownames(X_TFs) <- sample_names
# Identify experimental conditions
conditions <- sub("_[0-9]+$", "", sample_names)
# Group replicates
replicate_groups <- split(sample_names, conditions)
# Retain groups containing at least two replicates
replicate_groups <- Filter(
  function(x) length(x) >= 2,
  replicate_groups
)
print(replicate_groups)

# Generate all replicate pairs
replicate_pairs <- do.call(
  rbind,
  lapply(names(replicate_groups), function(group) {
    
    pairs <- t(combn(replicate_groups[[group]], 2))
    
    data.frame(
      group = group,
      sample1 = pairs[, 1],
      sample2 = pairs[, 2]
    )
  })
)
rownames(replicate_pairs) <- NULL

outdir <- "../data/RNAseq/SHGXH5_analysis/results"
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

tf_cor_results <- list()
tf_plots <- list()

for (i in seq_len(nrow(replicate_pairs))) {
  
  group <- replicate_pairs$group[i]
  s1 <- replicate_pairs$sample1[i]
  s2 <- replicate_pairs$sample2[i]
  
  # Extract TF activities
  x <- as.numeric(X_TFs[s1, ])
  y <- as.numeric(X_TFs[s2, ])
  
  # Remove missing values
  valid <- is.finite(x) & is.finite(y)
  
  x <- x[valid]
  y <- y[valid]
  tf_names <- colnames(X_TFs)[valid]
  
  # Correlation
  pearson <- if (length(x) >= 3 &&
                 sd(x) > 0 && sd(y) > 0) {
    cor(x, y, method = "pearson")
  } else NA_real_
  
  spearman <- if (length(x) >= 3 &&
                  sd(x) > 0 && sd(y) > 0) {
    cor(x, y, method = "spearman")
  } else NA_real_
  
  # Additional agreement metrics
  rmse <- sqrt(mean((x - y)^2))
  median_abs_diff <- median(abs(x - y))
  
  tf_cor_results[[i]] <- data.frame(
    group = group,
    sample1 = s1,
    sample2 = s2,
    n_TFs = length(x),
    pearson = pearson,
    spearman = spearman,
    RMSE = rmse,
    median_abs_diff = median_abs_diff
  )
  
  df <- data.frame(
    TF = tf_names,
    replicate1 = x,
    replicate2 = y
  )
  
  p <- ggplot(df, aes(replicate1, replicate2)) +
    geom_point(
      alpha = 0.4,
      size = 1.2,
      color = "steelblue"
    ) +
    geom_abline(
      slope = 1,
      intercept = 0,
      color = "red",
      linewidth = 0.7
    ) +
    coord_equal() +
    labs(
      title = paste0(group, ": ", s1, " vs ", s2),
      subtitle = sprintf(
        "Pearson = %.3f | Spearman = %.3f | RMSE = %.3f",
        pearson, spearman, rmse
      ),
      x = paste(s1, "TF activity"),
      y = paste(s2, "TF activity")
    ) +
    theme_classic(base_size = 12)
  
  tf_plots[[i]] <- p
}

# Combine results
tf_cor_summary <- do.call(rbind, tf_cor_results)

print(tf_cor_summary)

# Multipage PDF: one plot per replicate pair
pdf(
  file.path(outdir, "TF_replicate_scatterplots.pdf"),
  width = 7,
  height = 6
)

for (p in tf_plots) {
  print(p)
}

dev.off()

# Show an example
print(tf_plots[[1]])

### Visualize TP63, SMAD3, STAT2, STAT3----------
ggdotplot(TF_activities %>% filter(TF %in% c('TP63', 
                                             'SMAD3', 
                                             'STAT2', 
                                             'STAT3')),
          x='condition',y='activity',fill='condition') +
  facet_wrap(~TF,scales = 'free_y')+ 
  theme(legend.position = 'none')+
  geom_hline(yintercept = 0,linetype='dashed',lwd=1)
ggsave(filename = '../figures/RNA_TFact_validation_healthyMedia.png',width = 9,height = 9, units = 'in')


### Load previous data and use PLSR to score phenotype-----------
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

### Intersection of current and previous gene expression
# common_genes <- unique(intersect(colnames(Xh),colnames(gex)))

TF_human <- run_viper(t(Xh), dorotheaData, options =  settings)
TF_human <-t(TF_human)
TF_human <- TF_human[,tfs]

## train one model with all the data
plsr_model <- opls(x = TF_human, #Xh[,common_genes], 
                   y = Yh,
                   predI = 8,
                   crossvalI = 1,
                   scaleC = "center",
                   fig.pdfC = "none",
                   info.txtC = "none")
yhat <- as.data.frame(predict(plsr_model,t(tf_mat)[,tfs]))
# yhat <- as.data.frame(predict(plsr_model,gex[,common_genes]))
yhat <- yhat %>%
  rownames_to_column('condition') %>%
  separate('condition', into=c('condition','repID'),
           sep = "_(?=[0-9]+$)")
yhat <- yhat %>% gather('phenotype','score',-condition,-repID)

ggdotplot(yhat,
          x='condition',y='score',fill='condition') +
  facet_wrap(~phenotype,scales = 'free_y')+ 
  theme(legend.position = 'none')+
  geom_hline(yintercept = 0,linetype='dashed',lwd=1)
ggsave(filename = '../figures/RNA_PhenoScore_validation.png',width = 9,height = 9, units = 'in')



### Replicate correlations: TF activities, phenotypes, gene expression ----

# Reconstruct phenotype matrix: phenotypes x samples
phenotype_mat <- yhat %>%
  mutate(sample = paste(condition, repID, sep = "_")) %>%
  select(sample, phenotype, score) %>%
  pivot_wider(names_from = phenotype, values_from = score) %>%
  column_to_rownames("sample") %>%
  as.matrix() %>%
  t()

# All three matrices: features x samples
feature_mats <- list(
  "TF activities" = as.matrix(tf_mat),
  "Phenotypes" = phenotype_mat,
  "Gene expression (log10 CPM)" = as.matrix(log_cpm)
)

# Samples shared by all three analyses
common_samples <- Reduce(intersect, lapply(feature_mats, colnames))

if (length(common_samples) == 0) {
  stop("No matching sample names across the three matrices.")
}

# Identify conditions by removing only the final replicate suffix
conditions <- unique(sub("_[0-9]+$", "", common_samples))

dir.create("../figures", recursive = TRUE, showWarnings = FALSE)

for (cond in conditions) {
  
  # Find replicates belonging to this condition
  samples <- common_samples[
    sub("_[0-9]+$", "", common_samples) == cond
  ]
  
  # Order replicates numerically
  samples <- samples[
    order(as.integer(sub("^.*_([0-9]+)$", "\\1", samples)))
  ]
  
  if (length(samples) < 2) next
  
  # All unique replicate pairs
  sample_pairs <- combn(samples, 2, simplify = FALSE)
  
  plots <- list()
  
  # Three rows: TFs, phenotypes, genes
  for (metric in names(feature_mats)) {
    
    mat <- feature_mats[[metric]]
    
    for (pair in sample_pairs) {
      
      # Each row of mat represents one TF, phenotype, or gene
      df <- data.frame(
        x = as.numeric(mat[, pair[1]]),
        y = as.numeric(mat[, pair[2]])
      )
      
      # Remove missing/non-finite values
      df <- df[is.finite(df$x) & is.finite(df$y), , drop = FALSE]
      
      n <- nrow(df)
      
      # Pearson correlation
      r <- if (n >= 2 && sd(df$x) > 0 && sd(df$y) > 0) {
        cor(df$x, df$y, method = "pearson")
      } else {
        NA_real_
      }
      
      r_label <- if (is.finite(r)) {
        sprintf("Pearson r = %.3f | n = %d", r, n)
      } else {
        sprintf("Pearson r = NA | n = %d", n)
      }
      
      # Only two phenotypes: correlation is not informative
      if (metric == "Phenotypes") {
        r_label <- paste0(r_label, "\n(n = 2: interpret cautiously)")
      }
      
      rep_labels <- paste0(
        "Rep ",
        sub("^.*_([0-9]+)$", "\\1", pair)
      )
      
      # Smaller, transparent points for gene-expression plots
      point_size <- if (metric == "Gene expression (log10 CPM)") 0.45 else 2
      point_alpha <- if (metric == "Gene expression (log10 CPM)") 0.3 else 0.8
      
      p <- ggplot(df, aes(x = x, y = y)) +
        geom_abline(
          slope = 1, intercept = 0,
          linetype = "dashed", color = "grey60"
        ) +
        geom_point(
          size = point_size,
          alpha = point_alpha
        ) +
        coord_equal() +
        labs(
          title = paste(metric, "-", paste(rep_labels, collapse = " vs ")),
          subtitle = r_label,
          x = pair[1],
          y = pair[2]
        ) +
        theme_bw(base_size = 10) +
        theme(
          plot.title = element_text(face = "bold", size = 10),
          plot.subtitle = element_text(size = 9),
          axis.title = element_text(size = 9)
        )
      
      plots[[length(plots) + 1]] <- p
    }
  }
  
  # Arrange three rows and one column per replicate pair
  combined_plot <- wrap_plots(
    plots,
    ncol = length(sample_pairs),
    byrow = TRUE
  ) +
    plot_annotation(
      title = paste("Replicate correlations:", cond),
      subtitle = "Pearson correlation; dashed line represents y = x"
    )
  
  # Save one panel per condition
  ggsave(
    filename = file.path(
      "../figures",
      paste0("RNA_ReplicateCorrelations_", cond, ".png")
    ),
    plot = combined_plot,
    width = 4.7 * length(sample_pairs),
    height = 11,
    units = "in",
    dpi = 300,
    limitsize = FALSE
  )
  
  print(combined_plot)
}


#### For TMM
# library(edgeR)
# 
# y <- DGEList(counts = counts)
# 
# # Filter low-expression genes
# keep <- filterByExpr(y, group = conditions)
# y <- y[keep, , keep.lib.sizes = FALSE]
# 
# # TMM normalization
# y <- calcNormFactors(y, method = "TMM")
# 
# # TMM-normalized log2-CPM
# logCPM <- cpm(y, log = TRUE, prior.count = 2)