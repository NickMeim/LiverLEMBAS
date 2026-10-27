library(tidyverse)
library(dorothea)
library(ggplot2)
library(ggpubr)
library(ggsignif)
library(patchwork)
library(ropls)
source("plotting_functions.R")
source('functions_translation.R')

### Load bulk RNA counts --------------
zip_path <- '../data/RNAseq/SHGXH5_results.zip'
file_in_zip <- "SHGXH5-expression-matrix.tsv"
sample_map <- c(
  "SHGXH5_1_count"  = "Both_DMSO_1",
  "SHGXH5_2_count"  = "Both_DMSO_2",
  "SHGXH5_3_count"  = "Both_DMSO_3",
  "SHGXH5_4_count"  = "Both_DEU_1",
  "SHGXH5_5_count"  = "Both_DEU_2",
  "SHGXH5_6_count"  = "Both_DEU_3",
  "SHGXH5_7_count"  = "Both_DPP_1",
  "SHGXH5_8_count"  = "Both_DPP_2",
  "SHGXH5_9_count"  = "Both_DPP_3",
  "SHGXH5_10_count" = "Both_SIS3_1",
  "SHGXH5_11_count" = "Both_SIS3_2",
  "SHGXH5_12_count" = "Both_SIS3_3",
  "SHGXH5_13_count" = "Both_NSC_1",
  "SHGXH5_14_count" = "Both_NSC_2",
  "SHGXH5_15_count" = "Both_NSC_3",
  "SHGXH5_16_count" = "TGFb_1",
  "SHGXH5_17_count" = "TGFb_2",
  "SHGXH5_18_count" = "TGFb_3",
  "SHGXH5_19_count" = "IFNa_1",
  "SHGXH5_20_count" = "IFNa_2",
  "SHGXH5_21_count" = "DMSO_1",
  "SHGXH5_22_count" = "DMSO_2",
  "SHGXH5_23_count" = "DMSO_3"
)
# Base R
counts <- read.delim(
  file = unz(zip_path, file_in_zip),
  header = TRUE,
  sep = "\t",
  check.names = FALSE
)

counts <- counts %>% mutate(gene_name=ifelse(gene_name=="",gene_id,gene_name))

# Case-insensitive matching (matches "cpm", "CPM", etc.)
cols_to_keep <- grep("gene_name|gene_id|gene_biotype|count", colnames(counts), ignore.case = TRUE, value = TRUE)
# Subset columns
counts <- counts[, cols_to_keep, drop = FALSE]
counts <- counts %>% select(-gene_name) %>%
  column_to_rownames('gene_id')
colnames(counts) <- ifelse(
  colnames(counts) %in% names(sample_map),
  sample_map[colnames(counts)],
  colnames(counts)
)
counts <- counts %>% select(-gene_biotype)
counts <- counts %>% rownames_to_column('gene_id')

# Newly generated featureCounts matrix
custom_counts <- read.delim(
  "../data/RNAseq/SHGXH5_analysis/results/SHGXH5_counts_matrix.tsv",
  header = TRUE,
  sep = "\t",
  check.names = FALSE
)
cols_to_keep_custom <- grep("gene_name|gene_id|gene_biotype|SHGX", colnames(custom_counts), ignore.case = TRUE, value = TRUE)
# Inspect columns
head(colnames(counts), 15)
head(colnames(custom_counts), 15)

# Original Plasmidsaurus: counts
# Our featureCounts: custom_counts

# Remove sequencing order prefix from custom sample names
colnames(custom_counts) <- sub(
  "^SHGXH5_[0-9]+_",
  "",
  colnames(custom_counts)
)

# Identify common samples
common_samples <- intersect(
  setdiff(colnames(counts), "gene_id"),
  setdiff(colnames(custom_counts), "gene_id")
)

print(common_samples)
cat("Common samples:", length(common_samples), "\n")

# Match gene IDs, removing Ensembl version suffixes if present
original_ids <- sub("\\.[0-9]+$", "", counts$gene_id)
custom_ids <- sub("\\.[0-9]+$", "", custom_counts$gene_id)

common_genes <- intersect(original_ids, custom_ids)

idx_original <- match(common_genes, original_ids)
idx_custom <- match(common_genes, custom_ids)

cat("Common genes:", length(common_genes), "\n")

### Plot agreement between mine and their analysis
outdir <- "../data/RNAseq/SHGXH5_analysis/results"

dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

plots <- list()
stats <- list()

set.seed(42)

for (s in common_samples) {
  
  # Match counts by gene ID
  x <- as.numeric(counts[[s]][idx_original])
  y <- as.numeric(custom_counts[[s]][idx_custom])
  
  # Keep valid nonnegative counts
  valid <- is.finite(x) & is.finite(y) & x >= 0 & y >= 0
  x <- x[valid]
  y <- y[valid]
  
  # Log-transform raw counts
  lx <- log10(x + 1)
  ly <- log10(y + 1)
  
  # Genes expressed in at least one dataset
  expressed <- x > 0 | y > 0
  
  # Correlations
  pearson_all <- cor(lx, ly, method = "pearson")
  pearson_expr <- cor(lx[expressed], ly[expressed],
                      method = "pearson")
  
  spearman_all <- cor(x, y, method = "spearman")
  spearman_expr <- cor(x[expressed], y[expressed],
                       method = "spearman")
  
  # Agreement statistics
  exact_match <- mean(abs(x - y) < 1e-6) * 100
  exact_expr <- mean(abs(x[expressed] -
                           y[expressed]) < 1e-6) * 100
  
  stats[[s]] <- data.frame(
    sample = s,
    n_genes = length(x),
    n_expressed = sum(expressed),
    pearson_all = pearson_all,
    pearson_expressed = pearson_expr,
    spearman_all = spearman_all,
    spearman_expressed = spearman_expr,
    exact_match_percent = exact_match,
    exact_expressed_percent = exact_expr,
    original_total = sum(x),
    custom_total = sum(y),
    total_ratio = sum(y) / sum(x)
  )
  
  # Plot only genes expressed in at least one dataset
  df <- data.frame(
    original = lx[expressed],
    custom = ly[expressed]
  )
  
  # Limit plotted points for manageable PDF sizes.
  # Correlations above use ALL genes.
  if (nrow(df) > 20000) {
    df <- df[sample.int(nrow(df), 20000), ]
  }
  
  p <- ggplot(df, aes(original, custom)) +
    geom_point(alpha = 0.15, size = 0.5) +
    geom_abline(
      slope = 1,
      intercept = 0,
      color = "red",
      linewidth = 0.7
    ) +
    coord_equal() +
    labs(
      title = s,
      subtitle = sprintf(
        "Pearson = %.4f | Spearman = %.4f | Exact = %.1f%%",
        pearson_expr, spearman_expr, exact_expr
      ),
      x = "Plasmidsaurus log10(raw count + 1)",
      y = "Custom featureCounts log10(raw count + 1)"
    ) +
    theme_classic(base_size = 12)
  
  plots[[s]] <- p
}

# Summary table
correlation_results <- do.call(rbind, stats)
rownames(correlation_results) <- NULL

print(correlation_results)

# Save statistics
write.csv(
  correlation_results,
  file.path(outdir, "count_comparison_statistics.csv"),
  row.names = FALSE
)

# Save all scatterplots
pdf(
  file.path(outdir, "count_comparison_scatterplots.pdf"),
  width = 7,
  height = 6
)

for (p in plots) print(p)

dev.off()

# Display example
print(plots[["Both_DMSO_1"]])