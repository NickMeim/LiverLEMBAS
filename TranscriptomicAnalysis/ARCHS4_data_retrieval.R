# Discover PRIMARY HUMAN HEPATOCYTES and BULK LIVER PATIENT SAMPLES from ARCHS4
# This script searches the entire ARCHS4 database to find relevant samples
library(rhdf5)
library(dplyr)
library(stringr)

# CONFIGURATION ------------
source('ARCHS4_LiverSamples.R')
# ARCHS4 URL - human gene expression v2.2
url <- "https://s3.dev.maayanlab.cloud/archs4/files/human_gene_v2.5.h5"

# Output directory - change this to your preferred location
export_dir <- 'ARCHS4_RESULTS'

# Create output directory if needed
if (!dir.exists(export_dir)) {
  dir.create(export_dir, recursive = TRUE)
}

# CUSTOMIZABLE FILTERS------------
# Include disease samples? (TRUE = include CANCER etc. It will however include MASH always)
include_disease <- FALSE

# Minimum samples per series to be included
min_samples_per_series <- 3

print("Search settings:")
cat(sprintf("  Include disease samples: %s", include_disease))
cat(sprintf("  Minimum samples per series: %d", min_samples_per_series))

# DEFINE SEARCH PATTERNS
# EXCLUSIONS - non-human and cell lines (always excluded)
exclude_always <- c(
  # Cell lines
  "HepG2", "Huh-7", "Huh7", "HepaRG", "Hep3B", "HepG3", "Hep-G2",
  "cell line", "immortalized", "transformed","endothelial","iPSC",
  "leukocytes","leukocyte","hepatocyte-like", "HLC","hESC","stem cells",
  "reendothelialized",
  
  # Non-human species
  "mouse", "murine", "rat", "rodent", 
  "porcine", "pig", "monkey", "canine", 
  "bovine", "cow", "dog",
  
  # Developmental stages
  "fetal", "embryonic", "fetus", "prenatal"
)

# DISEASE EXCLUSIONS (optional, based on include_disease setting)
disease_keywords <- c(
  "cancer", "tumor", "carcinoma", 
  "hepatocellular carcinoma", "HCC",
  "hepatoblastoma", "metasta", 
  "malignant", "neoplasm",
  "Alagille",
  "hepatitis B", "hepatitis C", "HBV", "HCV"
)

# Build final exclusion list
if (include_disease) {
  all_exclusions <- exclude_always
  print("Including disease samples (NASH, cirrhosis, HCC, etc.)")
} else {
  all_exclusions <- c(exclude_always, disease_keywords)
  print("Excluding disease samples (only healthy/normal tissue)")
}

# LOAD ARCHS4 METADATA------------------------
print("Loading ARCHS4 metadata...")
print("This will take a few minutes (downloading ~200MB of metadata)...")

# Read metadata from ARCHS4
all_samples <- h5read(url, "meta/samples/geo_accession", s3 = TRUE)
all_series <- h5read(url, "meta/samples/series_id", s3 = TRUE)
all_titles <- h5read(url, "meta/samples/title", s3 = TRUE)
all_sources <- h5read(url, "meta/samples/source_name_ch1", s3 = TRUE)
all_chars <- h5read(url, "meta/samples/characteristics_ch1", s3 = TRUE)

cat(sprintf("✓ Loaded metadata for %s samples", format(length(all_samples), big.mark = ",")))

# Create metadata dataframe
metadata <- data.frame(
  sample_id = all_samples,
  series_id = all_series,
  sample_title = all_titles,
  sample_source = all_sources,
  sample_characteristics = all_chars,
  stringsAsFactors = FALSE
)

# SEARCH AND FILTER SAMPLES------------------
print("Searching for primary hepatocytes and liver tissue samples...")

# Create combined search text (lowercase for case-insensitive matching)
metadata$search_text <- tolower(paste(
  metadata$sample_title,
  metadata$sample_source,
  metadata$sample_characteristics,
  sep = " | "
))

# Filter for PRIMARY HEPATOCYTES and LIVER
print("Filtering for primary hepatocytes...")
metadata <- metadata %>% filter(sample_id %in% liver_samples) %>%
  filter(series_id!='GSE159491') # this is a weird experimental organoid

hepatocyte_mask <- !grepl(paste(all_exclusions, collapse = "|"),metadata$search_text, ignore.case = TRUE)

all_liver_related <- metadata[hepatocyte_mask, ]

# ORGANIZE BY GEO SERIES-------------------
print("Organizing samples by GEO series...")

# Group hepatocytes by series
all_liver_related_by_series <- all_liver_related %>%
  group_by(series_id) %>%
  summarize(
    n_samples = n(),
    sample_ids = list(sample_id),
    example_title = first(sample_title),
    example_source = first(sample_source),
    .groups = "drop"
  ) %>%
  filter(n_samples >= min_samples_per_series) %>%
  arrange(desc(n_samples))

cat(sprintf("  Total series: %s", format(nrow(all_liver_related_by_series), big.mark = ",")))
cat(sprintf("  GEO series: %d (with >=%d samples each)", 
            nrow(all_liver_related_by_series), min_samples_per_series))
cat(sprintf("  Samples in filtered series: %s",
            format(sum(all_liver_related_by_series$n_samples), big.mark = ",")))


# EXPORT RESULTS----------------------------
# Export all individual samples
write.csv(all_liver_related,
          file.path(export_dir, "archs4_all_liver_related_samples.csv"),
          row.names = FALSE)
print("✓ archs4_all_liver_related_samples.csv")

# Download gene expression--------------------
# Read samples and genes in archive
samples = h5read(url, "meta/samples/geo_accession", s3 = T)
genes = h5read(url, "meta/genes/symbol", s3 = T)
sample_locations = which(samples %in% all_liver_related$sample_id)
# Now read expression
h5closeAll()
expression = h5read(url, "data/expression", index=list(sample_locations, 1:length(genes)), s3 = T)
gc()
# saveRDS(expression,'expression.rds')
expression <- t(expression)
rownames(expression) = genes
colnames(expression) = samples[sample_locations]
saveRDS(expression,'ARCHS4_RESULTS/expression_transposed.rds')