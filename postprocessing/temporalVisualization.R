library(tidyverse)
library(ggplot2)
library(ggpubr)
library(patchwork)
library(progress)
library(tolerance)
library(survival)
library(multtest)
library(data.table)
library(doFuture)
library(progressr)
options(future.globals.maxSize = 800 * 1024^2)
# parallel: set number of workers
cores <- 16
registerDoFuture()
plan(multisession,workers = cores)


# Load signaling state and visualize-----------
df_signaling_state_2hours <- data.table::fread('../results/temporal_analysis/All_intermediate_SignalingStates.csv') %>%
  select(-V1) %>% select(-time) %>% 
  separate(col = 'condition',into = c('drug','dose','timepoint'),sep = '\\+')
df_signaling_state_2hours <- df_signaling_state_2hours %>% 
  mutate(condition=paste(drug,dose,sep = '_'))
df_signaling_state_2hours_v0 <- df_signaling_state_2hours
df_signaling_state_2hours <- distinct(df_signaling_state_2hours %>% 
                                        group_by(condition,node,step) %>% 
                                        mutate(sd_val = sd(signal)) %>% 
                                        mutate(signal=mean(signal)) %>%
                                        # mutate(max_val = quantile(signal, 0.75)) %>% 
                                        # mutate(min_val=quantile(signal, 0.25)) %>%
                                        # mutate(signal=median(signal)) %>%
                                        ungroup() %>% select(-model_no)) 

df_signaling_state_8hours <- data.frame()
pb <- progress_bar$new(
  format = " Processing [:bar] :percent (:eta remaining)",
  total = 50,  # Total iterations = number of rows
  clear = FALSE,
  width = 50
)
for (i in 1:50){
  tmp <- data.table::fread(paste0('../results/temporal_analysis/All_intermediate_SignalingStates_8 hour_model',i,'.csv')) %>%
    select(-V1) %>% select(-time) %>%
    separate(col = 'condition',into = c('drug','dose','timepoint'),sep = '\\+')
  df_signaling_state_8hours <- rbind(df_signaling_state_8hours,tmp)
  # Update progress bar
  pb$tick()
}
df_signaling_state_8hours <- df_signaling_state_8hours %>% 
  mutate(condition=paste(drug,dose,sep = '_'))
df_signaling_state_8hours_v0 <- df_signaling_state_8hours
df_signaling_state_8hours <- distinct(df_signaling_state_8hours %>% 
                                        group_by(condition,node,step) %>% 
                                        mutate(sd_val = sd(signal)) %>% 
                                        mutate(signal=mean(signal)) %>%
                                        # mutate(max_val = quantile(signal, 0.75)) %>% 
                                        # mutate(min_val=quantile(signal, 0.25)) %>%
                                        # mutate(signal=median(signal)) %>%
                                        ungroup() %>% select(-model_no))

# df_signaling_state_24hours <- data.table::fread('../results/temporal_analysis/All_intermediate_SignalingStates_24 hour_model1.csv') %>%
#   select(-V1) %>% select(-time) %>% 
#   separate(col = 'condition',into = c('drug','dose','timepoint'),sep = '\\+')

# Keep only common condition
# df_compare_2h_vs_8h <- left_join(df_signaling_state_2hours %>% 
#                                    select(-drug,-dose,-timepoint,-step,-max_val,-min_val),
#                                  df_signaling_state_8hours %>% 
#                                    select(-drug,-dose,-timepoint,-max_val,-min_val),
#                                  by=c('condition','node')) %>% 
#   filter(!is.na(signal.y)) %>% filter(!is.na(signal.x)) %>%
#   gather('origin','signal',-condition,-node,-step)%>%
#   mutate(origin=ifelse(origin=='signal.x','2 hours','8 hours'))
# tmp1 <- left_join(df_signaling_state_2hours %>% 
#                select(-drug,-dose,-timepoint,-step,-max_val,-signal),
#              df_signaling_state_8hours %>% 
#                select(-drug,-dose,-timepoint,-max_val,-signal),
#              by=c('condition','node')) %>% 
#   filter(!is.na(min_val.y)) %>% filter(!is.na(min_val.x)) %>%
#   gather('origin','min_val',-condition,-node,-step)%>%
#   mutate(origin=ifelse(origin=='min_val.x','2 hours','8 hours'))
# tmp2 <- left_join(df_signaling_state_2hours %>% 
#                     select(-drug,-dose,-timepoint,-step,-min_val,-signal),
#                   df_signaling_state_8hours %>% 
#                     select(-drug,-dose,-timepoint,-min_val,-signal),
#                   by=c('condition','node')) %>% 
#   filter(!is.na(max_val.y)) %>% filter(!is.na(max_val.x)) %>%
#   gather('origin','max_val',-condition,-node,-step)%>%
#   mutate(origin=ifelse(origin=='max_val.x','2 hours','8 hours'))
df_compare_2h_vs_8h <- left_join(df_signaling_state_2hours %>% 
                                   select(-drug,-dose,-timepoint,-step,-sd_val),
                                 df_signaling_state_8hours %>% 
                                   select(-drug,-dose,-timepoint,-sd_val),
                                 by=c('condition','node')) %>% 
  filter(!is.na(signal.y)) %>% filter(!is.na(signal.x)) %>%
  gather('origin','signal',-condition,-node,-step)%>%
  mutate(origin=ifelse(origin=='signal.x','2 hours','8 hours'))
tmp1 <- left_join(df_signaling_state_2hours %>% 
                    select(-drug,-dose,-timepoint,-step,-signal),
                  df_signaling_state_8hours %>% 
                    select(-drug,-dose,-timepoint,-signal),
                  by=c('condition','node')) %>% 
  filter(!is.na(sd_val.y)) %>% filter(!is.na(sd_val.x)) %>%
  gather('origin','sd_val',-condition,-node,-step)%>%
  mutate(origin=ifelse(origin=='sd_val.x','2 hours','8 hours'))
# tmp2 <- left_join(df_signaling_state_2hours %>% 
#                     select(-drug,-dose,-timepoint,-step,-min_val,-signal),
#                   df_signaling_state_8hours %>% 
#                     select(-drug,-dose,-timepoint,-min_val,-signal),
#                   by=c('condition','node')) %>% 
#   filter(!is.na(max_val.y)) %>% filter(!is.na(max_val.x)) %>%
#   gather('origin','max_val',-condition,-node,-step)%>%
#   mutate(origin=ifelse(origin=='max_val.x','2 hours','8 hours'))
df_compare_2h_vs_8h <- left_join(df_compare_2h_vs_8h,tmp1)
# df_compare_2h_vs_8h <- left_join(df_compare_2h_vs_8h,tmp2)
tmp1 <- NULL
tmp2 <- NULL
gc()
df_compare_2h_vs_8h$origin <- factor(df_compare_2h_vs_8h$origin,levels = c('8 hours','2 hours'))

ggplot(df_compare_2h_vs_8h %>% filter(condition=='phenylbutazone_High'),
       aes(x=step,y=signal,color=origin,lty=origin)) +
  geom_line(lwd=1) +
  # geom_line(aes(x=step,y=min_val,color=origin),linetype='dashed',inherit.aes = FALSE)+
  # geom_line(aes(x=step,y=max_val,color=origin),linetype='dashed',inherit.aes = FALSE)+
  geom_ribbon(aes(ymin=signal - sd_val,ymax=signal + sd_val,color=origin),alpha=0.2)+
  theme_minimal()+ 
  facet_wrap(~node)

## Statistically compare the results-------------------------------
# =============================================================================
# --- 1. Data Preparation ---
# =============================================================================

# We assume your two data frames 'df_signaling_state_8hours_v0' (simulations)
# and 'df_signaling_state_2hours_v0' (targets) are loaded in your R environment.

# Filter data to ensure common conditions and molecules
common_conditions <- intersect(df_signaling_state_8hours_v0$condition, df_signaling_state_2hours_v0$condition)
common_nodes <- intersect(df_signaling_state_8hours_v0$node, df_signaling_state_2hours_v0$node)

# Prepare the simulation data table (Y_dt)
curves_mat <- df_signaling_state_8hours_v0 %>%
  filter(condition %in% common_conditions, node %in% common_nodes)

Y_dt <- as.data.table(curves_mat %>% select(condition, node, model_no, step, signal))
setnames(Y_dt, 
         old = c("node", "model_no", "signal"), 
         new = c("molecule", "replicate", "value"))
setkey(Y_dt, molecule, condition, replicate, step)


# Prepare the experimental target data table (target_dt)
target_values_mat <- df_signaling_state_2hours_v0 %>%
  filter(condition %in% common_conditions, node %in% common_nodes)

target_dt <- as.data.table(target_values_mat %>% select(condition, node, model_no, signal))
setnames(target_dt, 
         old = c("node", "model_no", "signal"), 
         new = c("molecule", "replicate", "value"))

# Create the tolerance interval lookup table (bounds_dt)
bounds_dt <- target_dt[, {
  ti <- normtol.int(x = value, P = 0.95, alpha = 0.05)
  .(L = ti$`1-sided.lower`, U = ti$`1-sided.upper`)
}, by = .(molecule, condition)]
setkey(bounds_dt, molecule, condition)

cat("Step 1: Data preparation complete.\n")


# =============================================================================
# --- 2. Calculate the OBSERVED Molecule-Level Statistic ---
# =============================================================================

# Merge the tolerance bounds with the simulation curves based on the correct pairings
Y_dt_merged <- bounds_dt[Y_dt, on = .(molecule, condition)]

# First, calculate the in-band proportion for every single curve
curve_summary_dt <- Y_dt_merged[, .(
  in_band_prop = sum(value >= L & value <= U) / .N
), by = .(molecule, condition, replicate)]

# Second, calculate the final observed statistic for each molecule:
# The mean of all its curve proportions.
observed_stats <- curve_summary_dt[, .(
  observed_mean_prop = mean(in_band_prop)
), by = .(molecule)]

setkey(observed_stats, molecule)

cat("Step 2: Observed goodness-of-fit statistics calculated.\n")
print(observed_stats)


# =============================================================================
# --- 3. Molecule-Level Permutation Test ---
# =============================================================================

# --- A. Setup Parallel Backend ---
B <- 200  # Number of permutations. Use at least 1,000 for publication.
unique_molecules <- unique(Y_dt$molecule)
cores <- availableCores() - 1 # Use all cores minus one

# Increase global size limit if your Y_dt is large (e.g., > 500MB)
options(future.globals.maxSize = 800 * 1024^2) # 800 MiB

# Register the parallel backend
registerDoFuture()
plan(multisession, workers = cores) # Use multicore on Linux/macOS for better performance

# Setup progress bar handlers
handlers("progress")

cat(paste("\nStep 3: Starting", B, "permutations on", cores, "cores...\n"))

# --- B. Run the Parallel `foreach` Loop ---
perm_results_matrix <- with_progress({
  p <- progressor(steps = B)
  
  # The result will be a matrix where rows=molecules, cols=permutations
  foreach(b = 1:B, .combine = 'cbind') %dopar% {
    
    # 1. Create a shuffled mapping of molecule identities.
    perm_map <- data.table(
      molecule = unique_molecules,
      permuted_molecule_for_bounds = sample(unique_molecules)
    )
    
    # 2. Create a temporary bounds table where the original TIs are now
    #    labeled with a permuted molecule ID.
    bounds_perm <- perm_map[bounds_dt, on = .(permuted_molecule_for_bounds = molecule)]
    setnames(bounds_perm, "molecule", "original_molecule_for_bounds")
    setnames(bounds_perm, "permuted_molecule_for_bounds", "molecule")
    
    # 3. Merge the original curves (Y_dt) with the SHUFFLED bounds.
    #    Now, curves for molecule 'A' are matched with bounds from a random molecule.
    Y_dt_perm <- bounds_perm[Y_dt, on = .(molecule, condition)]
    
    # 4. Calculate the in-band proportion for this incorrect pairing.
    perm_curve_summary <- Y_dt_perm[!is.na(L), .( # !is.na(L) handles cases of imperfect merges
      perm_prop = sum(value >= L & value <= U) / .N
    ), by = .(molecule, condition, replicate)]
    
    # 5. Calculate the molecule-level statistic for this permuted run.
    perm_stats_for_iteration <- perm_curve_summary[, .(
      perm_mean_prop = mean(perm_prop)
    ), by = .(molecule)]
    
    p() # Update the progress bar
    
    # Ensure the result vector is in the same order as observed_stats
    setkey(perm_stats_for_iteration, molecule)
    return(perm_stats_for_iteration$perm_mean_prop)
  }
})

# --- C. Cleanup ---
# It's good practice to shut down the parallel workers when done.
plan(sequential)
gc() # Garbage collect to free up memory

cat("\nStep 3: Permutation test complete.\n")


# =============================================================================
# --- 4. Calculate Final P-Values and Report Results ---
# =============================================================================

# Calculate the p-value for each molecule.
# For each molecule (row), count how many times a permuted statistic
# was greater than or equal to its single observed statistic.
p_values <- rowMeans(perm_results_matrix >= observed_stats$observed_mean_prop)

# Add p-values to our results table
observed_stats$p_value <- p_values

# Apply a multiple testing correction to control the False Discovery Rate
observed_stats$p_adj_BH <- p.adjust(observed_stats$p_value, method = "BH")

# --- FINAL RESULTS TABLE ---
# This is your final output: one line for each molecule, sorted by significance.
cat("\n--- FINAL MOLECULE-LEVEL STATISTICAL RESULTS ---\n")
print(observed_stats[order(p_adj_BH)])



# Calculate first hit time and event status for each curve-----------------
fpt_dt <- Y_dt[, {
  in_band <- value >= L & value <= U
  hit_time <- which.max(in_band)
  
  # Check if a true hit occurred
  if (hit_time == 1 && !in_band[1]) {
    .(tau = .N + 1, event = FALSE, curve_len = .N)
  } else {
    .(tau = hit_time, event = TRUE, curve_len = .N)
  }
}, by = .(molecule, condition, replicate)]

# > fpt_dt
#    molecule condition replicate tau event curve_len
# 1:        1         1         1   9  TRUE       100
# 2:        1         1         2 101 FALSE       100  # Censored at 100+1
# 3:        1         1         3  12  TRUE        95
# 4:        1         1         4  96 FALSE        95  # Censored at 95+1

# The rest of the analysis proceeds as before, but on this new `fpt_dt` table
# Loop through each (m, c) pair to fit Kaplan-Meier models
median_fpt <- matrix(NA, nrow = M, ncol = C)
sign_test_p_values <- matrix(NA, nrow = M, ncol = C)

for (m in 1:M) {
  for (c in 1:C) {
    # Subset the data for the specific condition
    subset_fpt <- fpt_dt[molecule == m & condition == c]
    
    # Create a survival object with variable censoring times
    surv_obj <- Surv(time = subset_fpt$tau, event = subset_fpt$event)
    
    # ... (rest of the km_fit and binom.test code is the same)
  }
}