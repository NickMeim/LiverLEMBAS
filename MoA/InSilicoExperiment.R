library(tidyverse)
library(forcats)
library(doFuture)
library(ropls)
library(patchwork)
# parallel: set number of workers
cores <- 16
registerDoFuture()
plan(multisession,workers = cores)
source("../postprocessing/plotting_functions.R")
source('../postprocessing/functions_translation.R')
options(future.globals.maxSize = 2 * 1024^3) # 2 GiB
library(rstatix)
library(plotly)
library(htmlwidgets)
library(igraph)
library(MESS)
library(scales)

add_type <- function(.data, default_label = no_ko_type_label) {
  if ('type' %in% names(.data)) dplyr::mutate(.data, type = .data[['type']])
  else dplyr::mutate(.data, type = default_label)
}

make_ifna_facet_plot <- function(
    phenotype_value = c("MAS", "Fibrosis stage"),
    Ysame, Yopp, Ystim_no_ko,
    mean_control, se_control,
    extreme_abs_ifna = 3000,
    p_adjust_method = "bonferroni",
    type_labels = NULL,
    type_col = "type",
    no_ko_type_label = "no KO",
    y_lower = -2,
    y_upper = 8,
    xlab = "stimulation strength",
    ylab = "score",
    plot_title = NULL,
    base_text_size = 18,
    legend_position = "top",
    legend_text_size = 14,
    legend_title_size = 14,
    facet_ncol = NULL,
    facet_nrow = NULL,
    curve_linewidth = 0.9,
    ribbon_alpha = 0.12,
    arrow_linewidth = 0.55,
    arrow_head_inches = 0.09,
    ann_text_size = 3.2,
    untreated_text_size = 3.2,
    x_nudge_frac = 0.02,
    ann_y_pad_frac = 0.04,
    prefer_paired_test = TRUE,
    untreated_x_frac = 0.35
) {
  phenotype_value <- match.arg(phenotype_value)
  if (is.null(plot_title)) plot_title <- phenotype_value
  
  add_type <- function(.data, default_label = no_ko_type_label) {
    if (type_col %in% names(.data)) dplyr::mutate(.data, type = .data[[type_col]])
    else dplyr::mutate(.data, type = default_label)
  }
  
  # Raw replicate-level data
  raw <- rbind(
    rbind(Ysame, Yopp) %>%
      dplyr::ungroup() %>%
      add_type() %>%
      dplyr::filter(!is.na(IFNa), !is.na(TGFb), phenotype == phenotype_value) %>%
      dplyr::select(model_no, IFNa, score, stimulation, type) %>%
      unique(),
    Ystim_no_ko %>%
      dplyr::ungroup() %>%
      add_type(default_label = no_ko_type_label) %>%
      dplyr::filter(!is.na(IFNAR1), !is.na(TGFRB1), phenotype == phenotype_value) %>%
      dplyr::select(model_no, IFNa = IFNAR1, score, stimulation, type) %>%
      unique()
  )
  if (!is.null(type_labels)) raw$type <- dplyr::recode(raw$type, !!!type_labels)
  
  stims <- if (is.factor(raw$stimulation)) levels(raw$stimulation) else sort(unique(raw$stimulation))
  stims <- stims[!is.na(stims)]
  if (length(stims) != 2) stop("Expected exactly 2 stimulation levels.")
  
  # Plot df recomputed from score (type-specific curves)
  plot_df <- raw %>%
    dplyr::group_by(type, IFNa, stimulation) %>%
    dplyr::summarise(
      mu = mean(score, na.rm = TRUE),
      se = stats::sd(score, na.rm = TRUE) / sqrt(sum(!is.na(score))),
      n = sum(!is.na(score)),
      .groups = "drop"
    )
  
  n_types <- dplyr::n_distinct(plot_df$type)
  if (is.null(facet_ncol) && is.null(facet_nrow)) facet_ncol <- ceiling(sqrt(n_types))
  
  x_range <- range(plot_df$IFNa, na.rm = TRUE)
  x_span  <- diff(x_range)
  x_nudge <- x_nudge_frac * x_span
  y_pad   <- ann_y_pad_frac * (y_upper - y_lower)
  
  # Tests at extremes
  extreme_raw <- raw %>% dplyr::filter(abs(IFNa) == extreme_abs_ifna)
  
  paired_ok <- extreme_raw %>%
    dplyr::group_by(type, IFNa, model_no) %>%
    dplyr::summarise(n_stims = dplyr::n_distinct(stimulation), .groups = "drop") %>%
    dplyr::summarise(all_paired = all(n_stims == 2), .groups = "drop") %>%
    dplyr::pull(all_paired)
  
  do_paired <- isTRUE(prefer_paired_test) && isTRUE(paired_ok)
  
  tt <- extreme_raw %>%
    dplyr::group_by(type, IFNa) %>%
    dplyr::summarise(
      p_raw = {
        dat <- dplyr::cur_data()
        if (do_paired) {
          wide <- dat %>%
            dplyr::select(model_no, stimulation, score) %>%
            tidyr::pivot_wider(names_from = stimulation, values_from = score)
          x <- wide[[stims[1]]]; y <- wide[[stims[2]]]
          tryCatch(stats::t.test(x, y, paired = TRUE)$p.value, error = function(e) NA_real_)
        } else {
          tryCatch(stats::t.test(score ~ stimulation, data = dat)$p.value, error = function(e) NA_real_)
        }
      },
      .groups = "drop"
    ) %>%
    dplyr::mutate(
      p_adj = stats::p.adjust(p_raw, method = p_adjust_method),
      stars = dplyr::case_when(
        is.na(p_adj) ~ "",
        p_adj < 0.001 ~ "***",
        p_adj < 0.01  ~ "**",
        p_adj < 0.05  ~ "*",
        TRUE          ~ ""
      )
    )
  
  # Annotation endpoints from plotted means at extremes
  ann <- plot_df %>%
    dplyr::filter(abs(IFNa) == extreme_abs_ifna) %>%
    dplyr::select(type, IFNa, stimulation, mu) %>%
    tidyr::pivot_wider(names_from = stimulation, values_from = mu) %>%
    dplyr::mutate(
      mu_a = .data[[stims[1]]],
      mu_b = .data[[stims[2]]],
      diff = mu_b - mu_a,
      y0 = pmin(mu_a, mu_b),
      y1 = pmax(mu_a, mu_b),
      ymid = (y0 + y1) / 2
    ) %>%
    dplyr::left_join(tt, by = c("type", "IFNa")) %>%
    dplyr::group_by(type) %>%
    dplyr::mutate(
      which = dplyr::if_else(IFNa == min(IFNa, na.rm = TRUE), "xmin", "xmax"),
      x_text = dplyr::if_else(which == "xmin", IFNa + x_nudge, IFNa - x_nudge),
      hjust  = dplyr::if_else(which == "xmin", 0, 1),
      
      # keep label near the arrow, but nudge it away from the untreated band height if needed
      untreated_y = mean_control + se_control,
      y_text = dplyr::if_else(abs(ymid - untreated_y) < 0.6 * y_pad, ymid - y_pad, ymid),
      
      label  = paste0(stars, "  Δ=", sprintf("%.2f", diff))
    ) %>%
    dplyr::ungroup()
  
  # Untreated band + label position (moved away from left extreme)
  untreated_band <- plot_df %>%
    dplyr::distinct(type) %>%
    dplyr::mutate(
      ymin = mean_control - se_control,
      ymax = mean_control + se_control,
      
      # place label around the middle-left instead of at xmin (prevents collision with Δ at xmin)
      x = x_range[1] + untreated_x_frac * x_span,
      
      # lift the label a bit above the band (clamped into plot)
      y = pmin(y_upper - 0.02 * (y_upper - y_lower), (mean_control + se_control) + 0.03 * (y_upper - y_lower)),
      label = "untreated score"
    )
  
  p <- ggplot2::ggplot(
    plot_df,
    ggplot2::aes(x = IFNa, y = mu, color = stimulation, fill = stimulation, group = stimulation)
  ) +
    ggplot2::geom_rect(
      data = untreated_band,
      ggplot2::aes(xmin = -Inf, xmax = Inf, ymin = ymin, ymax = ymax),
      inherit.aes = FALSE,
      fill = "grey", alpha = 0.2,
      color = "grey50", linetype = "dashed", linewidth = 0.35
    ) +
    ggplot2::geom_text(
      data = untreated_band,
      ggplot2::aes(x = x, y = y, label = label),
      inherit.aes = FALSE,
      color = "grey40", fontface = "bold",
      hjust = 0.5, vjust = 0, size = untreated_text_size
    ) +
    ggplot2::geom_ribbon(ggplot2::aes(ymin = mu - se, ymax = mu + se), alpha = ribbon_alpha, color = NA) +
    ggplot2::geom_line(linewidth = curve_linewidth) +
    ggplot2::geom_segment(
      data = ann,
      ggplot2::aes(x = IFNa, xend = IFNa, y = y0, yend = y1),
      inherit.aes = FALSE,
      arrow = grid::arrow(length = grid::unit(arrow_head_inches, "inches"), ends = "both", type = "closed"),
      linewidth = arrow_linewidth,
      color = "black"
    ) +
    
    # Δ labels with fully transparent background (no box)
    ggplot2::geom_label(
      data = ann,
      ggplot2::aes(x = x_text, y = y_text, label = label, hjust = hjust),
      inherit.aes = FALSE,
      size = ann_text_size,
      label.size = 0,
      fill = NA,          # transparent background
      alpha = 1,          # keep text fully opaque
      color = "black"
    ) +
    
    ggplot2::facet_wrap(~type, ncol = facet_ncol, nrow = facet_nrow) +
    ggplot2::labs(title = plot_title, x = xlab, y = ylab) +
    ggplot2::scale_y_continuous(limits = c(y_lower, y_upper)) +
    ggplot2::guides(
      color = ggplot2::guide_legend(order = 1, nrow = 1, byrow = TRUE),
      fill  = ggplot2::guide_legend(order = 1, nrow = 1, byrow = TRUE)
    )
  
  p <- add_theme(p) +
    ggplot2::theme(
      plot.title = ggplot2::element_text(size = base_text_size, face = "bold", hjust = 0.5),
      legend.position = legend_position,
      legend.box = "vertical",
      legend.direction = "horizontal",
      legend.box.just = "left",
      legend.spacing.y = grid::unit(0.12, "cm"),
      legend.text = ggplot2::element_text(size = legend_text_size),
      legend.title = ggplot2::element_text(size = legend_title_size),
      text = ggplot2::element_text(size = base_text_size)
    )
  
  list(plot = p, plot_df = plot_df, raw = raw, ann = ann, tests = tt, stim_levels = stims, paired_test_used = do_paired)
}




## Simulate grid combination of TGFBR1 and IFNAR1 plus inhibition of certain nodes---------------
simulation <- data.table::fread('../results/24 hour/allmodels/combo_simulation_grid_multiKO/TFs_all_conditions_grid_multiKO.csv') %>% 
  select(-knockout_node,-knockout_strength)
simulation <- simulation %>% filter(type!='untreated_with_KO') #%>% filter(type!="untreated_no_KO") 
Ystim_no_ko <- readRDS('../results/24 hour/allmodels/combo_simulation_expanded/Yall_same_stim.rds')

#Load annotation file to replace with gene names
annot <- read.delim('../../../Artificial-Signaling-Network/DrugsANNSignaling/data/uniprot-reviewed_yes+AND+organism__Homo+sapiens+(Human)+[9606]_.tab')
annot <- annot %>% select(c('node'='Entry'),c('TF'='Gene.names...primary..')) %>% unique()
annot <- annot %>% filter(node %in% colnames(simulation))
annot <- left_join(data.frame(node = colnames(simulation)[1:104]),annot)
print(all(colnames(simulation)[1:104]==annot$node))
colnames(simulation)[1:104] <- annot$TF

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
TF_human <- TF_human[,colnames(simulation)[1:104]]
## train one model with all the data
plsr_model <- opls(x = TF_human, 
                   y = Yh,
                   predI = 8,
                   crossvalI = 1,
                   scaleC = "center",
                   fig.pdfC = "none",
                   info.txtC = "none")
plot(Yh[,1],predict(plsr_model,TF_human)[,1])
plot(Yh[,2],predict(plsr_model,TF_human)[,2])
print(diag(cor(Yh,predict(plsr_model,TF_human))))
options(future.globals.maxSize = 2 * 1024^3) # 2 GiB
Yinsilico <- foreach(i = seq(0,49)) %dopar% {
  stimuli <- simulation %>% filter(model_no==i) %>% filter(type!='untreated') %>% select(IFNa,TGFb,type)
  expreriment <- as.matrix(simulation %>% filter(model_no==i) %>% filter(type!='untreated') %>%
                             select(-type,-model_no,-IFNa,-TGFb))
  expreriment[expreriment>1] <-1
  expreriment[expreriment<0] <-0
  expreriment <- log((expreriment+1e-6)/(1 - expreriment + 1e-6))
  colnames(expreriment) <- colnames(simulation)[1:104]
  ## predict
  yhat <- as.data.frame(predict(plsr_model,expreriment)) %>% mutate(model_no=i)
  yhat <- cbind(yhat,stimuli)
  yhat
}
Yinsilico <- do.call(rbind,Yinsilico)
control <- as.matrix(simulation %>% filter(type=='untreated') %>%
                       select(-type,-model_no,-IFNa,-TGFb))
control[control>1] <-1
control[control<0] <-0
control <- log((control+1e-6)/(1 - control + 1e-6))
colnames(control) <- colnames(simulation)[1:104]
Ycontrol <- predict(plsr_model,control)
mean_control_mas <- mean(Ycontrol[,1]) 
se_control_mas <- sd(Ycontrol[,1])/sqrt(50)
mean_control_fib <- mean(Ycontrol[,2]) 
se_control_fib <- mean(Ycontrol[,2]) /sqrt(50)
## save a centered version
Ycontrol <- as.data.frame(Ycontrol) %>% mutate(model_no=seq(0,49,1))
Yinsilico <- Yinsilico %>% gather('phenotype','score',-model_no,-IFNa,-TGFb,-type) %>%
  group_by(phenotype,IFNa,TGFb) %>% mutate(mu = mean(score)) %>% mutate(se=sd(score)/sqrt(50)) %>%
  ungroup()
# saveRDS(Ycontrol,'../results/allmodels/combo_simulation_grid_multiKO/Ycontrol.rds')
# saveRDS(Yinsilico,'../results/allmodels/combo_simulation_grid_multiKO/Yinsilico.rds')

### Get all results--------------------------
Ystim_no_ko <- readRDS('../results/24 hour/allmodels/combo_simulation_expanded/Yall_same_stim.rds')
Ycontrol <- readRDS('../results/24 hour/allmodels/combo_simulation_grid_multiKO/Ycontrol.rds')
mean_control_mas <- mean(Ycontrol[,1]) 
se_control_mas <- sd(Ycontrol[,1])/sqrt(50)
mean_control_fib <- mean(Ycontrol[,2]) 
se_control_fib <- mean(Ycontrol[,2]) /sqrt(50)
Yinsilico <- readRDS('../results/24 hour/allmodels/combo_simulation_grid_multiKO/Yinsilico.rds')
Yall <- Yinsilico %>%
  mutate(phenotype = ifelse(phenotype=='NAS','MAS','Fibrosis stage'))
Yall <- Yall %>% mutate(IFNa=as.numeric(IFNa)) %>% mutate(TGFb=as.numeric(TGFb))

Ysame <- Yall %>% filter(IFNa==TGFb) %>% mutate(stimulation='IFNAR1*TGFBR1>0')
Yopp <- Yall %>% filter(IFNa==(-TGFb))  %>% mutate(stimulation='IFNAR1*TGFBR1<0')

### Statistical Comparison of the whole grid against the no KO case------------
# aggregated_grids <- distinct(rbind(Yall %>% select(-score,-model_no),
#                              Ystim_no_ko %>% mutate(type='no KO') %>%
#                                rename(IFNa = IFNAR1,
#                                       TGFb = TGFRB1) %>%
#                                select(all_of(colnames(Yall))) %>%
#                                select(-score,-model_no)))
aggregated_grids <- distinct(rbind(
  Yall %>% select("model_no", "IFNa", "TGFb", "type", "phenotype", "score", "mu", "se"),
  Ystim_no_ko %>%
    mutate(type = 'no KO') %>%
    rename(IFNa = IFNAR1, TGFb = TGFRB1) %>%
    group_by(model_no, IFNa, TGFb, type, phenotype) %>%
    mutate(mu = mean(score), se = sd(score)) %>% ungroup() %>%
    select("model_no", "IFNa", "TGFb", "type", "phenotype", "score", "mu", "se")
))

compute_vol <- function(df, ...) {
  ifna_vals <- sort(unique(df$IFNa))
  tgfb_vals <- sort(unique(df$TGFb))
  
  if (length(ifna_vals) < 2 || length(tgfb_vals) < 2) {
    message("Skipping group with insufficient grid points: ",
            paste(unique(df$type), unique(df$model_no)))
    return(tibble(VOL = NA_real_, x_range = NA_real_, VOL_norm = NA_real_))
  }
  
  total <- 0
  for (i in seq_along(ifna_vals)[-length(ifna_vals)]) {
    for (j in seq_along(tgfb_vals)[-length(tgfb_vals)]) {
      dI <- ifna_vals[i+1] - ifna_vals[i]
      dT <- tgfb_vals[j+1] - tgfb_vals[j]
      corners <- df$abs_diff[df$IFNa == ifna_vals[i]   & df$TGFb == tgfb_vals[j]  ] +
        df$abs_diff[df$IFNa == ifna_vals[i+1] & df$TGFb == tgfb_vals[j]  ] +
        df$abs_diff[df$IFNa == ifna_vals[i]   & df$TGFb == tgfb_vals[j+1]] +
        df$abs_diff[df$IFNa == ifna_vals[i+1] & df$TGFb == tgfb_vals[j+1]]
      total <- total + (dI * dT * corners / 4)
    }
  }
  tibble(
    VOL      = total,
    x_range  = (max(df$IFNa) - min(df$IFNa)) * (max(df$TGFb) - min(df$TGFb)),
    VOL_norm = total / x_range
  )
}

make_volume_plot <- function(phen, baseline_col, outpath,
                             width_cm = 16, height_cm = 20, dpi = 600) {
  
  grid_clean <- aggregated_grids %>%
    filter(!is.na(IFNa), !is.na(TGFb),
           phenotype == phen,
           type != "no KO") %>%
    select(model_no, IFNa, TGFb, score, type) %>%
    left_join(
      Ycontrol %>% transmute(model_no, baseline = .data[[baseline_col]]),
      by = "model_no"
    ) %>%
    mutate(abs_diff = abs(score - baseline))
  
  curves_3d <- grid_clean %>%
    group_by(model_no, type) %>%
    arrange(IFNa, TGFb) %>%
    group_modify(compute_vol) %>%
    ungroup() %>%
    filter(!is.na(VOL))
  
  # order types by Cohen's d vs TP63 KO
  cohens_d_order_3d <- curves_3d %>%
    rstatix::cohens_d(VOL ~ type, ref.group = "TP63 KO") %>%
    arrange(desc(effsize)) %>%
    pull(group2)
  
  curves_3d <- curves_3d %>%
    mutate(type = factor(type, levels = c("TP63 KO", cohens_d_order_3d)))
  
  stats_3d <- curves_3d %>%
    rstatix::t_test(VOL ~ type, ref.group = "TP63 KO", paired = FALSE) %>%
    rstatix::adjust_pvalue(method = "BH") %>%
    rstatix::add_significance() %>%
    mutate(y.position = max(curves_3d$VOL, na.rm = TRUE) * 1.05)
  
  keep_types <- c("TP63 KO", stats_3d$group2[stats_3d$p.adj < 0.05])
  
  p <- ggboxplot(
    curves_3d %>% filter(type %in% keep_types),
    x = "type", y = "VOL", add = "jitter"
  ) +
    stat_pvalue_manual(
      stats_3d %>% filter(p.adj < 0.05),
      label = "p.adj.signif",
      remove.bracket = TRUE
    ) +
    geom_hline(
      yintercept = mean(curves_3d$VOL[curves_3d$type == "TP63 KO"], na.rm = TRUE),
      lty = "dashed", color = "red", lwd = 2
    ) +
    coord_flip() +
    labs(title = phen)+
    theme(plot.title = element_text(hjust=0.5))
  
  ggsave(outpath, plot = p, width = width_cm, height = height_cm, units = "cm", dpi = dpi)
  return(p)
}
p_mas <- make_volume_plot(
  phen = "MAS",
  baseline_col = "NAS",
  outpath = "../figures/volume_comparison_MAS_24hr.pdf"
)
p_fib <- make_volume_plot(
  phen = "Fibrosis stage",
  baseline_col = "fibrosis",
  outpath = "../figures/volume_comparison_Fibrosis_24hr.pdf"
)

## 3D surface plot
phen <- "Fibrosis stage"
baseline_col <- "fibrosis"

ko_genes <- c("STAT2","STAT3","SMAD3","TP63","TP53","JUN","MYC")
ko_types <- paste0(ko_genes, " KO")
ko_cols <- c(
  STAT2 = "#0072B2",
  STAT3 = "#E69F00",
  SMAD3 = "#009E73",
  TP63  = "#D55E00",
  TP53  = "#CC79A7",
  JUN   = "#56B4E9",
  MYC   = "#F0E442"
)

# Average grids across all 50 models (strict)
# make sure these are numeric (in case they came in as character/factor)
surf_df <- aggregated_grids %>%
  filter(phenotype == phen,
         type %in% ko_types,
         !is.na(IFNa), !is.na(TGFb)) %>%
  mutate(
    IFNa = as.numeric(IFNa),
    TGFb = as.numeric(TGFb)
  ) %>%
  group_by(type, IFNa, TGFb) %>%
  summarise(
    z = mean(score, na.rm = TRUE),
    n_models = n_distinct(model_no),
    .groups = "drop"
  )

ifna_vals <- sort(unique(surf_df$IFNa))
tgfb_vals <- sort(unique(surf_df$TGFb))

zmat_for_type <- function(tt) {
  tmp <- surf_df %>%
    filter(type == tt) %>%
    select(IFNa, TGFb, z) %>%   # IMPORTANT: drop n_models (and type) before pivot
    complete(IFNa = ifna_vals, TGFb = tgfb_vals) %>%
    arrange(IFNa, TGFb) %>%
    pivot_wider(id_cols = IFNa, names_from = TGFb, values_from = z)
  
  zmat <- as.matrix(tmp[, -1, drop = FALSE])
  storage.mode(zmat) <- "numeric"  # force numeric matrix
  zmat
}

# baseline plane (numeric matrix)
baseline_mean <- Ycontrol %>%
  summarise(b = mean(.data[[baseline_col]], na.rm = TRUE)) %>%
  pull(b)
baseline_z <- matrix(as.numeric(baseline_mean),
                     nrow = length(ifna_vals),
                     ncol = length(tgfb_vals))
# very light transparent grey (baseline)
baseline_col <- "rgba(0,0,0,1)" 

p <- plot_ly() %>%
  add_surface(
    x = ifna_vals, y = tgfb_vals, z = baseline_z,
    name = "Control baseline",
    opacity = 1,
    showscale = FALSE,
    colorscale = list(list(0, baseline_col), list(1, baseline_col))
  )


# Add each KO surface (uniform color per KO)

for (g in ko_genes) {
  tt <- paste0(g, " KO")
  if (!tt %in% unique(surf_df$type)) next
  
  zmat <- zmat_for_type(tt)
  col <- ko_cols[[g]]
  
  p <- p %>%
    add_surface(
      x = ifna_vals, y = tgfb_vals, z = zmat,
      name = tt,
      opacity = 0.70,
      showscale = FALSE,
      colorscale = list(list(0, col), list(1, col))
    )
}

p <- p %>%
  layout(
    scene = list(
      xaxis = list(title = "IFNa"),
      yaxis = list(title = "TGFb"),
      zaxis = list(title = paste0("Score (", phen, ")"))
    ),
    legend = list(
      orientation = "h",   # horizontal legend
      x = 0.5, xanchor = "center",
      y = 1.08, yanchor = "bottom"
    ),
    margin = list(t = 90)
  )

# Force a legend by adding dummy scatter3d traces (surfaces don't reliably appear in legend)
legend_x <- ifna_vals[1]
legend_y <- tgfb_vals[1]
legend_z <- baseline_mean

# Baseline legend item
p <- p %>%
  add_trace(
    type = "scatter3d", mode = "markers",
    x = legend_x, y = legend_y, z = legend_z,
    name = "Control baseline",
    marker = list(color = "rgba(0,0,0,1)", size = 6),
    showlegend = TRUE,
    visible = "legendonly"
  )

# KO legend items
for (g in ko_genes) {
  tt  <- paste0(g, " KO")
  col <- ko_cols[[g]]
  
  p <- p %>%
    add_trace(
      type = "scatter3d", mode = "markers",
      x = legend_x, y = legend_y, z = legend_z,
      name = tt,
      marker = list(color = col, size = 6),
      showlegend = TRUE,
      visible = "legendonly"
    )
}

# Put legend at top
p <- p %>%
  layout(
    showlegend = TRUE,
    legend = list(
      orientation = "h",
      x = 0.5, xanchor = "center",
      y = 1.12, yanchor = "top"
    ),
    margin = list(t = 120)
  )

p
# Save interactive
htmlwidgets::saveWidget(p, file = paste0("../figures/avg50_24hour_3D_surfaces_", gsub(" ", "_", phen), ".html"),
                        selfcontained = TRUE)


### Statistical Comparison of the differences of curves------------
## INFa extreme inhibition, TGFB extreme activation
compute_curves_point <- function(phen, ifna_extreme = -3000) {
  
  bind_rows(
    bind_rows(Ysame, Yopp) %>%
      ungroup() %>%
      add_type() %>%
      filter(!is.na(IFNa), !is.na(TGFb), phenotype == phen) %>%
      select(model_no, IFNa, score, stimulation, type) %>%
      distinct(),
    Ystim_no_ko %>%
      ungroup() %>%
      add_type(default_label = "no KO") %>%
      filter(!is.na(IFNAR1), !is.na(TGFRB1), phenotype == phen) %>%
      select(model_no, IFNa = IFNAR1, score, stimulation, type) %>%
      distinct()
  ) %>%
    select(model_no, x = IFNa, score, stimulation, type) %>%
    tidyr::pivot_wider(names_from = stimulation, values_from = score) %>%
    filter(x == ifna_extreme) %>%
    group_by(model_no, type) %>%
    summarise(
      diff_point = `IFNAR1*TGFBR1<0` - `IFNAR1*TGFBR1>0`,
      .groups = "drop"
    ) %>%
    mutate(phenotype = phen)
}

make_point_plots <- function(phen,
                             ifna_extreme = -3000,
                             out1 = NULL,
                             out2 = NULL,
                             width_cm = 16,
                             height_cm = 20,
                             dpi = 600) {
  
  curves_point <- compute_curves_point(phen, ifna_extreme = ifna_extreme)
  
  ## Plot 1: compare to "no KO"
  cohens_d_order_pt <- curves_point %>%
    rstatix::cohens_d(diff_point ~ type, ref.group = "no KO") %>%
    arrange(desc(effsize)) %>%
    pull(group2)
  
  curves_point1 <- curves_point %>%
    mutate(type = factor(type, levels = c("no KO", cohens_d_order_pt)))
  
  stats_pt <- curves_point1 %>%
    rstatix::t_test(diff_point ~ type, ref.group = "no KO", paired = TRUE) %>%
    rstatix::adjust_pvalue(method = "BH") %>%
    rstatix::add_significance() %>%
    mutate(y.position = max(curves_point1$diff_point, na.rm = TRUE) * 1.05)
  
  keep_types1 <- c("no KO", stats_pt$group2[stats_pt$p.adj < 0.05])
  
  p1 <- ggboxplot(curves_point1 %>% filter(type %in% keep_types1),
                  x = "type", y = "diff_point", add = "jitter") +
    stat_pvalue_manual(stats_pt %>% filter(p.adj < 0.05),
                       label = "p.adj.signif", remove.bracket = TRUE) +
    geom_hline(yintercept = mean(curves_point1$diff_point[curves_point1$type == "no KO"], na.rm = TRUE),
               lty = "dashed", color = "red", lwd = 2) +
    coord_flip() +
    labs(title = paste0("Point diff @ IFNa=", ifna_extreme, " — ", phen),
         subtitle = "Reference: no KO")
  
  if (!is.null(out1)) {
    ggsave(out1, plot = p1, width = width_cm, height = height_cm, units = "cm", dpi = dpi)
  }
  
  ## Plot 2: restrict to significant types, then compare to "TP63 KO"
  sig_types_pt <- unique(c("TP63 KO", stats_pt$group2[stats_pt$p.adj < 0.05]))
  
  curves_point_sub <- curves_point %>%
    filter(type %in% sig_types_pt) %>%
    mutate(type = droplevels(factor(type)))
  
  cohens_d_order_pt2 <- curves_point_sub %>%
    rstatix::cohens_d(diff_point ~ type, ref.group = "TP63 KO") %>%
    arrange(desc(effsize)) %>%
    pull(group2)
  
  curves_point2 <- curves_point_sub %>%
    mutate(type = factor(type, levels = c("TP63 KO", cohens_d_order_pt2)))
  
  stats_pt2 <- curves_point2 %>%
    rstatix::t_test(diff_point ~ type, ref.group = "TP63 KO", paired = TRUE) %>%
    rstatix::adjust_pvalue(method = "BH") %>%
    rstatix::add_significance() %>%
    mutate(y.position = max(curves_point2$diff_point, na.rm = TRUE) * 1.05)
  
  p2 <- ggboxplot(curves_point2, x = "type", y = "diff_point", add = "jitter") +
    stat_pvalue_manual(stats_pt2, label = "p.adj.signif", remove.bracket = TRUE) +
    geom_hline(yintercept = mean(curves_point2$diff_point[curves_point2$type == "TP63 KO"], na.rm = TRUE),
               lty = "dashed", color = "red", lwd = 2) +
    coord_flip() +
    labs(title = paste0("Point diff @ IFNa=", ifna_extreme, " — ", phen),
         subtitle = "Reference: TP63 KO (subset to significant vs no KO)")
  
  if (!is.null(out2)) {
    ggsave(out2, plot = p2, width = width_cm, height = height_cm, units = "cm", dpi = dpi)
  }
  
  list(curves_point = curves_point, p_noKO = p1, p_TP63 = p2,
       stats_noKO = stats_pt, stats_TP63 = stats_pt2)
}

# ---- run for both phenotypes and save
res_mas <- make_point_plots(
  phen = "MAS",
  out1 = "../figures/pointdiff_noKO_MAS_24hr.pdf",
  out2 = "../figures/pointdiff_TP63_MAS_24hr.pdf"
)

res_fib <- make_point_plots(
  phen = "Fibrosis stage",  # change string if your label differs
  out1 = "../figures/pointdiff_noKO_FibrosisStage_24hr.pdf",
  out2 = "../figures/pointdiff_TP63_FibrosisStage_24hr.pdf"
)

### Comparison of AUCs of the curves---------
compute_auc_curves <- function(phen) {
  
  dat <- bind_rows(
    bind_rows(Ysame, Yopp) %>%
      ungroup() %>%
      add_type() %>%
      filter(!is.na(IFNa), !is.na(TGFb), phenotype == phen) %>%
      select(model_no, IFNa, score, stimulation, type) %>%
      distinct(),
    Ystim_no_ko %>%
      ungroup() %>%
      add_type(default_label = "no KO") %>%
      filter(!is.na(IFNAR1), !is.na(TGFRB1), phenotype == phen) %>%
      select(model_no, IFNa = IFNAR1, score, stimulation, type) %>%
      distinct()
  ) %>%
    select(model_no, x = IFNa, score, stimulation, type) %>%
    tidyr::pivot_wider(names_from = stimulation, values_from = score) %>%
    # require both stim values at a given x
    tidyr::drop_na(`IFNAR1*TGFBR1<0`, `IFNAR1*TGFBR1>0`) %>%
    mutate(y = abs(`IFNAR1*TGFBR1<0` - `IFNAR1*TGFBR1>0`))
  
  curves <- dat %>%
    group_by(model_no, type) %>%
    summarise(
      AUC = MESS::auc(x[order(x)], y[order(x)]),
      x_range = max(x, na.rm = TRUE) - min(x, na.rm = TRUE),
      AUC_norm = AUC / x_range,
      .groups = "drop"
    ) %>%
    mutate(phenotype = phen)
  
  curves
}

make_auc_plots <- function(phen,
                           out1 = NULL,
                           out2 = NULL,
                           width_cm = 16,
                           height_cm = 20,
                           dpi = 600) {
  
  curves <- compute_auc_curves(phen)
  
  # ---- Plot 1: vs no KO ----
  if (!("no KO" %in% curves$type)) stop("No 'no KO' rows found for phenotype = ", phen)
  
  cohens_d_order <- curves %>%
    rstatix::cohens_d(AUC ~ type, ref.group = "no KO") %>%
    arrange(desc(effsize)) %>%
    pull(group2)
  
  curves1 <- curves %>%
    mutate(type = factor(type, levels = c("no KO", cohens_d_order)))
  
  stats <- curves1 %>%
    rstatix::t_test(AUC ~ type, ref.group = "no KO", paired = TRUE) %>%
    rstatix::adjust_pvalue(method = "BH") %>%
    rstatix::add_significance() %>%
    mutate(y.position = max(curves1$AUC, na.rm = TRUE) * 1.05)
  
  keep1 <- c("no KO", stats$group2[stats$p.adj < 0.05])
  
  p1 <- ggboxplot(curves1 %>% filter(type %in% keep1),
                  x = "type", y = "AUC", add = "jitter") +
    stat_pvalue_manual(stats %>% filter(p.adj < 0.05),
                       label = "p.adj.signif", remove.bracket = TRUE) +
    geom_hline(yintercept = mean(curves1$AUC[curves1$type == "no KO"], na.rm = TRUE),
               lty = "dashed", color = "red", lwd = 2) +
    coord_flip() +
    labs(title = paste0("AUC— ", phen),
         subtitle = "Reference: no KO")
  
  if (!is.null(out1)) ggsave(out1, plot = p1, width = width_cm, height = height_cm, units = "cm", dpi = dpi)
  
  # ---- Plot 2: restrict to sig vs no KO, then compare vs TP63 KO ----
  sig_types <- unique(c("TP63 KO", stats$group2[stats$p.adj < 0.05]))
  curves_sub <- curves %>%
    filter(type %in% sig_types) %>%
    mutate(type = droplevels(factor(type)))  # avoid droplevels(character) crash
  
  if (!("TP63 KO" %in% curves_sub$type)) {
    warning("No 'TP63 KO' rows found in subset for phenotype = ", phen, " — skipping Plot 2.")
    return(list(curves = curves, p_noKO = p1, p_TP63 = NULL, stats_noKO = stats, stats_TP63 = NULL))
  }
  
  cohens_d_order2 <- curves_sub %>%
    rstatix::cohens_d(AUC ~ type, ref.group = "TP63 KO") %>%
    arrange(desc(effsize)) %>%
    pull(group2)
  
  curves2 <- curves_sub %>%
    mutate(type = factor(type, levels = c("TP63 KO", cohens_d_order2)))
  
  stats2 <- curves2 %>%
    rstatix::t_test(AUC ~ type, ref.group = "TP63 KO", paired = TRUE) %>%
    rstatix::adjust_pvalue(method = "BH") %>%
    rstatix::add_significance() %>%
    mutate(y.position = max(curves2$AUC, na.rm = TRUE) * 1.05)
  
  p2 <- ggboxplot(curves2, x = "type", y = "AUC", add = "jitter") +
    stat_pvalue_manual(stats2, label = "p.adj.signif", remove.bracket = TRUE) +
    geom_hline(yintercept = mean(curves2$AUC[curves2$type == "TP63 KO"], na.rm = TRUE),
               lty = "dashed", color = "red", lwd = 2) +
    coord_flip() +
    labs(title = paste0("AUC— ", phen),
         subtitle = "Reference: TP63 KO (subset to significant vs no KO)")
  
  if (!is.null(out2)) ggsave(out2, plot = p2, width = width_cm, height = height_cm, units = "cm", dpi = dpi)
  
  list(curves = curves, p_noKO = p1, p_TP63 = p2, stats_noKO = stats, stats_TP63 = stats2)
}
res_auc_mas <- make_auc_plots(
  phen = "MAS",
  out1 = "../figures/auc_noKO_MAS_24hr.pdf",
  out2 = "../figures/auc_TP63_MAS_24hr.pdf"
)

res_auc_fib <- make_auc_plots(
  phen = "Fibrosis stage",
  out1 = "../figures/auc_noKO_FibrosisStage_24hr.pdf",
  out2 = "../figures/auc_TP63_FibrosisStage_24hr.pdf"
)

### Statstical comparison of 2D curves AUC and control line----
compute_auc_ctrl <- function(phen, baseline_col) {
  
  curves_ctrl <- bind_rows(Ysame, Yopp) %>%
    ungroup() %>%
    add_type() %>%
    filter(!is.na(IFNa), !is.na(TGFb), phenotype == phen) %>%
    select(model_no, x = IFNa, score, type) %>%
    distinct() %>%
    left_join(
      Ycontrol %>% transmute(model_no, baseline = .data[[baseline_col]]),
      by = "model_no"
    ) %>%
    mutate(abs_diff = abs(score - baseline)) %>%
    group_by(model_no, type) %>%
    summarise(
      AUC      = MESS::auc(x[order(x)], abs_diff[order(x)]),
      x_range  = max(x, na.rm = TRUE) - min(x, na.rm = TRUE),
      AUC_norm = AUC / x_range,
      .groups = "drop"
    ) %>%
    mutate(phenotype = phen)
  
  curves_ctrl
}

make_auc_ctrl_plot <- function(phen, baseline_col, out = NULL,
                               width_cm = 16, height_cm = 20, dpi = 600) {
  
  curves_ctrl <- compute_auc_ctrl(phen, baseline_col)
  
  if (!("TP63 KO" %in% curves_ctrl$type)) stop("No 'TP63 KO' rows found for phenotype = ", phen)
  
  cohens_d_order_ctrl <- curves_ctrl %>%
    rstatix::cohens_d(AUC_norm ~ type, ref.group = "TP63 KO") %>%
    arrange(desc(effsize)) %>%
    pull(group2)
  
  curves_ctrl2 <- curves_ctrl %>%
    mutate(type = factor(type, levels = c("TP63 KO", cohens_d_order_ctrl)))
  
  stats_ctrl <- curves_ctrl2 %>%
    rstatix::t_test(AUC_norm ~ type, ref.group = "TP63 KO", paired = TRUE) %>%
    rstatix::adjust_pvalue(method = "BH") %>%
    rstatix::add_significance() %>%
    mutate(y.position = max(curves_ctrl2$AUC_norm, na.rm = TRUE) * 1.05)
  
  keep <- c("TP63 KO", stats_ctrl$group2[stats_ctrl$p.adj < 0.05])
  
  p <- ggboxplot(curves_ctrl2 %>% filter(type %in% keep),
                 x = "type", y = "AUC_norm", add = "jitter") +
    stat_pvalue_manual(stats_ctrl %>% filter(p.adj < 0.05),
                       label = "p.adj.signif", remove.bracket = TRUE) +
    geom_hline(yintercept = mean(curves_ctrl2$AUC_norm[curves_ctrl2$type == "TP63 KO"], na.rm = TRUE),
               lty = "dashed", color = "red", lwd = 2) +
    coord_flip() +
    labs(title = paste0("AUC_norm(|score - baseline|) — ", phen),
         subtitle = paste0("Baseline: Ycontrol$", baseline_col, " (Reference: TP63 KO)"))
  
  if (!is.null(out)) ggsave(out, plot = p, width = width_cm, height = height_cm, units = "cm", dpi = dpi)
  
  list(curves_ctrl = curves_ctrl2, stats_ctrl = stats_ctrl, p_ctrl = p)
}
res_ctrl_mas <- make_auc_ctrl_plot(
  phen = "MAS",
  baseline_col = "NAS",
  out = "../figures/auc_norm_control_TP63_MAS_24hr.pdf"
)

res_ctrl_fib <- make_auc_ctrl_plot(
  phen = "Fibrosis stage",
  baseline_col = "fibrosis",  # <-- change to your real Ycontrol column
  out = "../figures/auc_norm_control_TP63_FibrosisStage_24hr.pdf"
)


### Compare with IFNA=0 and TGFB=0 baseline when there is no KO-------------
# Step A: Build network + map KO type -> receptor class (IFNAR1 / TGFBR1 / BOTH)
network <- read.table("../preprocessing/preprocessed_data/PKN/Hepatocytes-PKN-Model.tsv",
                      header = TRUE, sep = "\t", stringsAsFactors = FALSE)
annotations <- read.table("../preprocessing/preprocessed_data/PKN/Hepatocytes-PKN-Annotation.tsv",
                          header = TRUE, sep = "\t", stringsAsFactors = FALSE)

g <- graph_from_data_frame(network[, c("source", "target")], directed = TRUE)

IFNAR1_id <- "P17181"
TGFBR1_id <- "P36897"

downstream_IFNAR1 <- V(g)$name[subcomponent(g, IFNAR1_id, mode = "out")]
downstream_TGFBR1 <- V(g)$name[subcomponent(g, TGFBR1_id, mode = "out")]
both <- intersect(downstream_IFNAR1, downstream_TGFBR1)

type_receptor <- Yall %>%
  distinct(type) %>%
  filter(!is.na(type)) %>%
  mutate(gene = gsub(" KO$", "", type)) %>%
  left_join(
    bind_rows(
      tibble(code = setdiff(downstream_IFNAR1, both), receptor = "IFNAR1"),
      tibble(code = setdiff(downstream_TGFBR1, both), receptor = "TGFBR1"),
      tibble(code = both, receptor = "BOTH")
    ) %>%
      left_join(annotations %>% select(code, name), by = "code") %>%
      select(name, receptor) %>%
      distinct(),
    by = c("gene" = "name")
  ) %>%
  mutate(receptor = ifelse(is.na(receptor), "IFNAR1", receptor)) %>%
  select(type, receptor) %>%
  distinct()


# Step B: Rebuild raw 1D axis curves from Yall (you said raw_curves was lost)
make_raw_curves_axis <- function(phen, Yall, type_receptor) {
  
  base <- Yall %>%
    filter(phenotype == phen, !is.na(IFNa), !is.na(TGFb), !is.na(type)) %>%
    mutate(
      IFNa = as.numeric(IFNa),
      TGFb = as.numeric(TGFb)
    ) %>%
    left_join(type_receptor, by = "type") %>%
    mutate(receptor = ifelse(is.na(receptor), "IFNAR1", receptor))
  
  raw_ifna <- base %>%
    filter(TGFb == 0, receptor %in% c("IFNAR1", "BOTH")) %>%
    transmute(model_no, type, receptor, receptor_used = "IFNAR1", x = IFNa, score)
  
  raw_tgfb <- base %>%
    filter(IFNa == 0, receptor %in% c("TGFBR1", "BOTH")) %>%
    transmute(model_no, type, receptor, receptor_used = "TGFBR1", x = TGFb, score)
  
  raw_curves <- bind_rows(raw_ifna, raw_tgfb) %>%
    filter(is.finite(x), is.finite(score)) %>%
    arrange(receptor_used, type, model_no, x)
  
  # auc needs >=2 x points per curve
  raw_curves %>%
    group_by(model_no, type, receptor_used) %>%
    filter(n_distinct(x) >= 2) %>%
    ungroup()
}

compute_curves_ctrl2d_baseline0 <- function(phen, Yall, Ystim_no_ko, type_receptor) {
  
  raw_curves <- make_raw_curves_axis(phen, Yall, type_receptor)
  
  baseline_IFNAR1 <- Ystim_no_ko %>%
    filter(phenotype == phen, !is.na(IFNAR1), !is.na(TGFRB1), IFNAR1 == 0) %>%
    group_by(model_no) %>%
    summarise(baseline = mean(score, na.rm = TRUE), .groups = "drop")
  
  baseline_TGFBR1 <- Ystim_no_ko %>%
    filter(phenotype == phen, !is.na(IFNAR1), !is.na(TGFRB1), TGFRB1 == 0) %>%
    group_by(model_no) %>%
    summarise(baseline = mean(score, na.rm = TRUE), .groups = "drop")
  
  curves_ctrl2d <- bind_rows(
    raw_curves %>%
      filter(receptor_used == "IFNAR1") %>%
      left_join(baseline_IFNAR1, by = "model_no"),
    raw_curves %>%
      filter(receptor_used == "TGFBR1") %>%
      left_join(baseline_TGFBR1, by = "model_no")
  ) %>%
    filter(!is.na(baseline)) %>%
    mutate(abs_diff = abs(score - baseline)) %>%
    group_by(model_no, type, receptor, receptor_used) %>%
    summarise(
      AUC      = MESS::auc(x[order(x)], abs_diff[order(x)]),
      x_range  = max(x, na.rm = TRUE) - min(x, na.rm = TRUE),
      AUC_norm = AUC / x_range,
      .groups  = "drop"
    ) %>%
    filter(is.finite(AUC_norm))
  
  curves_ctrl2d
}
order_types_by_d <- function(df, ref, value_col = "AUC_norm") {
  ref_df <- df %>% filter(type == ref) %>% select(model_no, ref_val = all_of(value_col))
  
  df %>%
    filter(type != ref) %>%
    inner_join(ref_df, by = "model_no") %>%
    group_by(type) %>%
    summarise(
      d = (mean(.data[[value_col]], na.rm = TRUE) - mean(ref_val, na.rm = TRUE)) /
        sqrt((var(.data[[value_col]], na.rm = TRUE) + var(ref_val, na.rm = TRUE)) / 2),
      .groups = "drop"
    ) %>%
    arrange(desc(d)) %>%
    pull(type)
}
plot_all_types_by_receptor <- function(curves_ctrl2d, phen, receptor_label, ref_type,
                                       width_cm = 16, height_cm = 20) {
  
  data <- curves_ctrl2d %>%
    filter(receptor_used == receptor_label) %>%
    filter(!is.na(AUC_norm))
  
  if (!(ref_type %in% data$type)) {
    stop("Reference type '", ref_type, "' not found in receptor_used='",
         receptor_label, "' for phenotype='", phen, "'.")
  }
  
  ord <- order_types_by_d(data, ref = ref_type, value_col = "AUC_norm")
  data <- data %>%
    mutate(type = factor(as.character(type), levels = c(ref_type, ord)))
  
  ref_mean <- mean(data$AUC_norm[as.character(data$type) == ref_type], na.rm = TRUE)
  
  p <- ggboxplot(
    data,
    x = "type", y = "AUC_norm",
    add = "jitter",
    title = paste0("Baseline: stim=0 (no KO) | ", receptor_label, " downstream | ", phen)
  ) +
    geom_hline(yintercept = ref_mean, lty = "dashed", color = "red", lwd = 2) +
    coord_flip() +
    labs(subtitle = paste0("Reference = ", ref_type, " (red dashed = mean)")) +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold"),
      axis.text.y = element_text(size = 7)  # shrink labels if many KOs
    )
  
  p
}
outdir <- "../figures"
run_and_save_baseline0_plots <- function(phen) {
  
  curves_ctrl2d <- compute_curves_ctrl2d_baseline0(
    phen = phen,
    Yall = Yall,
    Ystim_no_ko = Ystim_no_ko,
    type_receptor = type_receptor
  )
  
  # IFNAR1 plot: compare all IFNAR1/BOTH types to STAT2 KO
  p_ifn <- plot_all_types_by_receptor(
    curves_ctrl2d = curves_ctrl2d,
    phen = phen,
    receptor_label = "IFNAR1",
    ref_type = "STAT2 KO"
  )
  
  # TGFBR1 plot: compare all TGFBR1/BOTH types to TP63 KO
  p_tgf <- plot_all_types_by_receptor(
    curves_ctrl2d = curves_ctrl2d,
    phen = phen,
    receptor_label = "TGFBR1",
    ref_type = "TP63 KO"
  )
  
  phen_tag <- gsub("[^A-Za-z0-9]+", "", phen)
  
  ggsave(file.path(outdir, paste0("baseline0_AUCnorm_IFNAR1_", phen_tag, "_24hr.pdf")),
         plot = p_ifn, width = 16, height = 20, units = "cm", dpi = 600)
  
  ggsave(file.path(outdir, paste0("baseline0_AUCnorm_TGFBR1_", phen_tag, "_24hr.pdf")),
         plot = p_tgf, width = 16, height = 20, units = "cm", dpi = 600)
  
  list(curves_ctrl2d = curves_ctrl2d, p_ifnar = p_ifn, p_tgfbr1 = p_tgf)
}

res_mas0 <- run_and_save_baseline0_plots("MAS")
res_fib0 <- run_and_save_baseline0_plots("Fibrosis stage")


# --- Baseline0 3D volume: IFNAR1=0 or TGFRB1=0 (no-KO), receptor-restricted -----------
# Build 2D grid with abs_diff to the appropriate baseline (per receptor_used)
compute_grid_absdiff_baseline0_3d <- function(phen, receptor_label, Yall, Ystim_no_ko, type_receptor) {
  
  base <- Yall %>%
    filter(phenotype == phen, !is.na(IFNa), !is.na(TGFb), !is.na(type)) %>%
    mutate(IFNa = as.numeric(IFNa), TGFb = as.numeric(TGFb)) %>%
    left_join(type_receptor, by = "type") %>%
    mutate(receptor = ifelse(is.na(receptor), "IFNAR1", receptor))
  
  # model-wise baselines from the no-KO grid:
  baseline_IFNAR1 <- Ystim_no_ko %>%
    filter(phenotype == phen, !is.na(IFNAR1), IFNAR1 == 0) %>%
    group_by(model_no) %>%
    summarise(baseline = mean(score, na.rm = TRUE), .groups = "drop")
  
  baseline_TGFBR1 <- Ystim_no_ko %>%
    filter(phenotype == phen, !is.na(TGFRB1), TGFRB1 == 0) %>%
    group_by(model_no) %>%
    summarise(baseline = mean(score, na.rm = TRUE), .groups = "drop")
  
  if (receptor_label == "IFNAR1") {
    base %>%
      filter(receptor %in% c("IFNAR1", "BOTH")) %>%
      mutate(receptor_used = "IFNAR1") %>%
      left_join(baseline_IFNAR1, by = "model_no") %>%
      filter(!is.na(baseline)) %>%
      mutate(abs_diff = abs(score - baseline))
  } else if (receptor_label == "TGFBR1") {
    base %>%
      filter(receptor %in% c("TGFBR1", "BOTH")) %>%
      mutate(receptor_used = "TGFBR1") %>%
      left_join(baseline_TGFBR1, by = "model_no") %>%
      filter(!is.na(baseline)) %>%
      mutate(abs_diff = abs(score - baseline))
  } else {
    stop("receptor_label must be 'IFNAR1' or 'TGFBR1'")
  }
}

# Make one plot (one receptor_used) for one phenotype
make_volume_plot_baseline0 <- function(phen, receptor_label, ref_type, outpath,
                                       width_cm = 16, height_cm = 20, dpi = 600,
                                       value_col = c("VOL_norm","VOL")) {
  
  value_col <- match.arg(value_col)
  
  grid_clean <- compute_grid_absdiff_baseline0_3d(
    phen = phen,
    receptor_label = receptor_label,
    Yall = Yall,
    Ystim_no_ko = Ystim_no_ko,
    type_receptor = type_receptor
  )
  
  curves_3d <- grid_clean %>%
    group_by(model_no, type, receptor_used) %>%
    arrange(IFNa, TGFb) %>%
    group_modify(compute_vol) %>%
    ungroup() %>%
    filter(!is.na(.data[[value_col]]))
  
  if (!(ref_type %in% curves_3d$type)) {
    stop("Reference type '", ref_type, "' not found for receptor_used='",
         receptor_label, "' and phenotype='", phen, "'.")
  }
  
  # order by Cohen's d vs reference
  cohens_order <- curves_3d %>%
    rstatix::cohens_d(reformulate("type", response = value_col), ref.group = ref_type) %>%
    arrange(desc(effsize)) %>%
    pull(group2)
  
  curves_3d <- curves_3d %>%
    mutate(type = factor(as.character(type), levels = c(ref_type, cohens_order)))
  
  stats_3d <- curves_3d %>%
    rstatix::t_test(reformulate("type", response = value_col), ref.group = ref_type, paired = FALSE) %>%
    rstatix::adjust_pvalue(method = "BH") %>%
    rstatix::add_significance()
  
  # label height
  y_max <- max(curves_3d[[value_col]], na.rm = TRUE)
  stats_3d <- stats_3d %>% mutate(y.position = y_max * 1.05)
  
  keep_types <- c(ref_type, stats_3d$group2[stats_3d$p.adj < 0.05])
  
  ref_mean <- mean(curves_3d[[value_col]][as.character(curves_3d$type) == ref_type], na.rm = TRUE)
  
  p <- ggpubr::ggboxplot(
    curves_3d %>% filter(type %in% keep_types),
    x = "type", y = value_col, add = "jitter"
  ) +
    ggpubr::stat_pvalue_manual(
      stats_3d %>% filter(p.adj < 0.05),
      label = "p.adj.signif",
      remove.bracket = TRUE
    ) +
    geom_hline(yintercept = ref_mean, lty = "dashed", color = "red", lwd = 2) +
    coord_flip() +
    labs(
      title = paste0("Baseline: ", receptor_label, "=0 (no KO) | ", receptor_label,
                     " downstream | ", phen),
      subtitle = paste0("Reference = ", ref_type, " (red dashed = mean)")
    ) +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"),
          axis.text.y = element_text(size = 7))
  
  ggsave(outpath, plot = p, width = width_cm, height = height_cm, units = "cm", dpi = dpi)
  p
}

# Run + save both receptor plots per phenotype
run_and_save_baseline0_volume_plots <- function(phen, outdir = "../figures",
                                                width_cm = 16, height_cm = 20, dpi = 600,
                                                value_col = "VOL_norm") {
  
  phen_tag <- gsub("[^A-Za-z0-9]+", "", phen)
  
  p_ifn <- make_volume_plot_baseline0(
    phen = phen,
    receptor_label = "IFNAR1",
    ref_type = "STAT2 KO",
    outpath = file.path(outdir, paste0("baseline0_VOLnorm_IFNAR1_", phen_tag, "_24hr.pdf")),
    width_cm = width_cm, height_cm = height_cm, dpi = dpi,
    value_col = value_col
  )
  
  p_tgf <- make_volume_plot_baseline0(
    phen = phen,
    receptor_label = "TGFBR1",
    ref_type = "TP63 KO",
    outpath = file.path(outdir, paste0("baseline0_VOLnorm_TGFBR1_", phen_tag, "_24hr.pdf")),
    width_cm = width_cm, height_cm = height_cm, dpi = dpi,
    value_col = value_col
  )
  
  list(p_ifnar = p_ifn, p_tgfbr1 = p_tgf)
}

# ---- run
res_mas_vol0 <- run_and_save_baseline0_volume_plots("MAS")
res_fib_vol0 <- run_and_save_baseline0_volume_plots("Fibrosis stage")

### Visualize curves-----------------
# mas_plot <- ggplot(Yall %>% filter(phenotype=='MAS') %>% select(IFNa,TGFb,mu,se,phenotype,type) %>% unique(),
#                    aes(x=IFNa,y=TGFb,fill=mu))+
#   geom_tile()+
#   scale_fill_gradient2(midpoint = mean_control_mas)+
#   labs(fill='MAS',x = 'stimulation of IFNAR1',y='stimulation of TGFBR1')+
#   scale_x_continuous(n.breaks = 10)+
#   scale_y_continuous(n.breaks = 10)+
#   geom_hline(yintercept = 0,lwd=1,color='black')+
#   geom_vline(xintercept = 0,lwd=1,color='black')+
#   facet_wrap(~type)
# mas_plot <- add_theme(mas_plot) + 
#   theme(legend.position = 'top',text = element_text(size=20),
#         panel.border = element_blank(),
#         panel.grid = element_blank())
# print(mas_plot)
# ggsave(filename = "../figures/expanded_ifna_tgfb_combo_mas_plot_multiKOs.pdf", plot = mas_plot, 
#        device = cairo_pdf,
#        units = "cm", width = 28, height = 24)
# 
# fib_plot <- ggplot(Yall %>% filter(phenotype!='MAS') %>% select(IFNa,TGFb,mu,se,phenotype,type) %>% unique(),
#                    aes(x=IFNa,y=TGFb,fill=mu))+
#   geom_tile()+
#   scale_fill_gradient2(midpoint = mean_control_fib)+
#   labs(fill='Fibrosis stage',x = 'stimulation of IFNAR1',y='stimulation of TGFBR1')+
#   scale_x_continuous(n.breaks = 10)+
#   scale_y_continuous(n.breaks = 10)+
#   geom_hline(yintercept = 0,lwd=1,color='black')+
#   geom_vline(xintercept = 0,lwd=1,color='black')+
#   facet_wrap(~type)
# fib_plot <- add_theme(fib_plot) +
#   theme(legend.position = 'top',text = element_text(size=20),
#         panel.border = element_blank(),
#         panel.grid = element_blank())
# print(fib_plot)
# ggsave(filename = "../figures/expanded_ifna_tgfb_combo_fibrosis_plot_multiKOs.pdf", plot = fib_plot, 
#        device = cairo_pdf,
#        units = "cm", width = 16, height = 12)

res_mas <- make_ifna_facet_plot(
  phenotype_value = "MAS",
  Ysame = Ysame %>% filter(type %in% c('no KO',
                                       'STAT2 KO',
                                       'STAT3 KO',
                                       'SMAD3 KO',
                                       'TP63 KO',
                                       'TP53 KO',
                                       'JUN KO',
                                       'MYC KO')), 
  Yopp = Yopp %>% filter(type %in% c('no KO',
                                     'STAT2 KO',
                                     'STAT3 KO',
                                     'SMAD3 KO',
                                     'TP63 KO',
                                     'TP53 KO',
                                     'JUN KO',
                                     'MYC KO')), 
  Ystim_no_ko = Ystim_no_ko,
  mean_control = mean_control_mas,
  se_control = se_control_mas,
  type_labels = c("TP63 KO" = "TP63 KO", "no KO" = "no KO"),
  p_adjust_method = "bonferroni",
  y_upper = 11,
  y_lower = -2,
  legend_text_size = 14,
  legend_title_size =14
)
res_mas$plot
ggsave(filename = "../figures/ifna_tgfb_withKOs_combo_mas_line_plot_24h.pdf", plot = res_mas$plot,
       device = cairo_pdf,
       units = "cm", width = 16, height = 16)

# Fibrosis stage (pass fibrosis untreated band stats you have)
res_fib <- make_ifna_facet_plot(
  phenotype_value = "Fibrosis stage",
  Ysame = Ysame %>% filter(type %in% c('no KO',
                                       'STAT2 KO',
                                       'STAT3 KO',
                                       'SMAD3 KO',
                                       'TP63 KO',
                                       'TP53 KO',
                                       'JUN KO',
                                       'MYC KO')), 
  Yopp = Yopp %>% filter(type %in% c('no KO',
                                     'STAT2 KO',
                                     'STAT3 KO',
                                     'SMAD3 KO',
                                     'TP63 KO',
                                     'TP53 KO',
                                     'JUN KO',
                                     'MYC KO')), 
  Ystim_no_ko = Ystim_no_ko,
  mean_control = mean_control_fib,
  se_control = se_control_fib,
  y_upper = 6,
  y_lower = -2,
  legend_text_size = 14,
  legend_title_size =14
)
res_fib$plot
ggsave(filename = "../figures/ifna_tgfb_withKOs_combo_fibrosis_line_plot_24h.pdf", plot = res_fib$plot, 
       device = cairo_pdf,
       units = "cm", width = 16, height = 16)

###### ONE PLOT ALL
types_keep <- c("no KO","STAT2 KO","STAT3 KO","SMAD3 KO","TP63 KO","TP53 KO","JUN KO","MYC KO")
# 1) KO curves from Yopp (already opposing)
opp_ko <- Yopp %>%
  filter(type %in% setdiff(types_keep, "no KO")) %>%
  transmute(
    phenotype,
    model_no,
    IFNa = as.numeric(IFNa),
    score,
    type
  )

# 2) no-KO curve from Ystim_no_ko on the opposing diagonal: IFNAR1 == -TGFRB1
tol <- 1e-9
opp_no_ko <- Ystim_no_ko %>%
  filter(!is.na(IFNAR1), !is.na(TGFRB1)) %>%
  filter(abs(as.numeric(IFNAR1) + as.numeric(TGFRB1)) < tol) %>%
  transmute(
    phenotype,
    model_no,
    IFNa = as.numeric(IFNAR1),
    score,
    type = "no KO"
  )

opp_all <- bind_rows(opp_no_ko, opp_ko) %>%
  filter(phenotype %in% c("MAS", "Fibrosis stage")) %>%
  mutate(
    phenotype = factor(phenotype, levels = c("MAS", "Fibrosis stage")),
    type      = factor(type, levels = types_keep)
  )

# mean ± SE per (phenotype, type, IFNa)
plot_df <- opp_all %>%
  group_by(phenotype, type, IFNa) %>%
  summarise(
    mu = mean(score, na.rm = TRUE),
    se = sd(score, na.rm = TRUE) / sqrt(sum(!is.na(score))),
    .groups = "drop"
  )

# control band per phenotype (your precomputed numbers)
control_band <- tibble(
  phenotype = factor(c("MAS", "Fibrosis stage"), levels = c("MAS", "Fibrosis stage")),
  mean_control = c(mean_control_mas, mean_control_fib),
  se_control   = c(se_control_mas,   se_control_fib)
) %>%
  mutate(
    ymin = mean_control - se_control,
    ymax = mean_control + se_control
  )

# --- color mapping: no KO black, others automatic palette
other_types <- setdiff(types_keep, "no KO")
pal <- hue_pal()(length(other_types))
col_map <- c("no KO" = "black", setNames(pal, other_types))

# Use same mapping for ribbon fill (but hide fill legend)
fill_map <- col_map

p <- ggplot(plot_df, aes(x = IFNa, y = mu, color = type, group = type)) +
  
  # control band (no legend)
  geom_rect(
    data = control_band,
    aes(xmin = -Inf, xmax = Inf, ymin = ymin, ymax = ymax),
    inherit.aes = FALSE,
    fill = "grey70",
    alpha = 0.20,
    color = NA,
    show.legend = FALSE
  ) +
  
  # control mean (no legend)
  geom_hline(
    data = control_band,
    aes(yintercept = mean_control),
    inherit.aes = FALSE,
    linetype = "dashed",
    linewidth = 0.7,
    color = "grey35",
    show.legend = FALSE
  ) +
  
  # curve ribbons (shaded area around curves), no extra legend
  geom_ribbon(
    aes(ymin = mu - se, ymax = mu + se, fill = type),
    alpha = 0.12,
    color = NA,
    show.legend = FALSE
  ) +
  
  geom_line(linewidth = 1.0) +
  
  facet_wrap(~phenotype, ncol = 2, scales = "free_y") +
  
  scale_color_manual(values = col_map, name = "KO type") +
  scale_fill_manual(values = fill_map, guide = "none") +
  
  labs(
    title = "Opposing stimulation only (IFNAR1*TGFBR1 < 0)",
    x = "IFNa stimulation strength",
    y = "Score"
  ) +
  
  theme_bw(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    legend.position = "top",
    legend.box = "vertical"
  )

p

ggsave(
  "../figures/opposing_only_types_MAS_vs_Fibrosis_with_curve_ribbons_noKO_black_24h.pdf",
  plot = p, device = cairo_pdf,
  units = "cm", width = 24, height = 14, dpi = 600
)
