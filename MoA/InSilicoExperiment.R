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
# Yinsilico_centered <- left_join(Ycontrol,Yinsilico,by = 'model_no')
# Yinsilico_centered <- Yinsilico_centered %>% mutate(NAS = NAS.y - NAS.x)
# Yinsilico_centered <- Yinsilico_centered %>% mutate(fibrosis = fibrosis.y-fibrosis.x)
# Yinsilico_centered <- Yinsilico_centered %>% 
#   select(c("IFNAR1"="IFNa"),c("TGFBR1"="TGFb"),model_no,NAS,fibrosis)
Yinsilico <- Yinsilico %>% gather('phenotype','score',-model_no,-IFNa,-TGFb,-type) %>%
  group_by(phenotype,IFNa,TGFb) %>% mutate(mu = mean(score)) %>% mutate(se=sd(score)/sqrt(50))

### Visualize simulation
Yall <- Yinsilico %>%
  mutate(phenotype = ifelse(phenotype=='NAS','MAS','Fibrosis stage'))
Yall <- Yall %>% mutate(IFNa=as.numeric(IFNa)) %>% mutate(TGFb=as.numeric(TGFb))



mas_plot <- ggplot(Yall %>% filter(phenotype=='MAS') %>% select(IFNa,TGFb,mu,se,phenotype,type) %>% unique(),
                   aes(x=IFNa,y=TGFb,fill=mu))+
  geom_tile()+
  scale_fill_gradient2(midpoint = mean_control_mas)+
  labs(fill='MAS',x = 'stimulation of IFNAR1',y='stimulation of TGFBR1')+
  scale_x_continuous(n.breaks = 10)+
  scale_y_continuous(n.breaks = 10)+
  geom_hline(yintercept = 0,lwd=1,color='black')+
  geom_vline(xintercept = 0,lwd=1,color='black')+
  facet_wrap(~type)
mas_plot <- add_theme(mas_plot) + 
  theme(legend.position = 'top',text = element_text(size=20),
        panel.border = element_blank(),
        panel.grid = element_blank())
print(mas_plot)
ggsave(filename = "../figures/expanded_ifna_tgfb_combo_mas_plot_multiKOs_24hr.pdf", plot = mas_plot, 
       device = cairo_pdf,
       units = "cm", width = 28, height = 24)

fib_plot <- ggplot(Yall %>% filter(phenotype!='MAS') %>% select(IFNa,TGFb,mu,se,phenotype,type) %>% unique(),
                   aes(x=IFNa,y=TGFb,fill=mu))+
  geom_tile()+
  scale_fill_gradient2(midpoint = mean_control_fib)+
  labs(fill='Fibrosis stage',x = 'stimulation of IFNAR1',y='stimulation of TGFBR1')+
  scale_x_continuous(n.breaks = 10)+
  scale_y_continuous(n.breaks = 10)+
  geom_hline(yintercept = 0,lwd=1,color='black')+
  geom_vline(xintercept = 0,lwd=1,color='black')+
  facet_wrap(~type)
fib_plot <- add_theme(fib_plot) +
  theme(legend.position = 'top',text = element_text(size=20),
        panel.border = element_blank(),
        panel.grid = element_blank())
print(fib_plot)
ggsave(filename = "../figures/expanded_ifna_tgfb_combo_fibrosis_plot_multiKOs_24hr.pdf", plot = fib_plot, 
       device = cairo_pdf,
       units = "cm", width = 16, height = 12)

Ysame <- Yall %>% filter(IFNa==TGFb) %>% mutate(stimulation='IFNAR1*TGFBR1>0')
Yopp <- Yall %>% filter(IFNa==(-TGFb))  %>% mutate(stimulation='IFNAR1*TGFBR1<0')
res_mas <- make_ifna_facet_plot(
  phenotype_value = "MAS",
  Ysame = Ysame, Yopp = Yopp, Ystim_no_ko = Ystim_no_ko,
  mean_control = mean_control_mas,
  se_control = se_control_mas,
  type_labels = c("TP63 KO" = "TP63 KO", "no KO" = "no KO"),
  p_adjust_method = "bonferroni",
  y_upper = 8,
  y_lower = 0,
  legend_text_size = 14,
  legend_title_size =14
)
res_mas$plot
ggsave(filename = "../figures/ifna_tgfb_withKOs_combo_mas_line_plot_24hr.pdf", plot = res_mas$plot, 
       device = cairo_pdf,
       units = "cm", width = 16, height = 16)

# Fibrosis stage (pass fibrosis untreated band stats you have)
res_fib <- make_ifna_facet_plot(
  phenotype_value = "Fibrosis stage",
  Ysame = Ysame, Yopp = Yopp, Ystim_no_ko = Ystim_no_ko,
  mean_control = mean_control_fib,
  se_control = se_control_fib,
  y_upper = 4,
  y_lower = -0.6,
  legend_text_size = 14,
  legend_title_size =14
)
res_fib$plot
ggsave(filename = "../figures/ifna_tgfb_withKOs_combo_fibrosis_line_plot_24hr.pdf", plot = res_fib$plot, 
       device = cairo_pdf,
       units = "cm", width = 16, height = 16)
