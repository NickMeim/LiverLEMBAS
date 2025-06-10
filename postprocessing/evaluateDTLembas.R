library(tidyverse)
library(ggplot2)
library(ggpubr)
library(viridis)
source('plotting_functions.R')

### First compare one model in 10-fold validation with random model-------
res_model <- data.table::fread('../results/2 hour/CV/hepatocytes_dtlembas1_fold_results.csv') %>% select(-V1)
colnames(res_model)[3:4] <- c('train','test')
shuffled_model <- data.table::fread('../results/2 hour/CV/shuffledX_hepatocytes_dtlembas1_fold_results.csv')%>%
  select(fold,TF,c('shuffled Y'='r_test'))
res_all <- left_join(res_model,shuffled_model)
res_all <- res_all %>% gather('set','r',-fold,-TF)
res_all <- res_all %>% mutate(subject_id = paste0(TF,'_',fold))

random_comparison <- ggboxplot(res_all,x='set',y='r',color = 'set',outlier.shape = NA) +
  geom_jitter(aes(x=set,y=r,color = set),width = 0.01)+
  geom_line(aes(x = set, y = r, group = subject_id), 
            color = 'gray50', size = 0.5, alpha = 0.1)+
  geom_hline(yintercept = 0,linetype='solid',lwd=0.5,color='black')+
  ylab('Pearson`s correlation')+
  scale_y_continuous(breaks = seq(-0.5,1,0.1),limits = c(NA,1))+
  stat_compare_means(comparisons = list(c('train','test'),
                                        c('test','shuffled Y'),
                                        c('train','shuffled Y')),
                     method = 'wilcox.test',
                     paired = TRUE,
                     label.y = c(0.8,0.8,0.9))
random_comparison <- add_theme(random_comparison) + theme_pubr() + 
  theme(axis.text.x = element_blank(),
        axis.title.x = element_blank(),
        axis.ticks.x = element_blank(),
        axis.line.x=element_blank(),
        legend.position = 'top')
print(random_comparison)
ggsave(filename = "../figures/2 hour/dtlembas_2hr_performance_random_comparison.pdf", plot = random_comparison, units = "cm", width = 12, height = 12)
data.table::fwrite(res_all,'../results/CV_2hr_with_random_long.csv')

### First compare one model in LOOCV validation with random model-------
res_model <- rbind(data.table::fread('../results/2 hour/High/performance_hepatocytes_dtlembas1.csv') %>% 
  select(-V1,-time,-model_no),
  data.table::fread('../results/2 hour/Middle/performance_hepatocytes_dtlembas1.csv') %>% 
    select(-V1,-time,-model_no),
  data.table::fread('../results/2 hour/Low/performance_hepatocytes_dtlembas1.csv') %>% 
    select(-V1,-time,-model_no))
shuffled_model <- rbind(data.table::fread('../results/2 hour/High/performance_hepatocytes_random_dtlembas1.csv') %>% 
  select(TF,c('shuffled Y'='test'),fold),
  data.table::fread('../results/2 hour/Middle/performance_hepatocytes_random_dtlembas1.csv') %>% 
    select(TF,c('shuffled Y'='test'),fold),
  data.table::fread('../results/2 hour/Low/performance_hepatocytes_random_dtlembas1.csv') %>% 
    select(TF,c('shuffled Y'='test'),fold))

res_all <- left_join(res_model,shuffled_model)
res_all <- res_all %>% gather('set','r',-fold,-TF)
res_all <- res_all %>% mutate(subject_id = paste0(TF,'_',fold))
res_all$fold <- factor(res_all$fold,levels = c('Low','Middle','High'))

random_comparison <- ggboxplot(res_all,x='set',y='r',color = 'set',outlier.shape = NA) +
  geom_jitter(aes(x=set,y=r,color = set),width = 0.01)+
  # geom_line(aes(x = set, y = r, group = subject_id), 
  #           color = 'gray50', size = 0.5, alpha = 0.1)+
  geom_hline(yintercept = 0,linetype='solid',lwd=0.5,color='black')+
  ylab('Pearson`s correlation')+
  scale_y_continuous(breaks = seq(-0.5,1,0.1),limits = c(NA,1.1))+
  stat_compare_means(comparisons = list(c('train','test'),
                                        c('test','shuffled Y'),
                                        c('train','shuffled Y')),
                     method = 'wilcox.test',
                     paired = TRUE,
                     label.y = c(0.9,0.9,1),
                     label = 'p.signif')+
  facet_wrap(~fold)
random_comparison <- add_theme(random_comparison) + theme_pubr() + 
  theme(axis.text.x = element_blank(),
        axis.title.x = element_blank(),
        axis.ticks.x = element_blank(),
        axis.line.x=element_blank(),
        legend.position = 'top')
print(random_comparison)
ggsave(filename = "../figures/2 hour/dtlembas_2hr_performance_random_comparison_LOOCV.pdf", plot = random_comparison, units = "cm", width = 12, height = 12)
data.table::fwrite(res_all,'../results/LOOCV_2hr_with_random_long.csv')

## Compare LOOCV with linear_baseline---------------------------------
### Load linear baseline results for all time points
linear_baseline <- rbind(
  rbind(
  data.table::fread('../results/High/tf_pearson_correlations.csv') %>%
  select(TF,c('fold'='dose'),c('set'='method'),c('r'='pearson_r')) %>%
  mutate(subject_id=NA) %>% mutate(set='linear'),
  data.table::fread('../results/Middle/tf_pearson_correlations.csv') %>%
    select(TF,c('fold'='dose'),c('set'='method'),c('r'='pearson_r')) %>%
    mutate(subject_id=NA) %>% mutate(set='linear'),
  data.table::fread('../results/Low/tf_pearson_correlations.csv') %>%
    select(TF,c('fold'='dose'),c('set'='method'),c('r'='pearson_r')) %>%
    mutate(subject_id=NA) %>% mutate(set='linear')) %>% mutate(time = '8 hours'),
  rbind(
    data.table::fread('../results/2 hour/High/tf_pearson_correlations.csv') %>%
      select(TF,c('fold'='dose'),c('set'='method'),c('r'='pearson_r')) %>%
      mutate(subject_id=NA) %>% mutate(set='linear'),
    data.table::fread('../results/2 hour/Middle/tf_pearson_correlations.csv') %>%
      select(TF,c('fold'='dose'),c('set'='method'),c('r'='pearson_r')) %>%
      mutate(subject_id=NA) %>% mutate(set='linear'),
    data.table::fread('../results/2 hour/Low/tf_pearson_correlations.csv') %>%
      select(TF,c('fold'='dose'),c('set'='method'),c('r'='pearson_r')) %>%
      mutate(subject_id=NA) %>% mutate(set='linear')) %>% mutate(time = '2 hours'),
  rbind(
    data.table::fread('../results/24 hour/High/tf_pearson_correlations.csv') %>%
      select(TF,c('fold'='dose'),c('set'='method'),c('r'='pearson_r')) %>%
      mutate(subject_id=NA) %>% mutate(set='linear'),
    data.table::fread('../results/24 hour/Middle/tf_pearson_correlations.csv') %>%
      select(TF,c('fold'='dose'),c('set'='method'),c('r'='pearson_r')) %>%
      mutate(subject_id=NA) %>% mutate(set='linear'),
    data.table::fread('../results/24 hour/Low/tf_pearson_correlations.csv') %>%
      select(TF,c('fold'='dose'),c('set'='method'),c('r'='pearson_r')) %>%
      mutate(subject_id=NA) %>% mutate(set='linear')) %>% mutate(time = '24 hours')
  )
linear_baseline$fold <- factor(linear_baseline$fold,levels = c('Low','Middle','High'))
linear_baseline$time <- factor(linear_baseline$time,levels = c('2 hours','8 hours','24 hours'))
### Load DT-LEMBAS results for all time points
res_all_times <- data.frame()
for (time in c('2 hour','8 hour','24 hour')){
  for (i in 1:10){
    if (time=='8 hour'){
      res_model <- rbind(data.table::fread(paste0('../results/High/performance_hepatocytes_dtlembas',i,'.csv')) %>% 
                           select(-V1,-time),
                         data.table::fread(paste0('../results/Middle/performance_hepatocytes_dtlembas',i,'.csv')) %>% 
                           select(-V1,-time),
                         data.table::fread(paste0('../results/Low/performance_hepatocytes_dtlembas',i,'.csv')) %>% 
                           select(-V1,-time))
    }else{
      res_model <- rbind(data.table::fread(paste0('../results/',time,'/High/performance_hepatocytes_dtlembas',i,'.csv')) %>% 
                           select(-V1,-time,),
                         data.table::fread(paste0('../results/',time,'/Middle/performance_hepatocytes_dtlembas',i,'.csv')) %>% 
                           select(-V1,-time),
                         data.table::fread(paste0('../results/',time,'/Low/performance_hepatocytes_dtlembas',i,'.csv')) %>% 
                           select(-V1,-time))
    }
    res_model <- res_model %>% gather('set','r',-fold,-TF,-model_no)
    res_model <- res_model %>% mutate(subject_id = paste0(TF,'_',fold))
    res_model <- res_model %>% select(TF,fold,set,r,subject_id,model_no) %>% mutate(time = paste0(time,'s'))
    res_all_times <- rbind(res_all_times,res_model)
  }
}
joined_df <- left_join(linear_baseline %>% select(-subject_id),
                                   res_all_times %>% select(-subject_id),
                                   by = c('TF','fold','time')) %>% filter(!is.na(r.y)) %>%
  mutate(dr= r.y-r.x)
statistical_comparion <- joined_df %>% 
  group_by(TF,fold,time,set.y) %>%
  rstatix::t_test(dr ~ 1, mu = 0) %>% ungroup() %>%
  select(TF,fold,time,c('set'='set.y'),p)
statistical_comparion <- left_join(statistical_comparion,
                                   joined_df %>% 
                                     select(TF,fold,time,c('set'='set.y'),model_no,dr) %>%
                                     group_by(TF,fold,time,set) %>% summarise(Dr=mean(dr)))

res_all_times$fold <- factor(res_all_times$fold,levels = c('Low','Middle','High'))
res_all_times$time <- factor(res_all_times$time,levels = c('2 hours','8 hours','24 hours'))

# Plot
baseline_comparison <- ggboxplot(rbind(res_all_times %>% filter(model_no==1) %>% select(-model_no),
                                       linear_baseline),
                                 x='set',y='r',color = 'set',outlier.shape = NA) +
  geom_jitter(aes(x=set,y=r,color = set),width = 0.1)+
  geom_hline(yintercept = 0,linetype='solid',lwd=0.5,color='black')+
  ylab('Pearson`s correlation')+
  scale_y_continuous(breaks = seq(-0.5,1,0.1),limits = c(NA,1.1))+
  stat_compare_means(comparisons = list(c('train','test'),
                                        c('test','linear'),
                                        c('train','linear')),
                     method = 'wilcox.test',
                     paired = TRUE,
                     label.y = c(0.9,0.9,1),
                     label = 'p.signif')+
  facet_wrap(vars(time,fold))
baseline_comparison <- add_theme(baseline_comparison) + theme_pubr() + 
  theme(axis.text.x = element_blank(),
        axis.title.x = element_blank(),
        axis.ticks.x = element_blank(),
        axis.line.x=element_blank(),
        legend.position = 'top')
print(baseline_comparison)
ggsave('../figures/boxplot_comparison_linear.png',
       plot = baseline_comparison,
       scale = 1,
       width = 9,
       height = 9,
       units = "in",
       dpi = 600)
ggsave('../figures/boxplot_comparison_linear.pdf',
       plot = baseline_comparison,
       device = cairo_pdf,
       scale = 1,
       width = 9,
       height = 9,
       units = "in",
       dpi = 600)

### make volcano-like plot
statistical_comparion <- statistical_comparion %>% 
  mutate(p_trans=-log10(p))
# statistical_comparion_summary <- statistical_comparion  %>%
#   mutate(flag = ifelse(Dr>0,ifelse(p<=0.01,TRUE,FALSE),FALSE)) %>% group_by(fold,time,set) %>% 
#   mutate(no_tfs = sum(flag)) %>% 
#   ungroup() %>%
#   select(fold,time,set,flag,no_tfs) %>% unique() %>%
#   mutate(percentage_success = no_tfs/104)
# statistical_comparion <- left_join(statistical_comparion,statistical_comparion_summary)
# ggplot(statistical_comparion_summary,aes(x=percentage_success)) +
#   geom_histogram() +
#   geom_vline(xintercept = 0.5,linetype='dashed',lwd=1)

## LABEL WITH THEIR NAMES THE IMPORTANT TFs
important_tfs <- c('P10914','Q00978','P42226','P42229','P51692',
                     'P40763','P52630','P42224','Q14765','P84022',
                   'P01106','P05412','O00716','Q14209','Q01094',
                   'Q16254','P04637')
annot <- read.delim('../../../Artificial-Signaling-Network/DrugsANNSignaling/data/uniprot-reviewed_yes+AND+organism__Homo+sapiens+(Human)+[9606]_.tab')
annot <- annot %>% select(c('label'='Entry'),c('label_symbol'='Gene.names...primary..')) %>% unique()
annot <- annot %>% filter(label %in% important_tfs)
statistical_comparion <- statistical_comparion %>% 
  mutate(label = ifelse(TF %in% important_tfs,TF,NA))
statistical_comparion <- left_join(statistical_comparion,annot)

### Volcano-like plots ###
#Test set
ggscatter(statistical_comparion %>%
            mutate(flag1 = ifelse(Dr>0,TRUE,FALSE))%>%
            mutate(flag2 = ifelse(p<=0.01,TRUE,FALSE)) %>%
            mutate(flag = flag1&flag2) %>% filter(set=='test'),
          x='Dr',y='p_trans',color = 'flag') +
  geom_label_repel(aes(label=label_symbol,color=flag),max.overlaps = 40)+
  ylab(expression(-log10('p-value')))+
  geom_hline(yintercept = -log10(0.01),linetype='dashed',lwd=1)+
  geom_vline(xintercept = 0,linetype='dashed',lwd=1)+
  facet_wrap(vars(time,fold),scales = 'free_y')
ggsave('../figures/volcano_test_comparison_linear.png',
       scale = 1,
       width = 9,
       height = 9,
       units = "in",
       dpi = 600)
#Train set
ggscatter(statistical_comparion %>%
            mutate(flag1 = ifelse(Dr>0,TRUE,FALSE))%>%
            mutate(flag2 = ifelse(p<=0.01,TRUE,FALSE)) %>%
            mutate(flag = flag1&flag2) %>% filter(set=='train'),
          x='Dr',y='p_trans',color = 'flag') +
  geom_label_repel(aes(label=label_symbol,color=flag),max.overlaps = 40)+
  ylab(expression(-log10('p-value')))+
  geom_hline(yintercept = -log10(0.01),linetype='dashed',lwd=1)+
  geom_vline(xintercept = 0,linetype='dashed',lwd=1)+
  facet_wrap(vars(time,fold),scales = 'free_y')
ggsave('../figures/volcano_train_comparison_linear.png',
       scale = 1,
       width = 9,
       height = 9,
       units = "in",
       dpi = 600)

### Confusion-like matrix with percentages of imporant better than random ###
#Important
confusion_important <- statistical_comparion %>%
  mutate(flag1 = ifelse(Dr>0,TRUE,FALSE))%>%
  mutate(flag2 = ifelse(p<=0.01,TRUE,FALSE)) %>%
  mutate(flag = flag1&flag2) %>% 
  filter(TF %in% important_tfs) %>%
  group_by(set,time,fold) %>%
  mutate(better_than_random = 100*sum(flag)/length(important_tfs)) %>%
  ungroup() %>%
  select(set,time,fold,better_than_random) %>%
  unique()
confusion_important$set <- factor(confusion_important$set,levels = c('train','test'))
confusion_important$fold <- factor(confusion_important$fold,levels = c('Low','Middle','High'))
confusion_important$time <- factor(confusion_important$time,levels = c('2 hours','8 hours','24 hours'))
plot_confusion_important <- ggplot(confusion_important, 
                         aes(x = fold, y = time, fill = better_than_random)) +
  geom_tile() +
  scale_fill_viridis(limits = c(0,100)) +
  geom_text(aes(label = paste0('~',round(better_than_random,1),'%')), size = 14) +
  labs(title = '% of TFs predicted better than linear baseline',
       x = "Dose",
       y = "Timepoint") +
  theme_minimal() +
  theme(text = element_text(family = 'Arial',size=24),
        axis.text = element_text(family = 'Arial',size = 28),
        axis.title = element_text(family = 'Arial',size = 28, face = "bold"),
        legend.text = element_text(family = 'Arial',size = 23),
        legend.position = 'none',
        plot.title = element_text(family = 'Arial',size = 24, hjust = 0.5))+
  facet_wrap(~set)
print(plot_confusion_important)
ggsave('../figures/confusion_better_than_linear_important.png',
       plot = plot_confusion_important,
       scale = 1,
       width = 16,
       height = 9,
       units = "in",
       dpi = 600)
ggsave('../figures/confusion_better_than_linear_important.pdf',
       device = cairo_pdf,
       plot = plot_confusion_important,
       scale = 1,
       width = 16,
       height = 9,
       units = "in",
       dpi = 600)

## All tfs
confusion_all <- statistical_comparion %>%
  mutate(flag1 = ifelse(Dr>0,TRUE,FALSE))%>%
  mutate(flag2 = ifelse(p<=0.01,TRUE,FALSE)) %>%
  mutate(flag = flag1&flag2) %>% 
  group_by(set,time,fold) %>%
  mutate(better_than_random = 100*sum(flag)/104) %>%
  ungroup() %>%
  select(set,time,fold,better_than_random) %>%
  unique()
confusion_all$set <- factor(confusion_all$set,levels = c('train','test'))
confusion_all$fold <- factor(confusion_all$fold,levels = c('Low','Middle','High'))
confusion_all$time <- factor(confusion_all$time,levels = c('2 hours','8 hours','24 hours'))
plot_confusion_all <- ggplot(confusion_all, 
                         aes(x = fold, y = time, fill = better_than_random)) +
  geom_tile() +
  scale_fill_viridis(limits = c(0,100)) +
  geom_text(aes(label = paste0('~',round(better_than_random,1),'%')), size = 14) +
  labs(title = '% of TFs predicted better than linear baseline',
       x = "Dose",
       y = "Timepoint") +
  theme_minimal() +
  theme(text = element_text(family = 'Arial',size=24),
        axis.text = element_text(family = 'Arial',size = 28),
        axis.title = element_text(family = 'Arial',size = 28, face = "bold"),
        legend.text = element_text(family = 'Arial',size = 23),
        legend.position = 'none',
        plot.title = element_text(family = 'Arial',size = 24, hjust = 0.5))+
  facet_wrap(~set)
print(plot_confusion_all)
ggsave('../figures/confusion_better_than_linear_all.png',
       plot = plot_confusion_all,
       scale = 1,
       width = 16,
       height = 9,
       units = "in",
       dpi = 600)
ggsave('../figures/confusion_better_than_linear_all.pdf',
       device = cairo_pdf,
       plot = plot_confusion_all,
       scale = 1,
       width = 16,
       height = 9,
       units = "in",
       dpi = 600)
  



## Compare random CV with linear baseline------------
## Load linear baseline results
linear_baseline <- rbind(
    data.table::fread('../results/CV/tf_pearson_correlations_linear_CV.csv') %>%
      select(TF,fold,c('r'='pearson_r'),c('time'='timepoint')) %>%
      mutate(subject_id=NA) %>% mutate(set='linear'),
    data.table::fread('../results/2 hour/CV/tf_pearson_correlations_linear_CV.csv') %>%
      select(TF,fold,c('r'='pearson_r'),c('time'='timepoint')) %>%
      mutate(subject_id=NA) %>% mutate(set='linear'),
    data.table::fread('../results/24 hour/CV/tf_pearson_correlations_linear_CV.csv') %>%
      select(TF,fold,c('r'='pearson_r'),c('time'='timepoint')) %>%
      mutate(subject_id=NA) %>% mutate(set='linear')
) %>% mutate(time = paste0(time,'s')) %>%
  select(subject_id,set,time,fold,TF,r) %>%
  mutate(subject_id = paste0(TF,'_',fold))
linear_baseline$time <- factor(linear_baseline$time,levels = c('2 hours','8 hours','24 hours'))
# Load CV results
res_all <- rbind(
  data.table::fread('../results/CV/hepatocytes_dtlembas1_fold_results.csv') %>%
    select(-V1) %>% mutate(time = '8 hours'),
  data.table::fread('../results/2 hour/CV/hepatocytes_dtlembas1_fold_results.csv') %>%
                     select(-V1) %>% mutate(time = '2 hours'),
  data.table::fread('../results/24 hour/CV/hepatocytes_dtlembas1_fold_results.csv') %>%
    select(-V1) %>% mutate(time = '24 hours')
  )
colnames(res_all)[3:4] <- c('train','test')
res_all <- res_all %>% gather('set','r',-fold,-TF,-time)
res_all <- res_all %>% mutate(subject_id = paste0(TF,'_',fold))
res_all <- res_all %>% select(all_of(colnames(linear_baseline)))
res_all$time <- factor(res_all$time,levels = c('2 hours','8 hours','24 hours'))

### Combine the to plot them as before
joined_df <- left_join(linear_baseline %>% select(-subject_id),
                       res_all %>% select(-subject_id),
                       by = c('TF','fold','time')) %>% filter(!is.na(r.y)) %>%
  mutate(dr= r.y-r.x)
statistical_comparion <- joined_df %>% 
  group_by(TF,time,set.y) %>%
  rstatix::t_test(dr ~ 1, mu = 0) %>% ungroup() %>%
  select(TF,time,c('set'='set.y'),p)
statistical_comparion <- left_join(statistical_comparion,
                                   joined_df %>% 
                                     select(TF,fold,time,c('set'='set.y'),dr) %>%
                                     group_by(TF,time,set) %>% summarise(Dr=mean(dr)))

#### Plot ###
baseline_comparison <- ggboxplot(rbind(res_all,linear_baseline),
                                 x='set',y='r',color = 'set',outlier.shape = NA) +
  geom_jitter(aes(x=set,y=r,color = set),width = 0.1)+
  geom_line(aes(x = set, y = r, group = subject_id), 
            color = 'gray50', size = 0.5, alpha = 0.1)+
  geom_hline(yintercept = 0,linetype='solid',lwd=0.5,color='black')+
  ylab('Pearson`s correlation')+
  scale_y_continuous(breaks = seq(-0.5,1,0.1),limits = c(NA,1.1))+
  stat_compare_means(comparisons = list(c('train','test'),
                                        c('test','linear'),
                                        c('train','linear')),
                     method = 'wilcox.test',
                     paired = TRUE,
                     label.y = c(0.9,0.9,1),
                     label = 'p.signif')+
  facet_wrap(vars(time))
baseline_comparison <- add_theme(baseline_comparison) + theme_pubr() + 
  theme(axis.text.x = element_blank(),
        axis.title.x = element_blank(),
        axis.ticks.x = element_blank(),
        axis.line.x=element_blank(),
        legend.position = 'top')
print(baseline_comparison)
ggsave('../figures/boxplot_comparison_linear_CV_allfolds.png',
       plot = baseline_comparison,
       scale = 1,
       width = 9,
       height = 9,
       units = "in",
       dpi = 600)
ggsave('../figures/boxplot_comparison_linear_CV_allfolds.pdf',
       plot = baseline_comparison,
       device = cairo_pdf,
       scale = 1,
       width = 9,
       height = 9,
       units = "in",
       dpi = 600)

### make volcano-like plot
statistical_comparion <- statistical_comparion %>% 
  mutate(p_trans=-log10(p))
# statistical_comparion_summary <- statistical_comparion  %>%
#   mutate(flag = ifelse(Dr>0,ifelse(p<=0.01,TRUE,FALSE),FALSE)) %>% group_by(fold,time,set) %>% 
#   mutate(no_tfs = sum(flag)) %>% 
#   ungroup() %>%
#   select(fold,time,set,flag,no_tfs) %>% unique() %>%
#   mutate(percentage_success = no_tfs/104)
# statistical_comparion <- left_join(statistical_comparion,statistical_comparion_summary)
# ggplot(statistical_comparion_summary,aes(x=percentage_success)) +
#   geom_histogram() +
#   geom_vline(xintercept = 0.5,linetype='dashed',lwd=1)

## LABEL WITH THEIR NAMES THE IMPORTANT TFs
important_tfs <- c('P10914','Q00978','P42226','P42229','P51692',
                   'P40763','P52630','P42224','Q14765','P84022',
                   'P01106','P05412','O00716','Q14209','Q01094',
                   'Q16254','P04637')
annot <- read.delim('../../../Artificial-Signaling-Network/DrugsANNSignaling/data/uniprot-reviewed_yes+AND+organism__Homo+sapiens+(Human)+[9606]_.tab')
annot <- annot %>% select(c('label'='Entry'),c('label_symbol'='Gene.names...primary..')) %>% unique()
annot <- annot %>% filter(label %in% important_tfs)
statistical_comparion <- statistical_comparion %>% 
  mutate(label = ifelse(TF %in% important_tfs,TF,NA))
statistical_comparion <- left_join(statistical_comparion,annot)

### Volcano-like plots ###
#Test set
ggscatter(statistical_comparion %>%
            mutate(flag1 = ifelse(Dr>0,TRUE,FALSE))%>%
            mutate(flag2 = ifelse(p<=0.01,TRUE,FALSE)) %>%
            mutate(flag = flag1&flag2) %>% filter(set=='test'),
          x='Dr',y='p_trans',color = 'flag') +
  geom_label_repel(aes(label=label_symbol,color=flag),max.overlaps = 40)+
  ylab(expression(-log10('p-value')))+
  geom_hline(yintercept = -log10(0.01),linetype='dashed',lwd=1)+
  geom_vline(xintercept = 0,linetype='dashed',lwd=1)+
  facet_wrap(~time,scales = 'free_y')
ggsave('../figures/volcano_test_comparison_linear_CV.png',
       scale = 1,
       width = 9,
       height = 9,
       units = "in",
       dpi = 600)
#Train set
ggscatter(statistical_comparion %>%
            mutate(flag1 = ifelse(Dr>0,TRUE,FALSE))%>%
            mutate(flag2 = ifelse(p<=0.01,TRUE,FALSE)) %>%
            mutate(flag = flag1&flag2) %>% filter(set=='train'),
          x='Dr',y='p_trans',color = 'flag') +
  geom_label_repel(aes(label=label_symbol,color=flag),max.overlaps = 40)+
  ylab(expression(-log10('p-value')))+
  geom_hline(yintercept = -log10(0.01),linetype='dashed',lwd=1)+
  geom_vline(xintercept = 0,linetype='dashed',lwd=1)+
  facet_wrap(~time,scales = 'free_y')
ggsave('../figures/volcano_train_comparison_linear_CV.png',
       scale = 1,
       width = 9,
       height = 9,
       units = "in",
       dpi = 600)

### Confusion-like matrix with percentages of imporant better than random ###
#Important
confusion_important <- statistical_comparion %>%
  # mutate(flag1 = ifelse(Dr>0,TRUE,FALSE))%>%
  # mutate(flag2 = ifelse(p<=0.01,TRUE,FALSE)) %>%
  # mutate(flag = flag1&flag2) %>% 
  mutate(flag = ifelse(Dr>0,TRUE,FALSE))%>%
  filter(TF %in% important_tfs) %>%
  group_by(set,time) %>%
  mutate(better_than_random = 100*sum(flag)/length(important_tfs)) %>%
  ungroup() %>%
  select(set,time,better_than_random) %>%
  unique()
confusion_important$set <- factor(confusion_important$set,levels = c('train','test'))
confusion_important$time <- factor(confusion_important$time,levels = c('2 hours','8 hours','24 hours'))
plot_confusion_important <- ggplot(confusion_important, 
                                   aes(x = set, y = time, fill = better_than_random)) +
  geom_tile() +
  scale_fill_viridis(limits = c(0,100)) +
  geom_text(aes(label = paste0('~',round(better_than_random,1),'%')), size = 14) +
  labs(title = '% of TFs with model r > r linear baseline',
       x = "Set",
       y = "Timepoint") +
  theme_minimal() +
  theme(text = element_text(family = 'Arial',size=24),
        axis.text = element_text(family = 'Arial',size = 28),
        axis.title = element_text(family = 'Arial',size = 28, face = "bold"),
        legend.text = element_text(family = 'Arial',size = 23),
        legend.position = 'none',
        plot.title = element_text(family = 'Arial',size = 24, hjust = 0.5))
print(plot_confusion_important)
ggsave('../figures/confusion_better_than_linear_important_CV.png',
       plot = plot_confusion_important,
       scale = 1,
       width = 16,
       height = 9,
       units = "in",
       dpi = 600)
ggsave('../figures/confusion_better_than_linear_important_CV.pdf',
       device = cairo_pdf,
       plot = plot_confusion_important,
       scale = 1,
       width = 16,
       height = 9,
       units = "in",
       dpi = 600)

## All tfs
confusion_all <- statistical_comparion %>%
  # mutate(flag1 = ifelse(Dr>0,TRUE,FALSE))%>%
  # mutate(flag2 = ifelse(p<=0.01,TRUE,FALSE)) %>%
  # mutate(flag = flag1&flag2) %>% 
  mutate(flag = ifelse(Dr>0,TRUE,FALSE))%>%
  group_by(set,time) %>%
  mutate(better_than_random = 100*sum(flag)/104) %>%
  ungroup() %>%
  select(set,time,better_than_random) %>%
  unique()
confusion_all$set <- factor(confusion_all$set,levels = c('train','test'))
confusion_all$time <- factor(confusion_all$time,levels = c('2 hours','8 hours','24 hours'))
plot_confusion_all <- ggplot(confusion_all, 
                             aes(x = set, y = time, fill = better_than_random)) +
  geom_tile() +
  scale_fill_viridis(limits = c(0,100)) +
  geom_text(aes(label = paste0('~',round(better_than_random,1),'%')), size = 14) +
  labs(title = '% of TFs with model r > r linear baseline',
       x = "Set",
       y = "Timepoint") +
  theme_minimal() +
  theme(text = element_text(family = 'Arial',size=24),
        axis.text = element_text(family = 'Arial',size = 28),
        axis.title = element_text(family = 'Arial',size = 28, face = "bold"),
        legend.text = element_text(family = 'Arial',size = 23),
        legend.position = 'none',
        plot.title = element_text(family = 'Arial',size = 24, hjust = 0.5))
print(plot_confusion_all)
ggsave('../figures/confusion_better_than_linear_all_CV.png',
       plot = plot_confusion_all,
       scale = 1,
       width = 16,
       height = 9,
       units = "in",
       dpi = 600)
ggsave('../figures/confusion_better_than_linear_all_CV.pdf',
       device = cairo_pdf,
       plot = plot_confusion_all,
       scale = 1,
       width = 16,
       height = 9,
       units = "in",
       dpi = 600)


## Ensemble performance VS single VS linear----------------
res_model <- data.frame()
ensemble_results <- data.frame()
for (time in c('2 hour','8 hour','24 hour')){
  if (time=='8 hour'){
    tmp_ensemble <- rbind(
      data.table::fread(paste0('../results/High/ensemble_performance_High_8 hour.csv')) %>%
        select(-V1),
      data.table::fread(paste0('../results/Middle/ensemble_performance_Middle_8 hour.csv')) %>%
        select(-V1),
      data.table::fread(paste0('../results/Low/ensemble_performance_Low_8 hour.csv')) %>%
        select(-V1))
  }else{
    tmp_ensemble <- rbind(
      data.table::fread(paste0('../results/',time,'/High/ensemble_performance_High_',time,'.csv')) %>%
        select(-V1),
      data.table::fread(paste0('../results/',time,'/Middle/ensemble_performance_Middle_',time,'.csv')) %>%
        select(-V1),
      data.table::fread(paste0('../results/',time,'/Low/ensemble_performance_Low_',time,'.csv')) %>%
        select(-V1))
  }
  ensemble_results <- rbind(ensemble_results,tmp_ensemble)
  for (i in 1:10){
    if (time!='8 hour'){
      tmp <- rbind(
        data.table::fread(paste0('../results/',time,'/High/performance_hepatocytes_dtlembas',i,'.csv')) %>% 
          select(-V1),
        data.table::fread(paste0('../results/',time,'/Middle/performance_hepatocytes_dtlembas',i,'.csv')) %>% 
          select(-V1),
        data.table::fread(paste0('../results/',time,'/Low/performance_hepatocytes_dtlembas',i,'.csv')) %>% 
          select(-V1))
    }else{
      tmp <- rbind(
        data.table::fread(paste0('../results/High/performance_hepatocytes_dtlembas',i,'.csv')) %>% 
          select(-V1),
        data.table::fread(paste0('../results/Middle/performance_hepatocytes_dtlembas',i,'.csv')) %>% 
          select(-V1),
        data.table::fread(paste0('../results/Low/performance_hepatocytes_dtlembas',i,'.csv')) %>% 
          select(-V1))
    }
    res_model <- rbind(res_model,tmp)
  }
}
res_model <- res_model %>% gather('set','r',-TF,-time,-fold,-model_no) %>% mutate(type='Single model')
ensemble_results <- ensemble_results %>% mutate(model_no=-1) %>% mutate(type='Ensemble') %>%
  select(all_of(colnames(res_model))) %>% mutate(set = 'Ensemble test')
all_results <- rbind(ensemble_results,res_model)
all_results$set <- factor(all_results$set,levels = c('train','test','Ensemble test'))
all_results$time <- factor(all_results$time,levels = c('2 hour','8 hour','24 hour'))
all_results$fold <- factor(all_results$fold,levels = c('Low','Middle','High'))
## Visualize
p <- ggviolin(all_results,x='set',y='r',fill = 'set',add='boxplot')+
  geom_hline(yintercept = 0,linetype='solid',lwd=0.5,color='black')+
  facet_wrap(vars(time,fold))+
  stat_compare_means(comparisons = list(c('test','Ensemble test')),
                     label.y = c(1,0.95),
                     method = 'wilcox')
p <- add_theme(p) + theme_pubr() + 
  theme(axis.text.x = element_blank(),
        axis.title.x = element_blank(),
        axis.ticks.x = element_blank(),
        axis.line.x=element_blank(),
        legend.position = 'top')
print(p)

## Load linear results and combine with ensemble
linear_baseline <- rbind(
  rbind(
    data.table::fread('../results/High/tf_pearson_correlations.csv') %>%
      select(TF,c('fold'='dose'),c('set'='method'),c('r'='pearson_r')) %>%
      mutate(subject_id=NA) %>% mutate(set='linear'),
    data.table::fread('../results/Middle/tf_pearson_correlations.csv') %>%
      select(TF,c('fold'='dose'),c('set'='method'),c('r'='pearson_r')) %>%
      mutate(subject_id=NA) %>% mutate(set='linear'),
    data.table::fread('../results/Low/tf_pearson_correlations.csv') %>%
      select(TF,c('fold'='dose'),c('set'='method'),c('r'='pearson_r')) %>%
      mutate(subject_id=NA) %>% mutate(set='linear')) %>% mutate(time = '8 hours'),
  rbind(
    data.table::fread('../results/2 hour/High/tf_pearson_correlations.csv') %>%
      select(TF,c('fold'='dose'),c('set'='method'),c('r'='pearson_r')) %>%
      mutate(subject_id=NA) %>% mutate(set='linear'),
    data.table::fread('../results/2 hour/Middle/tf_pearson_correlations.csv') %>%
      select(TF,c('fold'='dose'),c('set'='method'),c('r'='pearson_r')) %>%
      mutate(subject_id=NA) %>% mutate(set='linear'),
    data.table::fread('../results/2 hour/Low/tf_pearson_correlations.csv') %>%
      select(TF,c('fold'='dose'),c('set'='method'),c('r'='pearson_r')) %>%
      mutate(subject_id=NA) %>% mutate(set='linear')) %>% mutate(time = '2 hours'),
  rbind(
    data.table::fread('../results/24 hour/High/tf_pearson_correlations.csv') %>%
      select(TF,c('fold'='dose'),c('set'='method'),c('r'='pearson_r')) %>%
      mutate(subject_id=NA) %>% mutate(set='linear'),
    data.table::fread('../results/24 hour/Middle/tf_pearson_correlations.csv') %>%
      select(TF,c('fold'='dose'),c('set'='method'),c('r'='pearson_r')) %>%
      mutate(subject_id=NA) %>% mutate(set='linear'),
    data.table::fread('../results/24 hour/Low/tf_pearson_correlations.csv') %>%
      select(TF,c('fold'='dose'),c('set'='method'),c('r'='pearson_r')) %>%
      mutate(subject_id=NA) %>% mutate(set='linear')) %>% mutate(time = '24 hours')
)

linear_vs_ensemble <- rbind(linear_baseline %>% select(-subject_id)  %>% select(TF,fold,time,r,set),
                       ensemble_results %>% select(TF,fold,time,r,set) %>%
                         mutate(time=paste0(time,'s')))
# linear_vs_ensemble <- linear_vs_ensemble %>% mutate(type = set)
linear_vs_ensemble$set <- factor(linear_vs_ensemble$set,levels = c('Ensemble test','linear'))
linear_vs_ensemble$time <- factor(linear_vs_ensemble$time,levels = c('2 hours','8 hours','24 hours'))
linear_vs_ensemble$fold <- factor(linear_vs_ensemble$fold,levels = c('Low','Middle','High'))
# p_lin <- ggpaired(linear_vs_ensemble %>% spread('set','r'),
#                   cond1 = 'Ensemble test',
#                   cond2 = 'linear',
#                   line.color = "gray", line.size = 0.1)+
#   geom_hline(yintercept = 0,linetype='dashed',lwd=1,color='black')+
#   facet_wrap(vars(time,fold))+
#   stat_compare_means(comparisons = list(c('Ensemble test','linear')),
#                      label.y = 0.9,
#                      method = 'wilcox',
#                      paired = TRUE)
# p_lin <- add_theme(p_lin) + theme_pubr() + 
#   theme(axis.title.x = element_blank())
# print(p_lin)

p_lin <- ggboxplot(linear_vs_ensemble,
                  x = 'set',
                  y = 'r',
                  outlier.shape = NA,
                  add = 'jitter',
                  add.params = list(size=1))+
  scale_y_continuous(breaks = seq(-0.5,1,0.2),limits = c(NA,1))+
  geom_hline(yintercept = 0,linetype='dashed',lwd=1,color='black')+
  facet_wrap(vars(time,fold))+
  stat_compare_means(comparisons = list(c('Ensemble test','linear')),
                     label.y = 0.9,
                     method = 'wilcox',
                     paired = TRUE)
p_lin <- add_theme(p_lin) + theme_pubr() + 
  theme(axis.title.x = element_blank())
print(p_lin)
ggsave('../figures/ensemble_vs_linear_LOOCV.png',
       plot = p_lin,
       scale = 1,
       width = 9,
       height = 9,
       units = "in",
       dpi = 600)

# Only important
important_tfs <- c('P10914','Q00978','P42226','P42229','P51692',
                   'P40763','P52630','P42224','Q14765','P84022',
                   'P01106','P05412','O00716','Q14209','Q01094',
                   'Q16254','P04637')
p_lin_important <- ggboxplot(linear_vs_ensemble %>% filter(TF %in% important_tfs),
                   x = 'set',
                   y = 'r',
                   outlier.shape = NA,
                   add = 'jitter',
                   add.params = list(size=1))+
  scale_y_continuous(breaks = seq(-0.5,1,0.2),limits = c(NA,1))+
  geom_hline(yintercept = 0,linetype='dashed',lwd=1,color='black')+
  facet_wrap(vars(time,fold))+
  stat_compare_means(comparisons = list(c('Ensemble test','linear')),
                     label.y = 0.9,
                     method = 'wilcox',
                     paired = TRUE)
p_lin_important <- add_theme(p_lin_important) + theme_pubr() + 
  theme(axis.title.x = element_blank())
print(p_lin_important)
ggsave('../figures/ensemble_vs_linear_important_LOOCV.png',
       plot = p_lin_important,
       scale = 1,
       width = 9,
       height = 9,
       units = "in",
       dpi = 600)
