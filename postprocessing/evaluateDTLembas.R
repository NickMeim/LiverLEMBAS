library(tidyverse)
library(ggplot2)
library(ggpubr)
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

### Load CV ensemble predictions----------------
## to do

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

## Compare with linear_baseline---------------------------------
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
  if (time=='8 hour'){
    res_model <- rbind(data.table::fread('../results/High/performance_hepatocytes_dtlembas1.csv') %>% 
                         select(-V1,-time,-model_no),
                       data.table::fread('../results/Middle/performance_hepatocytes_dtlembas1.csv') %>% 
                         select(-V1,-time,-model_no),
                       data.table::fread('../results/Low/performance_hepatocytes_dtlembas1.csv') %>% 
                         select(-V1,-time,-model_no))
  }else{
    res_model <- rbind(data.table::fread(paste0('../results/',time,'/High/performance_hepatocytes_dtlembas1.csv')) %>% 
                         select(-V1,-time,-model_no),
                       data.table::fread(paste0('../results/',time,'/Middle/performance_hepatocytes_dtlembas1.csv')) %>% 
                         select(-V1,-time,-model_no),
                       data.table::fread(paste0('../results/',time,'/Low/performance_hepatocytes_dtlembas1.csv')) %>% 
                         select(-V1,-time,-model_no))
  }
  res_model <- res_model %>% gather('set','r',-fold,-TF)
  res_model <- res_model %>% mutate(subject_id = paste0(TF,'_',fold))
  res_model <- res_model %>% select(TF,fold,set,r,subject_id) %>% mutate(time = paste0(time,'s'))
  res_all_times <- rbind(res_all_times,res_model)
}
res_all_times$fold <- factor(res_all_times$fold,levels = c('Low','Middle','High'))
res_all_times$time <- factor(res_all_times$time,levels = c('2 hours','8 hours','24 hours'))

# Plot
baseline_comparison <- ggboxplot(rbind(res_all_times,linear_baseline),
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
