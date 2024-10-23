library(tidyverse)
library(ggplot2)
library(ggpubr)

### First compare one model in 10-fold validation with random model-------
res_model <- data.table::fread('../results/CV/hepatocytes_dtlembas8_fold_results.csv') %>% select(-V1)
colnames(res_model)[3:4] <- c('train','test')
shuffled_model <- data.table::fread('../results/CV/shuffledX_hepatocytes_dtlembas8_fold_results.csv')%>%
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
ggsave(filename = "../figures/dtlembas_performance_random_comparison.pdf", plot = random_comparison, units = "cm", width = 12, height = 12)
data.table::fwrite(res_all,'../results/CV_with_random_long.csv')

### Load CV ensemble predictions----------------
## to do

### First compare one model in LOOCV validation with random model-------
res_model <- rbind(data.table::fread('../results/High/performance_hepatocytes_dtlembas1.csv') %>% 
  select(-V1,-time,-model_no),
  data.table::fread('../results/Middle/performance_hepatocytes_dtlembas1.csv') %>% 
    select(-V1,-time,-model_no),
  data.table::fread('../results/Low/performance_hepatocytes_dtlembas1.csv') %>% 
    select(-V1,-time,-model_no))
shuffled_model <- rbind(data.table::fread('../results/High/performance_hepatocytes_random_dtlembas1.csv') %>% 
  select(TF,c('shuffled Y'='test'),fold),
  data.table::fread('../results/Middle/performance_hepatocytes_random_dtlembas1.csv') %>% 
    select(TF,c('shuffled Y'='test'),fold),
  data.table::fread('../results/Low/performance_hepatocytes_random_dtlembas1.csv') %>% 
    select(TF,c('shuffled Y'='test'),fold))

res_all <- left_join(res_model,shuffled_model)
res_all <- res_all %>% gather('set','r',-fold,-TF)
res_all <- res_all %>% mutate(subject_id = paste0(TF,'_',fold))

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
ggsave(filename = "../figures/dtlembas_performance_random_comparison_LOOCV.pdf", plot = random_comparison, units = "cm", width = 12, height = 12)
data.table::fwrite(res_all,'../results/LOOCV_with_random_long.csv')

