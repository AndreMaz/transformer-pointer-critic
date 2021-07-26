library('ggplot2')
library('dplyr')
library("reshape2")
library(ggpubr)
library(nortest)

base = './ResourceV3'
filename = 'time_logs_ep.csv'

file = paste(base, filename, sep='/')

## Load data from CSV
df1 <- read.csv(file=file, header = TRUE, sep = ';')

## Average the data
avgData <- df1 %>%
  group_by(Number.of.Nodes, resource_sample_size) %>% 
  summarise(
    avg = mean(Time),
    #count = n(),
    sd = sd(Time, na.rm = FALSE)
  )



inf_time <- ggplot(avgData, aes(x=resource_sample_size, y=Time, col=Number.of.Nodes, group = Number.of.Nodes))+
  geom_line(size=1.5, alpha=0.7)+
  theme(axis.text.x = element_text(angle = 35, hjust = 1))+
  labs(x="Number of Input Rules", y='Single Rule Placement Inference Time (ms)')+
  theme(legend.position="bottom")


ep_time <- ggplot(avgData, aes(x=resource_sample_size, y=Episode.Time, col=Number.of.Nodes, group = Number.of.Nodes))+
  geom_line(size=1.5, alpha=0.7)+
  theme(axis.text.x = element_text(angle = 35, hjust = 1))+
  labs(x="Number of Input Rules", y='Episode Time (ms)')+
  theme(legend.position="bottom")
ep_time

# ggarrange(inf_time, ep_time, ncol = 2, nrow = 1, common.legend = TRUE, legend="bottom")
