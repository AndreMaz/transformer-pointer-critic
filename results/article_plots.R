library('ggplot2')
library('dplyr')
library("reshape2")
library(ggpubr)
library(nortest)

legendLabels = c(
  "Agent",
  "DR-DC", # "DR-Fair", #"Dominant Resource ASC Node ASC",
  "DR-AC", #"DR-Node", #Dominant Resource ASC Node DESC",
  "AR-DC", #"AR-Fair", #"Dominant Resource DESC Node ASC",
  "AC-AC", #"AR-Node", #"Dominant Resource DESC Node DESC",
  "Random",
  "CPLEX"
)

labeller_fn <- function(value) {
  paste("Node Sample Size:", value)
}

base = './ResourceV3'
filename = 't3.csv'

file = paste(base, filename, sep='/')

## Load data from CSV
df1 <- read.csv(file=file, header = TRUE, sep = ';')

## Average the data
avgData <- df1 %>%
  group_by(name, resource_sample_size, Number.of.Nodes) %>% 
  summarise_each(mean)
  #summarise(
  #  avg = mean(Time),
    #count = n(),
  #  sd = sd(Time, na.rm = FALSE)
  #)

write.csv(avgData, "./ResourceV3/avg_time.csv")

inf_time <- ggplot(avgData, aes(x=resource_sample_size, y=Time, col=Number.of.Nodes, group = Number.of.Nodes))+
  geom_line(size=1.5, alpha=0.7)+
  theme(axis.text.x = element_text(angle = 35, hjust = 1))+
  labs(x="Number of Input Rules", y='Single Rule Placement Inference Time (ms)')+
  theme(legend.position="bottom")


ep_time <- ggplot(avgData, aes(x=resource_sample_size, y=Episode.Time, col=name, group = name))+
  geom_line(size=1.5, alpha=0.7)+
  theme(axis.text.x = element_text(angle = 35, hjust = 1))+
  facet_wrap(c("Number.of.Nodes"), labeller = labeller(Number.of.Nodes = labeller_fn), scales = "free_y")+
  labs(x="Number of Input Rules", y='Episode Time (ms)')+
  scale_color_hue(labels = legendLabels)+
  theme(legend.position="bottom")+
  ggtitle("Inference Time")
  #theme_update(plot.title = element_text(hjust = 0.5))
ep_time

ggsave(paste(base, "time.pdf", sep='/'), height = 7.5, width = 12, limitsize = FALSE)


# ggarrange(inf_time, ep_time, ncol = 2, nrow = 1, common.legend = TRUE, legend="bottom")
