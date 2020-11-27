library('ggplot2')
library('dplyr')
library("reshape2")
library(ggpubr)
library(nortest)

## Load data from CSV
og_data <- read.csv(file='./resource/test.csv', header = TRUE, sep = ';')

# Set up factor variables
og_data$Node <- as.factor(og_data$Node)
og_data$Step <- as.factor(og_data$Step)

branded_colors <- list(
  "blue"   = "#00798c",
  "red"    = "#d1495b",
  "yellow" = "#edae49",
  "green"  = "#66a182",
  "navy"   = "#2e4057", 
  "grey"   = "#8d96a3"
)

# Reshape into tall format
reshaped <- melt(og_data, id.vars = c("Step","Node"), variable.name = 'Type', value.name = 'Value')

ggplot(data = reshaped)+
  geom_point(aes(x=Step, y=Value, col=Type, group=Node), alpha=0.3)+
  facet_wrap(~Node)+
  labs(x="Placement Step", y='Load (%)')+
  scale_x_discrete(breaks = seq(10, 200, by = 20))+
  theme(
    # legend.position="bottom",
    # legend.title=element_blank()
  )
  


