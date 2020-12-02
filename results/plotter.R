library('ggplot2')
library('dplyr')
library("reshape2")
library(ggpubr)
library(nortest)


## Load data from CSV
net_og_data <- read.csv(file='./resource/test_net.csv', header = TRUE, sep = ';')
heuristic_og_data <- read.csv(file='./resource/test_heu.csv', header = TRUE, sep = ';')

# Stack/Merge the stats
og_data <- merge(net_og_data, heuristic_og_data)

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

reshaped <- reshaped %>%
  filter(Node != 0)

ggplot(data = reshaped)+
  geom_point(aes(x=Step, y=Value, col=Type), alpha=0.3)+
  # geom_line(aes(x=Step, y=Value, col=Type, group=Type), linetype='dashed')+
  # geom_line(aes(x=Step, y=Value, col=Type, group=Type), linetype='dotted')+
  facet_wrap(~Node)+
  labs(x="Placement Step", y='Load (%)')+
  scale_x_discrete(breaks = seq(10, 600, by = 20))+
  theme(
    # legend.position="bottom",
    # legend.title=element_blank()
  )


