library('ggplot2')
library('dplyr')
library("reshape2")
library(ggpubr)
library(nortest)


## Load data from CSV
net_og_data <- read.csv(file='./resource/test_net.csv', header = TRUE, sep = ';')
heuristic_og_data <- read.csv(file='./resource/test_heu.csv', header = TRUE, sep = ';')

# Stack/Merge the stats
og_data <- rbind(net_og_data, heuristic_og_data)

# og_data <- og_data %>%
#  mutate(Node = paste('Node',Node))

# Set up factor variables
og_data$Node <- as.factor(og_data$Node)
og_data$Step <- as.factor(og_data$Step)
og_data$Method <- as.factor(og_data$Method)

branded_colors <- list(
  "blue"   = "#00798c",
  "red"    = "#d1495b",
  "yellow" = "#edae49",
  "green"  = "#66a182",
  "navy"   = "#2e4057", 
  "grey"   = "#8d96a3"
)

# Reshape into tall format
reshaped <- melt(og_data, id.vars = c("Method", "Step","Node", "Percentage_Penalized"), variable.name = 'Type', value.name = 'Value')

# Filter out the EOS node
reshaped <- reshaped %>%
  # filter(Node != "Node 0")
  filter(Node != 0)

# Plot CPU, RAM and MEM utilization
ggplot(data = reshaped)+
  geom_point(aes(x=Step, y=Value, col=Type, group=Method), alpha=0.3)+
  # geom_smooth(aes(x=Step, y=Value, col=Type, group=Type), alpha=0.3, span=0.3)+
  # geom_line(aes(x=Step, y=Value, col=Type, group=Type), size=1.5, alpha=0.7)+
  facet_wrap(Node~Method, labeller = labeller(Node=label_both, Method=label_both) ,ncol = 4)+
  labs(x="Placement Step", y='Load (%)')+
  scale_x_discrete(breaks = seq(10, 600, by = 20))+
  theme(
    # legend.position="bottom",
    # legend.title=element_blank()
  )

# Plot Resource Penalization
ggplot(data = reshaped)+
  geom_point(aes(x=Step, y=Percentage_Penalized, col=Method, group=Method), alpha=0.3)+
  # geom_smooth(aes(x=Step, y=Value, col=Type, group=Type), alpha=0.3, span=0.3)+
  # geom_line(aes(x=Step, y=Value, col=Type, group=Type), size=1.5, alpha=0.7)+
  facet_wrap(~Node, labeller = labeller(Node=label_both) ,ncol = 4)+
  labs(x="Placement Step", y='Penalized Resources (%)')+
  scale_x_discrete(breaks = seq(10, 600, by = 20))+
  theme(
    # legend.position="bottom",
    # legend.title=element_blank()
  )

