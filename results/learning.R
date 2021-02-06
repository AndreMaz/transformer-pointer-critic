library('ggplot2')
library('dplyr')
library("reshape2")
library(ggpubr)
library(nortest)


## Load data from CSV
learning_data <- read.csv(file='../media/plots/ResourceV3/transformer/training.csv', header = TRUE, sep = ';')


# Reshape into tall format
reward_stats <- melt(learning_data, id.vars = c(
  "Step",
  "Value.Loss",
  "Resource.Entropy",
  "Total.Resource.Loss",
  "Resource.Policy.Loss",
  "Bin.Entropy",
  "Total.Bin.Loss",
  "Bin.Policy.Loss"), variable.name = 'Type', value.name = 'Value')

# Plot rewards
ggplot(data = reward_stats, aes(x=Step, y=Value, col=Type, group = Type))+
  geom_point(alpha=0.2)+
  geom_smooth(alpha=1.0, span=0.3, se = FALSE)+
  # geom_line(aes(x=Step, y=Value, col=Type, group=Type), size=1.5, alpha=0.7)+
  labs(x="Episode", y='Reward')+
  # scale_x_discrete()+
  theme(
    # legend.position="bottom",
    # legend.title=element_blank()
  )


learning_stats <- melt(learning_data, id.vars = c(
  "Step",
  "Avg.Reward",
  "Min.Reward",
  "Max.Reward"
  ), variable.name = 'Type', value.name = 'Value')

## Filter out by Types of Sets
learning_stats <- learning_stats %>%
 filter( Type != 'Value.Loss' & Type != 'Total.Bin.Loss' & Type != 'Total.Resource.Loss')
# filter( Step > 5000)

# Plot Losses and entropy
ggplot(data = learning_stats, aes(x=Step, y=Value, col=Type, group = Type))+
  geom_point(alpha=0.3)+
  #geom_line(alpha=1.0)+
  # geom_smooth(alpha=0.3, span=0.3, se = FALSE)+
  # geom_line(aes(x=Step, y=Value, col=Type, group=Type), size=1.5, alpha=0.7)+
  labs(x="Episode", y='Value')+
  # scale_x_discrete()+
  theme(
    # legend.position="bottom",
    # legend.title=element_blank()
  )


