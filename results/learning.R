library('ggplot2')
library('dplyr')
library("reshape2")
library(ggpubr)
library(nortest)


## Load data from CSV
base = './ResourceV3'
test_location = 'training'
filename = 'logs.csv'
date = '2021-03-09T16:44:23'

file = paste(base, date, test_location, filename, sep='/')

learning_data <- read.csv(file=file, header = TRUE, sep = ';')

# Reshape learning stats into tall format
learning_stats <- melt(learning_data, id.vars = c(
  "Step",
  "Avg.Reward",
  "Min.Reward",
  "Max.Reward"
), variable.name = 'Type', value.name = 'Value')

## Filter out by Types of Sets
learning_stats <- learning_stats %>%
  filter(
    Type == 'Value.Loss' #|
    # Type == 'Total.Bin.Loss' #|
    # Type == 'Total.Resource.Loss' |
    # Type == 'Bin.Policy.Loss'  #|
    # Type == 'Bin.Entropy' #|
    # Type == 'Resource.Policy.Loss' |
    # Type == 'Resource.Entropy'
  )
# filter( Step > 5000)

# Plot Losses and entropy
ggplot(data = learning_stats, aes(x=Step, y=Value, col=Type, group = Type))+
  # geom_point(alpha=0.3)+
  #geom_line(alpha=1.0)+
  geom_smooth(alpha=0.3, span=0.3, se = FALSE)+
  # geom_line(aes(x=Step, y=Value, col=Type, group=Type), size=1.5, alpha=0.7)+
  labs(x="Episode", y='Value')+
  # scale_x_discrete()+
  theme(
    # legend.position="bottom",
    # legend.title=element_blank()
  )

ggsave(paste(base, date, test_location, "learning.pdf", sep='/'))

# Reshape reward data into tall format
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
  # geom_point(alpha=0.2)+
  geom_smooth(alpha=1.0, span=0.3, se = FALSE)+
  # geom_line(aes(x=Step, y=Value, col=Type, group=Type), size=1.5, alpha=0.7)+
  labs(x="Episode", y='Reward')+
  # scale_x_discrete()+
  theme(
    # legend.position="bottom",
    # legend.title=element_blank()
  )
ggsave(paste(base, date, test_location, "rewards.pdf", sep='/'))
