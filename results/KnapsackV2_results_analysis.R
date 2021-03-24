library('ggplot2')
library('dplyr')
library("reshape2")
library(ggpubr)
library(nortest)


base = './KnapsackV2'
test_location = 'tests'
filename = 'test.csv'
date = '2021-03-24T12:39:13'

file = paste(base, date, test_location, filename, sep='/')

## Load data from CSV
df1 <- read.csv(file=file, header = TRUE, sep = ';')

##############################################
################ COMMON HELPERS ##############
##############################################

## Average the data
avgData <- df1 %>%
  group_by(bin_sample_size, bin_min_value, bin_max_value, item_sample_size) %>% 
  summarise_each(mean)
### Drop unecessary cols
avgData <- subset(avgData, select = -c(X, test_instance))

write.csv(avgData, paste(base, date, test_location, "avg.csv", sep='/'))

avgData$bin_min_value <- as.factor(avgData$bin_min_value)

xLabels <- c("0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80-90", "90-100")

##############################################
############# PLOT REWARDS STATS #############
##############################################

# Data frame with rejection stats
reward_data <- avgData  %>% 
  select(bin_sample_size, bin_min_value, bin_max_value, item_sample_size, ends_with("reward"))

# Reshape learning stats into tall format
stacked_reward_data <- melt(reward_data, id.vars = c(
  "bin_sample_size",
  "bin_min_value",
  "bin_max_value",
  "item_sample_size"
), variable.name = 'Type', value.name = 'Value')

#stacked_rejected_data <- stacked_rejected_data %>%
#  filter(
#    node_sample_size < 10 |
#      resource_sample_size < 10 
#    )


ggplot(data = stacked_reward_data, aes(x=bin_min_value, y=Value, col=Type, group = Type))+
  geom_point(alpha=0.7)+
  # geom_line(size=1.5, alpha=0.3)+
  facet_wrap(c("bin_sample_size", "item_sample_size"), labeller = "label_both", scales = "free")+
  labs(x="Bin Available Capacity", y='Total Reward')+
  scale_x_discrete(labels = (xLabels))+
  theme(axis.text.x = element_text(angle = 35, hjust = 1))+
  theme(legend.position="bottom")

ggsave(paste(base, date, test_location, "reward.pdf", sep='/'), height = 50, width = 30, limitsize = FALSE)

####################################################
############# PLOT NUMBER OF EMPTY BINS ############
####################################################

# Data frame with empty nodes stats
empty_nodes_data <- avgData  %>% 
  select(bin_sample_size, bin_min_value, bin_max_value, item_sample_size, ends_with("empty.nodes"))

# Reshape learning stats into tall format
stacked_empty_nodes_data <- melt(empty_nodes_data, id.vars = c(
  "bin_sample_size",
  "bin_min_value",
  "bin_max_value",
  "item_sample_size"
), variable.name = 'Type', value.name = 'Value')


ggplot(data = stacked_empty_nodes_data, aes(x=bin_min_value, y=Value, col=Type, group = Type))+
  geom_point(alpha=0.7)+
  # geom_line(size=1.5, alpha=0.3)+
  facet_wrap(c("bin_sample_size", "item_sample_size"), labeller = "label_both", scales = "free")+
  labs(x="Bin Available Capacity", y='Number of Empty Bins')+
  scale_x_discrete(labels = (xLabels))+
  theme(axis.text.x = element_text(angle = 35, hjust = 1))+
  theme(legend.position="bottom")

ggsave(paste(base, date, test_location, "empty_bins.pdf", sep='/'), height = 50, width = 30, limitsize = FALSE)


########################################################
############# PLOT NUMBER OF REJECTED ITEMS ############
########################################################

# Data frame with items rejection stats
rejected_items_data <- avgData  %>% 
  select(bin_sample_size, bin_min_value, bin_max_value, item_sample_size, ends_with("rejected.items"))

# Reshape learning stats into tall format
stacked_rejected_items_data <- melt(rejected_items_data, id.vars = c(
  "bin_sample_size",
  "bin_min_value",
  "bin_max_value",
  "item_sample_size"
), variable.name = 'Type', value.name = 'Value')


ggplot(data = stacked_rejected_items_data, aes(x=bin_min_value, y=Value, col=Type, group = Type))+
  geom_point(alpha=0.7)+
  # geom_line(size=1.5, alpha=0.3)+
  facet_wrap(c("bin_sample_size", "item_sample_size"), labeller = "label_both", scales = "free")+
  labs(x="Bin Available Capacity", y='Number of Rejected Items')+
  scale_x_discrete(labels = (xLabels))+
  theme(axis.text.x = element_text(angle = 35, hjust = 1))+
  theme(legend.position="bottom")

ggsave(paste(base, date, test_location, "rejected_items.pdf", sep='/'), height = 50, width = 30, limitsize = FALSE)

##############################################
############# PLOT REJECTED VALUE ############
##############################################

# Data frame with value rejection stats
rejected_value_data <- avgData  %>% 
  select(bin_sample_size, bin_min_value, bin_max_value, item_sample_size, ends_with("rejected.value"))

# Reshape learning stats into tall format
stacked_rejected_value_data <- melt(rejected_value_data, id.vars = c(
  "bin_sample_size",
  "bin_min_value",
  "bin_max_value",
  "item_sample_size"
), variable.name = 'Type', value.name = 'Value')


ggplot(data = stacked_rejected_value_data, aes(x=bin_min_value, y=Value, col=Type, group = Type))+
  geom_point(alpha=0.7)+
  # geom_line(size=1.5, alpha=0.3)+
  facet_wrap(c("bin_sample_size", "item_sample_size"), labeller = "label_both", scales = "free")+
  labs(x="Bin Available Capacity", y='Number of Rejected Items')+
  scale_x_discrete(labels = (xLabels))+
  theme(axis.text.x = element_text(angle = 35, hjust = 1))+
  theme(legend.position="bottom")

ggsave(paste(base, date, test_location, "rejected_value.pdf", sep='/'), height = 50, width = 30, limitsize = FALSE)


