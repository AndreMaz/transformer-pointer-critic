library('ggplot2')
library('dplyr')
library("reshape2")
library(ggpubr)
library(nortest)


base = './ResourceV3'
test_location = 'tests'
filename = 'test.csv'
date = '2021-04-21T13:06:53'

file = paste(base, date, test_location, filename, sep='/')

## Load data from CSV
df1 <- read.csv(file=file, header = TRUE, sep = ';')

##############################################
################ COMMON HELPERS ##############
##############################################

## Average the data
avgData <- df1 %>%
  group_by(node_sample_size, node_min_value, node_max_value, resource_sample_size) %>% 
  summarise_each(mean)
### Drop unecessary cols
avgData <- subset(avgData, select = -c(X, test_instance))

write.csv(avgData, paste(base, date, test_location, "avg.csv", sep='/'))

avgData$node_min_value <- as.factor(avgData$node_min_value)
avgData$resource_sample_size <- as.factor(avgData$resource_sample_size)

xLabels <- c("0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80-90", "90-100")

##############################################
############# PLOT REJECTION STATS ###########
##############################################

# Data frame with rejection stats
rejected_data <- avgData  %>% 
  select(node_sample_size, node_min_value, node_max_value, resource_sample_size, ends_with("rejected"))

# Reshape learning stats into tall format
stacked_rejected_data <- melt(rejected_data, id.vars = c(
  "node_sample_size",
  "node_min_value",
  "node_max_value",
  "resource_sample_size"
), variable.name = 'Type', value.name = 'Value')

#stacked_rejected_data <- stacked_rejected_data %>%
#  filter(
#    node_sample_size < 10 |
#      resource_sample_size < 10 
#    )


ggplot(data = stacked_rejected_data, aes(x=resource_sample_size, y=Value, col=Type, group = Type))+
  geom_point(alpha=0.7)+
  # geom_line(size=1.5, alpha=0.3)+
  facet_wrap(c("node_sample_size"), labeller = "label_both", scales = "free_y")+
  labs(x="Number of Input Requests", y='Number of Rejected Requests')+
  #scale_x_discrete(labels = TRUE)+
  theme(axis.text.x = element_text(angle = 35, hjust = 1))+
  theme(legend.position="bottom")

#ggsave(paste(base, date, test_location, "rejected.pdf", sep='/'), height = 50, width = 30, limitsize = FALSE)
ggsave(paste(base, date, test_location, "rejected.pdf", sep='/'), height = 10, width = 15, limitsize = FALSE)


##############################################
############# PLOT DOMINANT STATS ############
##############################################

# Data frame with rejection stats
dominant_data <- avgData  %>% 
  select(node_sample_size, node_min_value, node_max_value, resource_sample_size, ends_with("dominant"))

# Reshape learning stats into tall format
stacked_dominant_data <- melt(dominant_data, id.vars = c(
  "node_sample_size",
  "node_min_value",
  "node_max_value",
  "resource_sample_size"
), variable.name = 'Type', value.name = 'Value')


ggplot(data = stacked_dominant_data, aes(x=resource_sample_size, y=Value, col=Type, group = Type))+
  geom_point(alpha=0.2)+
  geom_line(size=1.5, alpha=0.7)+
  facet_wrap(c("node_sample_size"), labeller = "label_both", scales = "free_y")+
  labs(x="Number of Input Requests", y='Dominant Resource Value')+
  # scale_x_discrete(labels = (xLabels))+
  theme(axis.text.x = element_text(angle = 35, hjust = 1))+
  theme(legend.position="bottom")

# ggsave(paste(base, date, test_location, "dominant.pdf", sep='/'), height = 50, width = 30, limitsize = FALSE)
ggsave(paste(base, date, test_location, "dominant.pdf", sep='/'), height = 10, width = 15, limitsize = FALSE)

##############################################
############# PLOT EMPTY NODES ############
##############################################

# Data frame with rejection stats
empty_nodes_data <- avgData  %>% 
  select(node_sample_size, node_min_value, node_max_value, resource_sample_size, ends_with("empty.nodes"))

# Reshape learning stats into tall format
stacked_empty_nodes_data <- melt(empty_nodes_data, id.vars = c(
  "node_sample_size",
  "node_min_value",
  "node_max_value",
  "resource_sample_size"
), variable.name = 'Type', value.name = 'Value')


ggplot(data = stacked_empty_nodes_data, aes(x=resource_sample_size, y=Value, col=Type, group = Type))+
  geom_point(alpha=0.7)+
  # geom_line(size=1.5, alpha=0.7)+
  facet_wrap(c("node_sample_size"), labeller = "label_both", scales = "free_y")+
  labs(x="Number of Input Requests", y='Number of Empty Nodes')+
  #scale_x_discrete(labels = (xLabels))+
  theme(axis.text.x = element_text(angle = 35, hjust = 1))+
  theme(legend.position="bottom")


#ggsave(paste(base, date, test_location, "empty_nodes.pdf", sep='/'), height = 50, width = 30, limitsize = FALSE)
ggsave(paste(base, date, test_location, "empty_nodes.pdf", sep='/'), height = 10, width = 15, limitsize = FALSE)

