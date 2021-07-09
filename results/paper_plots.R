library('ggplot2')
library('dplyr')
library("reshape2")
library(ggpubr)
library(nortest)


base = './ResourceV3'
test_location = 'node/tests'
filename = 'test.csv'
date = 'paper_results'

file = paste(base, date, test_location, filename, sep='/')

## Load data from CSV
df1 <- read.csv(file=file, header = TRUE, sep = ';')

df1 <- df1 %>%
  filter(node_sample_size %% 10 == 0)

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

facet_labels <- c('Latency 150', 'Latency 200', 'Latency 250', 'Latency 300', 'Latency 350')

legendLabels = c(
  "Agent",
  "Dominant Resource ASC Node ASC",
  "Dominant Resource ASC Node DESC",
  "Dominant Resource DESC Node ASC",
  "Dominant Resource DESC Node DESC",
  "Random",
  "CPLEX"
)


labeller_fn <- function(value) {
  paste("Node Sample Size", value)
}

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

rejected_plot <- ggplot(stacked_rejected_data, aes(x=resource_sample_size, y=Value, col=Type, group = Type))+
  geom_point(alpha=0.2)+
  geom_line(size=1.5, alpha=0.7)+
  facet_wrap(c("node_sample_size"), labeller = labeller(node_sample_size = labeller_fn), scales = "free_y", ncol=1)+
  labs(x="Number of Input Requests", y='Number of Rejected Requests')+
  #scale_x_discrete(labels = TRUE)+
  theme(axis.text.x = element_text(angle = 35, hjust = 1))+
  scale_color_hue(labels = legendLabels)+ # Sets the labels
  theme(legend.position="bottom")

rejected_plot

##############################################
############# PLOT REJECTION STATS ###########
##############################################

# Data frame with dominant_data stats
dominant_data <- avgData  %>% 
  select(node_sample_size, node_min_value, node_max_value, resource_sample_size, ends_with("dominant"))

# Reshape learning stats into tall format
stacked_dominant_data <- melt(dominant_data, id.vars = c(
  "node_sample_size",
  "node_min_value",
  "node_max_value",
  "resource_sample_size"
), variable.name = 'Type', value.name = 'Value')


dominant_plot <- ggplot(data = stacked_dominant_data, aes(x=resource_sample_size, y=Value, col=Type, group = Type))+
  geom_point(alpha=0.2)+
  geom_line(size=1.5, alpha=0.7)+
  facet_wrap(c("node_sample_size"), labeller = labeller(node_sample_size = labeller_fn), scales = "free_y", ncol=1)+
  labs(x="Number of Input Requests", y='Dominant Resource Value')+
  # scale_x_discrete(labels = (xLabels))+
  theme(axis.text.x = element_text(angle = 35, hjust = 1))+
  scale_color_hue(labels = legendLabels)+ # Sets the labels
  theme(legend.position="bottom")



ggarrange(rejected_plot, dominant_plot, ncol = 2, nrow = 1, common.legend = TRUE, legend="bottom")
ggsave(paste(base, date, test_location, "test.pdf", sep='/'), height = 15, width = 15, limitsize = FALSE)