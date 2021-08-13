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
  "AR-AC", #"AR-Node", #"Dominant Resource DESC Node DESC",
  "Random",
  "CPLEX"
)


labeller_fn <- function(value) {
  paste("Node Sample Size:", value)
}

####################################################################################################################
################################################### REDUCED COSTS ##################################################
####################################################################################################################
base = './ResourceV3'
date = 'paper_results'
test_location = 'node/tests'
filename = 'test.csv'

file = paste(base, date, test_location, filename, sep='/')

## Load data from CSV
df1 <- read.csv(file=file, header = TRUE, sep = ';')
## Reduce the plot number for the publication
df1 <- df1 %>%
  filter(node_sample_size %% 10 == 0)

boxes = paste(base, date, test_location, "boxes.csv", sep='/')
boxesData <- read.csv(file=boxes, header = TRUE, sep = ';')

##############################################
################ COMMON HELPERS ##############
##############################################

## Average the data
avgData <- df1 %>%
  group_by(node_sample_size, node_min_value, node_max_value, resource_sample_size) %>% 
  summarise_each(mean)
### Drop unnecessary cols
avgData <- subset(avgData, select = -c(X, test_instance))

write.csv(avgData, paste(base, date, test_location, "avg.csv", sep='/'))

avgData$node_min_value <- as.factor(avgData$node_min_value)
avgData$resource_sample_size <- as.factor(avgData$resource_sample_size)

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

rejected_plot <- ggplot(stacked_rejected_data, aes(x=resource_sample_size, y=(Value/as.numeric(as.character(resource_sample_size)))*100, col=Type, group = Type))+
  geom_point(alpha=0.2)+
  geom_line(size=1.5, alpha=0.7)+
  geom_rect(fill="springgreen3", alpha=0.15, inherit.aes = FALSE, data = boxesData,
            aes(
              xmin = 1,
              xmax = xmax,
              ymin = 0,
              ymax = ymax
            )) +
  facet_wrap(c("node_sample_size"), labeller = labeller(node_sample_size = labeller_fn), scales = "free_y", ncol=1)+
  labs(x="Number of Input Rules", y='Rejection Rate (%)')+
  #scale_x_discrete(labels = TRUE)+
  theme(axis.text.x = element_text(angle = 35, hjust = 1))+
  scale_color_hue(labels = legendLabels)+ # Sets the labels
  theme(legend.position="bottom")

rejected_plot

##############################################
############# PLOT REJECTION STATS ###########
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


empty_plot <- ggplot(data = stacked_empty_nodes_data, aes(x=resource_sample_size, y=Value, col=Type, group = Type))+
  geom_point(alpha=0.2)+
  geom_line(size=1.5, alpha=0.7)+
  geom_rect(fill="springgreen3", alpha=0.15, inherit.aes = FALSE, data = boxesData,
            aes(
              xmin = 1,
              xmax = xmax,
              ymin = 0,
              ymax = ymax
            )) +
  facet_wrap(c("node_sample_size"), labeller = labeller(node_sample_size = labeller_fn), scales = "free_y", ncol=1)+
  labs(x="Number of Input Rules", y='Average Number of Empty Nodes')+
  #scale_x_discrete(labels = TRUE)+
  theme(axis.text.x = element_text(angle = 35, hjust = 1))+
  scale_color_hue(labels = legendLabels)+ # Sets the labels
  theme(legend.position="bottom")

empty_plot


node_plts <- ggarrange(rejected_plot, empty_plot, ncol = 2, nrow = 1, common.legend = TRUE, legend="bottom")
annotate_figure(node_plts,
                top=text_grob("Cost-Aware Greedy Optimization Perfomance")
)
ggsave(paste(base, date, test_location, "cost.pdf", sep='/'), height = 15, width = 12, limitsize = FALSE)


#
# rm(list = ls(all.names = TRUE)) #will clear all objects includes hidden objects.
# gc() #free up memrory and report the memory usage.



####################################################################################################################
################################################### CRITICAL RESOURCE ##################################################
####################################################################################################################
base = './ResourceV3'
date = 'paper_results'
test_location = 'critical/tests'
filename = 'test.csv'

file = paste(base, date, test_location, filename, sep='/')

## Load data from CSV
df1 <- read.csv(file=file, header = TRUE, sep = ';')
## Reduce the plot number for the publication
df1 <- df1 %>%
  filter(node_sample_size %% 10 == 0)

boxes = paste(base, date, test_location, "boxes.csv", sep='/')
boxesData <- read.csv(file=boxes, header = TRUE, sep = ';')

##############################################
################ COMMON HELPERS ##############
##############################################

## Average the data
avgData <- df1 %>%
  group_by(node_sample_size, node_min_value, node_max_value, resource_sample_size) %>% 
  summarise_each(mean)
### Drop unnecessary cols
avgData <- subset(avgData, select = -c(X, test_instance))

write.csv(avgData, paste(base, date, test_location, "avg.csv", sep='/'))

avgData$node_min_value <- as.factor(avgData$node_min_value)
avgData$resource_sample_size <- as.factor(avgData$resource_sample_size)

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

rejected_plot <- ggplot(stacked_rejected_data, aes(x=resource_sample_size, y=(Value/as.numeric(as.character(resource_sample_size)))*100, col=Type, group = Type))+
  geom_point(alpha=0.2)+
  geom_line(size=1.5, alpha=0.7)+
  geom_rect(fill="springgreen3", alpha=0.15, inherit.aes = FALSE, data = boxesData,
            aes(
              xmin = 1,
              xmax = xmax,
              ymin = 0,
              ymax = ymax
            )) +
  facet_wrap(c("node_sample_size"), labeller = labeller(node_sample_size = labeller_fn), scales = "free_y", ncol=1)+
  labs(x="Number of Input Rules", y='Rejection Rate (%)')+
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
  geom_rect(fill="springgreen3", alpha=0.15, inherit.aes = FALSE, data = boxesData,
            aes(
              xmin = 1,
              xmax = xmax,
              ymin = 0,
              ymax = ymax
            )) +
  facet_wrap(c("node_sample_size"), labeller = labeller(node_sample_size = labeller_fn), scales = "free_y", ncol=1)+
  labs(x="Number of Input Rules", y='Most Critical Resource')+
  # scale_x_discrete(labels = (xLabels))+
  theme(axis.text.x = element_text(angle = 35, hjust = 1))+
  scale_color_hue(labels = legendLabels)+ # Sets the labels
  theme(legend.position="bottom")


critical_plts <- ggarrange(rejected_plot, dominant_plot, ncol = 2, nrow = 1, common.legend = TRUE, legend="bottom")
annotate_figure(critical_plts,
    top=text_grob("Critical-Aware Greedy Optimization Perfomance")
)
ggsave(paste(base, date, test_location, "critical.pdf", sep='/'), height = 15, width = 12, limitsize = FALSE)

####################################################################################################################
################################################### GREEDY #########################################################
####################################################################################################################


base = './ResourceV3'
date = 'paper_results'
test_location = 'greedy/tests'
filename = 'test.csv'

file = paste(base, date, test_location, filename, sep='/')

## Load data from CSV
df1 <- read.csv(file=file, header = TRUE, sep = ';')

## Reduce the plot number for the publication
df1 <- df1 %>%
  filter(node_sample_size %% 10 == 0)

boxes = paste(base, date, test_location, "boxes.csv", sep='/')
boxesData <- read.csv(file=boxes, header = TRUE, sep = ';')

##############################################
################ COMMON HELPERS ##############
##############################################

#t <- df1 %>%
#  group_by(node_sample_size, node_min_value, node_max_value, resource_sample_size) %>% 
#  summarise(
#    across(everything(), list(mean = mean, sd = sd))
#  )

## Average the data
avgData <- df1 %>%
  group_by(node_sample_size, node_min_value, node_max_value, resource_sample_size) %>% 
  summarise_each(mean)
### Drop unnecessary cols
avgData <- subset(avgData, select = -c(X, test_instance))

write.csv(avgData, paste(base, date, test_location, "avg.csv", sep='/'))

avgData$node_min_value <- as.factor(avgData$node_min_value)
avgData$resource_sample_size <- as.factor(avgData$resource_sample_size)

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

rejected_plot <- ggplot(stacked_rejected_data, aes(x=resource_sample_size, y=(Value/as.numeric(as.character(resource_sample_size)))*100, col=Type, group = Type))+
  geom_point(alpha=0.2)+
  geom_line(size=1.5, alpha=0.7)+
  geom_rect(fill="springgreen3", alpha=0.15, inherit.aes = FALSE, data = boxesData,
            aes(
              xmin = 1,
              xmax = xmax,
              ymin = 0,
              ymax = ymax
            )) +
  facet_wrap(c("node_sample_size"), labeller = labeller(node_sample_size = labeller_fn), scales = "free_y", ncol=3, nrow = 2)+
  labs(x="Number of Rules", y='Rejection Rate (%)')+
  #scale_x_discrete(labels = TRUE)+
  theme_update(plot.title = element_text(hjust = 0.5))+
  theme(axis.text.x = element_text(angle = 35, hjust = 1))+
  scale_color_hue(labels = legendLabels)+ # Sets the labels
  theme(legend.position="bottom")+
  ggtitle("Greedy Optimization Perfomance")
  #theme_update(plot.title = element_text(hjust = 0.5))

rejected_plot

ggsave(paste(base, date, test_location, "greedy.pdf", sep='/'), height = 7.5, width = 12, limitsize = FALSE)
