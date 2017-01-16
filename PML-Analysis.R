library(caret)
library(rattle)
library(rpart)
library(randomForest)
library(ipred)
library(plyr)
library(e1071)

# Set wd
setwd('~/MachineLearningAssignment/')


# Download the files
download.file('https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv',
              'pml-training.csv')
download.file('https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv',
              'pml-testing.csv')
# Cruft removal values
na.values <- c('NA','#DIV/0!', '')
training <- read.csv('pml-training.csv', na.strings = na.values)
testing <- read.csv('pml-testing.csv', na.strings = na.values)

# Strip summary columns that are primarily NAs (and
# don't provide useful values for each attempt anyway since they're summaries)
# for both testing and training sets.

# Also remove the category columns
training_clean <- training[,(colSums(is.na(training)) == 0)]
training_clean <- training_clean[,-(1:7)]

testing_clean <- testing[,(colSums(is.na(testing)) == 0)]
testing_clean <- testing_clean[,-(1:7)]


# Set the seed for reproducibility
set.seed(4475)


# Split training set 60/40 to make a cross-validation set to establish
# the expected rate of error.

inTrain <-createDataPartition(y=training_clean$classe, p = 0.6, list=FALSE)
training <- training_clean[inTrain,]
validation <- training_clean[-inTrain,]

# Do a quick decision tree using rpart, just to get a sense of how effective
# a low accuracy sort might be on the dataset as is.
rpart_fit <- train(classe ~ ., data=training, method = 'rpart')
pred1 <- predict(rpart_fit, validation)

# Plot the decision tree using rattle
png('rpart_plot.png', width=960, height=960)
fancyRpartPlot(rpart_fit$finalModel)
dev.off()

# Get the out of sample error for the cross-validation data
confusionMatrix(pred1, validation$classe)

# Use a bagged CART for accuracy of a multi-variate
# data set but efficiency of processing,
# since the computationally easy method produced < 50% out-of-sample in the
# reserved cross-validation data.
bag_fit <- train(classe ~ ., data = training, method = 'treebag')
pred2 <- predict(bag_fit, validation)

# Plot the importance of variables as a sanity check on the model
# i.e., does it see some of the same major variables as rpart_fit?
png('bag_fit_var_import.png', width=960, height=960)
plot(varImp(bag_fit))
dev.off()

# Get the out of sample error for the cross-validation data
confusionMatrix(pred2, validation$classe)

# Based on output, expect an out of sample error for the final testing data of
# only ~1% using treebag

# Print final values against 20 external samples
predict(bag_fit, testing_clean)
