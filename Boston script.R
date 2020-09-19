# Title: Regression Predictions of Boston House Prices

# Last update: 2020.9



###############
# Project Notes
###############

# Summarize Project: Investigate the Boston House Price dataset. Each record in the database describes
# a Boston suburb or town. The data was drawn from the Boston Standard Metropolitan Statistical Area (SMSA)
# in 1970. The attributes are defined as follows.

# 1. CRIM: per capita crime rate by town
# 2. ZN: proportion of residential land zoned for lots over 25,000 sq.ft.
# 3. INDUS: proportion of non-retail business acres per town
# 4. CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
# 5. NOX: nitric oxides concentration (parts per 10 million)
# 6. RM: average number of rooms per dwelling
# 7. AGE: proportion of owner-occupied units built prior to 1940
# 8. DIS: weighted distances to five Boston employment centers
# 9. RAD: index of accessibility to radial highways
# 10. TAX: full-value property-tax rate per $10,000
# 11. PTRATIO: pupil-teacher ratio by town
# 12. B: 1000(Bk − 0.63)2 where Bk is the proportion of blacks by town
# 13. LSTAT: % lower status of the population
# 14. MEDV: Median value of owner-occupied homes in $1000s


# Business Objective: Can a model be built to predict house prices in Boston Area with 80% level of certainty?


################
# Load packages
################

library(mlbench)
library(caret)
library(corrplot)


##############
# Import data
##############

# import data
data("BostonHousing")

# split-out validation dataset
# create list of 80% of rows for training
set.seed(7)
validationIndex <- createDataPartition(BostonHousing$medv, p=0.80, list = FALSE)
# select 20% of validation
validation <- BostonHousing[-validationIndex,]
# use remaining 80% of data to training and testing the models
dataset <- BostonHousing[validationIndex,]



################
# Evaluate Data
################

# check structure
str(dataset)
head(dataset)  # scales for attributes are all over the place, transforms may be useful later

# summarize attributes
summary(dataset)

# change data types
dataset[,4] <- as.numeric(as.character(dataset[,4]))

# plot
cormatrix <- cor(dataset[,1:13])

# Observations
# * nox and indus with 0.77
# * dist and indus with -0.71
# * tax and indus with 0.72
# * age and nox with 0.73
# * dist and nox with -0.77

# histograms of each attribute
par(mfrow=c(2,7))
for(i in 1:13) {
  hist(dataset[,i], main=names(dataset)[i])
}
# exponential distribution: crim, zn, age, b
# bimodel distribution: rad, tax

# density plot for each attribute
par(mfrow=c(2,7))
for(i in 1:13) {
  plot(density(dataset[,i]), main=names(dataset)[i])
}
# exponential distribution: crim, zn, age, b
# bimodel distribution: rad, tax
# skewed Gaussian distributions: nox, ptratio, lstat, possibly rm

# box and whisker of each attribute
par(mfrow=c(2,7))
for(i in 1:13) {
  boxplot(dataset[,i], main=names(dataset)[i])
}

# multivariate visualizations
# scatter plot matrix
pairs(dataset[,1:13])
# attributes showing good structure (predictable curved relationships)
# crim & age
# indus & dis
# nox & age
# nox & dis
# rm & lstat
# age & lstat
# dis & lstat
# dis & age

# correlation plot
par(mfrow=c(1,1))
correlations <- cor(dataset[,1:13])
corrplot(correlations)

# highly positively correlated features (tax v rad)
# highly negatively correlated features (dis v indus, dis v nox, dis v age)

# Summary of ideas
# feature selection to remove collinearity
# normalize to reduce effect of differing scales
# standardize to reduce effects of differing distributions
# Box-Cox transform to see if flattening out some of the distributions improves accuracy


######################
# Evaluate Algorithms
######################

# set cross-validation
trainControl <- trainControl(method = 'repeatedcv', number = 10, repeats = 3)
metric <- 'RMSE'

# LM
set.seed(7)
fit.lm <- train(medv~., 
                data=dataset, 
                method="lm", 
                metric=metric, 
                preProc=c("center", "scale"), 
                trControl=trainControl)

# GLM
set.seed(7)
fit.glm <- train(medv~., 
                 data=dataset, 
                 method="glm", 
                 metric=metric, 
                 preProc=c("center","scale"), 
                 trControl=trainControl)

# GLMNET
set.seed(7)
fit.glmnet <- train(medv~., 
                    data=dataset, 
                    method="glmnet", 
                    metric=metric,
                    preProc=c("center", "scale"), 
                    trControl=trainControl)

# SVM
set.seed(7)
fit.svm <- train(medv~., 
                 data=dataset, 
                 method="svmRadial", 
                 metric=metric,
                 preProc=c("center", "scale"), 
                 trControl=trainControl)

# CART
set.seed(7)
grid <- expand.grid(.cp=c(0, 0.05, 0.1))
fit.cart <- train(medv~., 
                  data=dataset, 
                  method="rpart", 
                  metric=metric, 
                  tuneGrid=grid,
                  preProc=c("center", "scale"), 
                  trControl=trainControl)

# KNN
set.seed(7)
fit.knn <- train(medv~., 
                 data=dataset, 
                 method="knn", 
                 metric=metric, 
                 preProc=c("center", "scale"), 
                 trControl=trainControl)


# compare algorithms
results <- resamples(list(LM=fit.lm, GLM=fit.glm, GLMNET=fit.glmnet, SVM=fit.svm, CART=fit.cart, KNN=fit.knn))
summary(results)
dotplot(results)

# Non-linear algorithms (SVM, CART, KNN) appear to have lowest RMSE and highest R2


#####################
# Feature Selection
#####################

# Find and remove highly correlated variables to see effect on linear models
set.seed(7)
cutoff <- 0.70
correlations <- cor(dataset[,1:13])
highlyCorrelated <- findCorrelation(correlations, cutoff=cutoff)
for (value in highlyCorrelated) {
  print(names(dataset)[value])
}

# create a new dataset without highly correlated features
datasetFeatures <- dataset[,-highlyCorrelated]
dim(datasetFeatures)


#######################################################
# Evaluate Algorithms with Correlated Features Removed
#######################################################

# Run algorithms using 10-fold cross-validation
trainControl <- trainControl(method="repeatedcv", number=10, repeats=3)
metric <- "RMSE"

# LM
set.seed(7)
fit.lm <- train(medv~., 
                data=datasetFeatures, 
                method="lm", 
                metric=metric,
                preProc=c("center", "scale"), 
                trControl=trainControl)

# GLM
set.seed(7)
fit.glm <- train(medv~., 
                 data=datasetFeatures, 
                 method="glm", 
                 metric=metric,
                 preProc=c("center", "scale"), 
                 trControl=trainControl)

# GLMNET
set.seed(7)
fit.glmnet <- train(medv~., 
                    data=datasetFeatures, 
                    method="glmnet", 
                    metric=metric,
                    preProc=c("center", "scale"), 
                    trControl=trainControl)

# SVM
set.seed(7)
fit.svm <- train(medv~., 
                 data=datasetFeatures, 
                 method="svmRadial", 
                 metric=metric,
                 preProc=c("center", "scale"), 
                 trControl=trainControl)

# CART
set.seed(7)
grid <- expand.grid(.cp=c(0, 0.05, 0.1))
fit.cart <- train(medv~., 
                  data=datasetFeatures, 
                  method="rpart", 
                  metric=metric,
                  tuneGrid=grid, 
                  preProc=c("center", "scale"), 
                  trControl=trainControl)

# KNN
set.seed(7)
fit.knn <- train(medv~., 
                 data=datasetFeatures, 
                 method="knn", 
                 metric=metric,
                 preProc=c("center", "scale"), 
                 trControl=trainControl)

# Compare algorithms
feature_results <- resamples(list(LM=fit.lm, GLM=fit.glm, GLMNET=fit.glmnet, SVM=fit.svm,
                                  CART=fit.cart, KNN=fit.knn))

summary(feature_results)
dotplot(feature_results)

# removing correlated features worsened the results, correlated features are helping accuracy


##############################################
# Evaluate Algorithms with Box-Cox Transform
##############################################

# Run algorithms using 10-fold cross-validation
trainControl <- trainControl(method="repeatedcv", number=10, repeats=3)
metric <- "RMSE"

# lm
set.seed(7)
fit.lm <- train(medv~., data=dataset, method="lm", metric=metric, 
                preProc=c("center","scale", "BoxCox"), trControl=trainControl)

# GLM
set.seed(7)
fit.glm <- train(medv~., data=dataset, method="glm", metric=metric, 
                 preProc=c("center","scale", "BoxCox"), trControl=trainControl)

# GLMNET
set.seed(7)
fit.glmnet <- train(medv~., data=dataset, method="glmnet", metric=metric,
                    preProc=c("center", "scale", "BoxCox"), trControl=trainControl)
# SVM
set.seed(7)
fit.svm <- train(medv~., data=dataset, method="svmRadial", metric=metric,
                 preProc=c("center", "scale", "BoxCox"), trControl=trainControl)
# CART
set.seed(7)
grid <- expand.grid(.cp=c(0, 0.05, 0.1))
fit.cart <- train(medv~., data=dataset, method="rpart", metric=metric, tuneGrid=grid,
                  preProc=c("center", "scale", "BoxCox"), trControl=trainControl)

# KNN
set.seed(7)
fit.knn <- train(medv~., data=dataset, method="knn", metric=metric, 
                 preProc=c("center","scale", "BoxCox"), trControl=trainControl)

# Compare algorithms
transformResults <- resamples(list(LM=fit.lm, GLM=fit.glm, GLMNET=fit.glmnet, SVM=fit.svm,
                                   CART=fit.cart, KNN=fit.knn))

summary(transformResults)
dotplot(transformResults)


# Box-Cox transforms decreased RMSE and increased R2 on all except CART algorithms



###################
# Improve Accuracy
###################

# Model Tuning
print(fit.svm)
# c parameter (Cost) is used by SVM
# C=1 is best, use as starting point

# tune SVM sigma and C parametres
trainControl <- trainControl(method="repeatedcv", number=10, repeats=3)
metric <- "RMSE"
set.seed(7)
grid <- expand.grid(.sigma=c(0.025, 0.05, 0.1, 0.15), .C=seq(1, 10, by=1))
fit.svm <- train(medv~., 
                 data=dataset, 
                 method="svmRadial", 
                 metric=metric, tuneGrid=grid,
                 preProc=c("BoxCox"), 
                 trControl=trainControl)
print(fit.svm)
plot(fit.svm)  # Sigma=0.1 and Cost=9 is best


# Ensemble Methods
# Bagging: Random Forest
# Boosting: Gradient Boosting, Cubist

# try ensembles
trainControl <- trainControl(method="repeatedcv", number=10, repeats=3)
metric <- "RMSE"

# Random Forest
set.seed(7)
fit.rf <- train(medv~., data=dataset, method="rf", metric=metric, preProc=c("BoxCox"),
                trControl=trainControl)

# Stochastic Gradient Boosting
set.seed(7)
fit.gbm <- train(medv~., data=dataset, method="gbm", metric=metric, preProc=c("BoxCox"),
                 trControl=trainControl, verbose=FALSE)

# Cubist
set.seed(7)
fit.cubist <- train(medv~., data=dataset, method="cubist", metric=metric,
                    preProc=c("BoxCox"), trControl=trainControl)

# Compare algorithms
ensembleResults <- resamples(list(RF=fit.rf, GBM=fit.gbm, CUBIST=fit.cubist))
summary(ensembleResults)
dotplot(ensembleResults)

# Cubist highest R2 and lowest RMSE than achieved by tuning SVM

# Tune Cubist
print(fit.cubist)

trainControl <- trainControl(method="repeatedcv", number=10, repeats=3)
metric <- "RMSE"
set.seed(7)
grid <- expand.grid(.committees=seq(15, 25, by=1), .neighbors=c(3, 5, 7, 9))
tune.cubist <- train(medv~., data=dataset, method="cubist", metric=metric,
                     preProc=c("BoxCox"), tuneGrid=grid, trControl=trainControl)
print(tune.cubist)
plot(tune.cubist)

# Cubist with committees = 25 and neighbors = 3 is top model
# finalize by creating stand alone Cubist model

# prepare data transform using training data
library(Cubist)
set.seed(7)
x <- dataset[,1:13]
y <- dataset[,14]
preprocessParams <- preProcess(x, method=c("BoxCox"))
transX <- predict(preprocessParams, x)
# train the final model
finalModel <- cubist(x=transX, y=y, committees=25)
summary(finalModel)

# transform the validation dataset
set.seed(7)
valX <- validation[,1:13]
trans_valX <- predict(preprocessParams, valX)
valY <- validation[,14]
# use final model to make predictions on the validation dataset
predictions <- data.frame(predict(finalModel, newdata=trans_valX, neighbors = 3))
# calculate RMSE
rmse <- RMSE(predictions, valY)
r2 <- R2(predictions, valY)
print(rmse)


##--- Save/load top performing model ---##

# save top performing model: Cubist
saveRDS(finalModel, 'finalModel.rds')  

# load and name model
finalModel <- readRDS('finalModel.rds')

