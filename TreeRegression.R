# text book Chapter 8
#install.packages("AppliedPredictiveModeling")
library(AppliedPredictiveModeling)
library(caret)
install.packages("gbm")
library(gbm)
install.packages("ipred")
library(ipred)
install.packages("party")
library(party)
install.packages("partykit")
library(partykit)
install.packages("randomForest")
library(randomForest)
#library(RWeka)

## Single trees
library(rpart)

## training set and test set have been assigned --> solTrainX
## Also transformation has been done  --> solTrainXtrans
data(solubility)
# Make sure we don't access the unscaled variables by accident (we want to use the scaled variables): 
rm(solTrainX)
rm(solTestX)

# set up training data
trainData = data.frame( x=solTrainXtrans, y=solTrainY )

## Build a simple regression tree:  library(rpart)

# defaults: for rpart.control are cp=0.01, maxdepth=30
# decreasing cp makes deeper trees; increasing maxdepth do the same

rPartModel = rpart( y ~ ., data=trainData, method="anova", control=rpart.control(cp=0.01,maxdepth=6) ) 

# tree plotting   Fig 8.4
rpartTree = as.party(rPartModel)
dev.new()
plot(rpartTree)

# predict solubility with this regression tree: 
rPart_yHat = predict(rPartModel,newdata=data.frame(x=solTestXtrans))

## performance evaluation
rtPR = postResample(pred=rPart_yHat, obs=solTestY)
rtPR

## Bagged tree:  library(ipred)

BaggTree= bagging( y ~ ., data=trainData)

# predict solubility with this regression tree: 
Bagg_yHat = predict(BaggTree,newdata=data.frame(x=solTestXtrans))

## performance evaluation
BaggPR = postResample(pred=Bagg_yHat, obs=solTestY)
BaggPR

# fit a randomforest: library(randomForest)
#
rfModel = randomForest( y ~ ., data=trainData, ntree=500 ) # ntree=500

# predict solubility:
rf_yHat = predict(rfModel,newdata=data.frame(x=solTestXtrans))

## performance evaluation
rfPR = postResample(pred=rf_yHat, obs=solTestY)
rfPR

# Boosted tree: library(gbm)
set.seed=100
#gbmModel = gbm(  y ~ . , data=trainData) 
gbmModel = gbm.fit( solTrainXtrans, solTrainY, distribution="gaussian", n.trees =100, interaction.depth=7, shrinkage=0.1)

# predict solubility:
gbm_yHat = predict(gbmModel,n.trees = 100, newdata = data.frame(x=solTestXtrans))

## performance evaluation
gbmPR = postResample(pred=gbm_yHat, obs=solTestY)
gbmPR

## create a 10 folds CV control
ctrl <- trainControl(method = "cv", number = 10)

set.seed(100)
## rpart2 is used to tune max depth 
rpartTune <- train(x = solTrainXtrans, y = solTrainY, method = "rpart2",tuneLength = 20, trControl = ctrl)
rpartTune

FinalTree = rpartTune$finalModel

rpartTree = as.party(FinalTree)
dev.new()
plot(rpartTree)



