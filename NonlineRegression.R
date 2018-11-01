# Chapter 7

# install.packages("AppliedPredictiveModeling")
library(AppliedPredictiveModeling)
library(e1071) # misc library including skewness function
library(corrplot)
library(lattice)
library(caret) 

## training set and test set have been assigned --> solTrainX
## Also transformation has been done  --> solTrainXtrans
data(solubility)

### Chapter 7: Non-Linear Regression Models
###
### Required packages: AppliedPredictiveModeling, caret, doMC (optional), earth,
###                    kernlab, lattice, nnet

## Use Exercise 7.2 as an example

library(mlbench)
library(nnet)
#install.packages("earth")
library(earth)
#install.packages("kernlab")
library(kernlab)
#install.packages("pROC")
library(pROC)

set.seed(100)

# generate training data
trainingData = mlbench.friedman1(200,sd=1)

## trainingData$x contains a 200x10 sample predictor data
## Note that 5 of them are related to the response and the remaining five are unrelated random variables 
dim(trainingData$x)
## trainingData$y contains 200 response/outcome data
trainingData$y

## We convert the 'x' data from a matrix to a data frame
## One reason we do this is that this will give the columns names
trainingData$x = data.frame(trainingData$x)
## Look at the data using
featurePlot(trainingData$x, trainingData$y)

## Now generate a large test data set to estimate the error rate
testData = mlbench.friedman1(5000,sd=1)
testData$x = data.frame(testData$x)     ## 5,000 x 10 data samples

## Note we use the default "trainControl" bootstrap evaluations for each of the models below: 

# A K-NN model: (No cross validation here)
# 
set.seed(100)
# Without specifying train control, the default is bootstrap 
knnModel = train(x=trainingData$x, y=trainingData$y, method="knn",
                 preProc=c("center","scale"),
                 tuneLength=10)

knnModel

# plot the RMSE performance against the k
plot(knnModel$results$k, knnModel$results$RMSE, type="o",xlab="# neighbors",ylab="RMSE", main="KNNs for Friedman Benchmark")

## table the results for future display
## try the model on test data
knnPred = predict(knnModel, newdata=testData$x)
## The function 'postResample' can be used to get the test set performance values
knnPR = postResample(pred=knnPred, obs=testData$y)

knnPR

rmses = c(knnPR[1])
r2s = c(knnPR[2])
methods = c("KNN")

# A Neural Network model:
#
nnGrid = expand.grid( .decay=c(0,0.01,0.1), .size=1:10 )

set.seed(100)
# MaxNWts: The maximum allowable number of weights. There is no intrinsic limit in the code, 
# but increasing MaxNWts will probably allow fits that are very slow and time-consuming. We restrict it to 10 hidden units.
# linout: switch for linear output units. Default logistic output units.

nnetModel = train(x=trainingData$x, y=trainingData$y, method="nnet", preProc=c("center", "scale"), linout=TRUE, trace=FALSE, MaxNWts=10 * (ncol(trainingData$x)+1) + 10 + 1, maxit=500, tuneGrid = nnGrid)
                
nnetModel  
summary(nnetModel)
# Lets see what variables are most important: 
varImp(nnetModel)

nnetPred = predict(nnetModel, newdata=testData$x)
nnetPR = postResample(pred=nnetPred, obs=testData$y)

nnetPR

## cumulate the table the results for future display
rmses = c(rmses,nnetPR[1])
r2s = c(r2s,nnetPR[2])
methods = c(methods,"NN")

# Averaged Neural Network models:
#
set.seed(100)
avNNetModel = train(x=trainingData$x, y=trainingData$y, method="avNNet", preProc=c("center", "scale"), linout=TRUE,trace=FALSE,MaxNWts=10 * (ncol(trainingData$x)+1) + 10 + 1, maxit=500)

avNNetModel
# Lets see what variables are most important: 
varImp(avNNetModel)

avNNetPred = predict(avNNetModel, newdata=testData$x)
avNNetPR = postResample(pred=avNNetPred, obs=testData$y)

avNNetPR

## cumulate the table the results for future display
rmses = c(rmses,avNNetPR[1])
r2s = c(r2s,avNNetPR[2])
methods = c(methods,"AvgNN")

# MARS model:
#
marsGrid = expand.grid(.degree=1:2, .nprune=2:38)
set.seed(100)
marsModel = train(x=trainingData$x, y=trainingData$y, method="earth", preProc=c("center", "scale"), tuneGrid=marsGrid)
      
marsModel
# Lets see what variables are most important: 
varImp(marsModel)
      
marsPred = predict(marsModel, newdata=testData$x)
marsPR = postResample(pred=marsPred, obs=testData$y)

marsPR

## cumulate the table the results for future display
rmses = c(rmses,marsPR[1])
r2s = c(r2s,marsPR[2])
methods = c(methods,"MARS")

# A Support Vector Machine (SVM):
#

set.seed(100)
# tune against the cost C
svmRModel = train(x=trainingData$x, y=trainingData$y, method="svmRadial", preProc=c("center", "scale"), tuneLength=20)

svmRModel
# Lets see what variables are most important: 
varImp(svmRModel)

svmRPred = predict(svmRModel, newdata=testData$x)
svmPR = postResample(pred=svmRPred, obs=testData$y) 

svmPR

rmses = c(rmses,svmPR[1])
r2s = c(r2s,svmPR[2])
methods = c(methods,"SVM")

res = data.frame( rmse=rmses, r2=r2s )
rownames(res) = methods

# Order the dataframe so that the best results are at the bottom:
#
res = res[ order( -res$rmse ), ]
print( "Final Results" ) 
print( res )









