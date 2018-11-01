### Section 12.1 Case Study: Predicting Successful Grant Applications

library(AppliedPredictiveModeling)
library(caret)
install.packages("glmnet")
library(glmnet)
library(MASS)
install.packages("pamr")
library(pamr)
library(pls)
library(pROC)
install.packages("rms")
library(rms)
#install.packages("doMC")
#library(doMC)
#registerDoMC(12)
library(plyr)
library(reshape2)
install.packages("lubridate")
library(lubridate)
install.packages("mda")
library(mda)
install.packages("nnet")
library(nnet)

setwd("C:/Users/jxu13/Documents/GMUTeaching/OR568Fall2017/Slides")
source("CreateGrantData.R")

## Two different ways to split and resample the data. 

pre2008Data <- training[pre2008,]
year2008Data <- rbind(training[-pre2008,], testing)

set.seed(552)
test2008 <- createDataPartition(year2008Data$Class, p = .25)[[1]]

allData <- rbind(pre2008Data, year2008Data[-test2008,])
holdout2008 <- year2008Data[test2008,]

#######################
### Chapter 13
#######################

## This control object will be used across multiple models so that the
## data splitting is consistent

ctrl <- trainControl(method = "LGOCV",
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     index = list(TrainSet = pre2008),
                     savePredictions = TRUE)

set.seed(476)
mdaFit <- train(x = training[,reducedSet], 
                y = training$Class,
                method = "mda",
                metric = "ROC",
                tries = 40,
                tuneGrid = expand.grid(subclasses = 1:8),
                trControl = ctrl)
mdaFit

mdaFit$results <- mdaFit$results[!is.na(mdaFit$results$ROC),]                
mdaFit$pred <- merge(mdaFit$pred,  mdaFit$bestTune)
mdaCM <- confusionMatrix(mdaFit, norm = "none")
mdaCM

mdaRoc <- roc(response = mdaFit$pred$obs,
              predictor = mdaFit$pred$successful,
              levels = rev(levels(mdaFit$pred$obs)))
mdaRoc

plot(mdaFit, ylab = "ROC AUC (2008 Hold-Out Data)")

################################################################################
### Section 13.2 Neural Networks

nnetGrid <- expand.grid(size = c(1,10), decay = c(0, 2))
maxSize <- max(nnetGrid$size)

## Four different models are evaluate based on the data pre-processing and 
## whethera single or multiple models are used

set.seed(476)
nnetFit <- train(x = training[,reducedSet], 
                 y = training$Class,
                 method = "nnet",
                 metric = "ROC",
                 preProc = c("center", "scale"),
                 tuneGrid = nnetGrid,
                 trace = FALSE,
                 maxit = 2000,
                 MaxNWts = 1*(maxSize * (length(reducedSet) + 1) + maxSize + 1),
                 trControl = ctrl)
nnetFit

set.seed(476)
nnetFit2 <- train(x = training[,reducedSet], 
                  y = training$Class,
                  method = "nnet",
                  metric = "ROC",
                  preProc = c("center", "scale", "spatialSign"),
                  tuneGrid = nnetGrid,
                  trace = FALSE,
                  maxit = 2000,
                  MaxNWts = 1*(maxSize * (length(reducedSet) + 1) + maxSize + 1),
                  trControl = ctrl)
nnetFit2

nnetGrid$bag <- FALSE

set.seed(476)
nnetFit3 <- train(x = training[,reducedSet], 
                  y = training$Class,
                  method = "avNNet",
                  metric = "ROC",
                  preProc = c("center", "scale"),
                  tuneGrid = nnetGrid,
                  repeats = 2,
                  trace = FALSE,
                  maxit = 2000,
                  MaxNWts = 10*(maxSize * (length(reducedSet) + 1) + maxSize + 1),
                  allowParallel = FALSE, ## this will cause to many workers to be launched.
                  trControl = ctrl)
nnetFit3

set.seed(476)
nnetFit4 <- train(x = training[,reducedSet], 
                  y = training$Class,
                  method = "avNNet",
                  metric = "ROC",
                  preProc = c("center", "scale", "spatialSign"),
                  tuneGrid = nnetGrid,
                  trace = FALSE,
                  maxit = 2000,
                  repeats = 2,
                  MaxNWts = 10*(maxSize * (length(reducedSet) + 1) + maxSize + 1),
                  allowParallel = FALSE, 
                  trControl = ctrl)
nnetFit4

nnetFit4$pred <- merge(nnetFit4$pred,  nnetFit4$bestTune)
nnetCM <- confusionMatrix(nnetFit4, norm = "none")
nnetCM

nnetRoc <- roc(response = nnetFit4$pred$obs,
               predictor = nnetFit4$pred$successful,
               levels = rev(levels(nnetFit4$pred$obs)))


nnet1 <- nnetFit$results
nnet1$Transform <- "No Transformation"
nnet1$Model <- "Single Model"

nnet2 <- nnetFit2$results
nnet2$Transform <- "Spatial Sign"
nnet2$Model <- "Single Model"

nnet3 <- nnetFit3$results
nnet3$Transform <- "No Transformation"
nnet3$Model <- "Model Averaging"
nnet3$bag <- NULL

nnet4 <- nnetFit4$results
nnet4$Transform <- "Spatial Sign"
nnet4$Model <- "Model Averaging"
nnet4$bag <- NULL

nnetResults <- rbind(nnet1, nnet2, nnet3, nnet4)
nnetResults$Model <- factor(as.character(nnetResults$Model),
                            levels = c("Single Model", "Model Averaging"))
library(latticeExtra)

## Figure 13.5
useOuterStrips(
  xyplot(ROC ~ size|Model*Transform,
         data = nnetResults,
         groups = decay,
         as.table = TRUE,
         type = c("p", "l", "g"),
         lty = 1,
         ylab = "ROC AUC (2008 Hold-Out Data)",
         xlab = "Number of Hidden Units",
         auto.key = list(columns = 4, 
                         title = "Weight Decay", 
                         cex.title = 1)))

plot(nnetRoc, type = "s", legacy.axes = TRUE)

################################################################################
### Section 13.4 Support Vector Machines

library(kernlab)

set.seed(201)
sigmaRangeFull <- sigest(as.matrix(training[,fullSet]))
svmRGridFull <- expand.grid(sigma =  as.vector(sigmaRangeFull)[1],
                            C = 2^(-1:1))
set.seed(476)
svmRFitFull <- train(x = training[,fullSet], 
                     y = training$Class,
                     method = "svmRadial",
                     metric = "ROC",
                     preProc = c("center", "scale"),
                     tuneGrid = svmRGridFull,
                     trControl = ctrl)
svmRFitFull

set.seed(202)
sigmaRangeReduced <- sigest(as.matrix(training[,reducedSet]))
svmRGridReduced <- expand.grid(sigma = sigmaRangeReduced[1],
                               C = 2^(seq(-1, 1)))
set.seed(476)
svmRFitReduced <- train(x = training[,reducedSet], 
                        y = training$Class,
                        method = "svmRadial",
                        metric = "ROC",
                        preProc = c("center", "scale"),
                        tuneGrid = svmRGridReduced,
                        trControl = ctrl)
svmRFitReduced

svmPGrid <-  expand.grid(degree = 1:2,
                         scale = c(0.01),
                         C = 2^(seq(-6, -2, length = 2)))

set.seed(476)
svmPFitFull <- train(x = training[,fullSet], 
                     y = training$Class,
                     method = "svmPoly",
                     metric = "ROC",
                     preProc = c("center", "scale"),
                     tuneGrid = svmPGrid,
                     trControl = ctrl)
svmPFitFull

svmPGrid2 <-  expand.grid(degree = 1:2,
                          scale = c(0.01),
                          C = 2^(seq(-6, -2, length = 2)))
set.seed(476)
svmPFitReduced <- train(x = training[,reducedSet], 
                        y = training$Class,
                        method = "svmPoly",
                        metric = "ROC",
                        preProc = c("center", "scale"),
                        tuneGrid = svmPGrid2,
                        fit = FALSE,
                        trControl = ctrl)
svmPFitReduced

svmPFitReduced$pred <- merge(svmPFitReduced$pred,  svmPFitReduced$bestTune)
svmPCM <- confusionMatrix(svmPFitReduced, norm = "none")
svmPRoc <- roc(response = svmPFitReduced$pred$obs,
               predictor = svmPFitReduced$pred$successful,
               levels = rev(levels(svmPFitReduced$pred$obs)))


svmRadialResults <- rbind(svmRFitReduced$results,
                          svmRFitFull$results)
svmRadialResults$Set <- c(rep("Reduced Set", nrow(svmRFitReduced$result)),
                          rep("Full Set", nrow(svmRFitFull$result)))
svmRadialResults$Sigma <- paste("sigma = ", 
                                format(svmRadialResults$sigma, 
                                       scientific = FALSE, digits= 5))
svmRadialResults <- svmRadialResults[!is.na(svmRadialResults$ROC),]
xyplot(ROC ~ C|Set, data = svmRadialResults,
       groups = Sigma, type = c("g", "o"),
       xlab = "Cost",
       ylab = "ROC (2008 Hold-Out Data)",
       auto.key = list(columns = 2),
       scales = list(x = list(log = 2)))

svmPolyResults <- rbind(svmPFitReduced$results,
                        svmPFitFull$results)
svmPolyResults$Set <- c(rep("Reduced Set", nrow(svmPFitReduced$result)),
                        rep("Full Set", nrow(svmPFitFull$result)))
svmPolyResults <- svmPolyResults[!is.na(svmPolyResults$ROC),]
svmPolyResults$scale <- paste("scale = ", 
                              format(svmPolyResults$scale, 
                                     scientific = FALSE))
svmPolyResults$Degree <- "Linear"
svmPolyResults$Degree[svmPolyResults$degree == 2] <- "Quadratic"
useOuterStrips(xyplot(ROC ~ C|Degree*Set, data = svmPolyResults,
                      groups = scale, type = c("g", "o"),
                      xlab = "Cost",
                      ylab = "ROC (2008 Hold-Out Data)",
                      auto.key = list(columns = 2),
                      scales = list(x = list(log = 2))))

plot(nnetRoc, type = "s", col = rgb(.2, .2, .2, .2), legacy.axes = TRUE)
plot(svmPRoc, type = "s", add = TRUE, legacy.axes = TRUE)

################################################################################
### Section 13.5 K-Nearest Neighbors


set.seed(476)
knnFit <- train(x = training[,reducedSet], 
                y = training$Class,
                method = "knn",
                metric = "ROC",
                preProc = c("center", "scale"),
                tuneGrid = data.frame(k = c(4*(0:1)+1,20*(2:3)+1,50*(4:8)+1)),
                trControl = ctrl)
knnFit

knnFit$pred <- merge(knnFit$pred,  knnFit$bestTune)
knnCM <- confusionMatrix(knnFit, norm = "none")
knnCM
knnRoc <- roc(response = knnFit$pred$obs,
              predictor = knnFit$pred$successful,
              levels = rev(levels(knnFit$pred$obs)))

update(plot(knnFit, ylab = "ROC (2008 Hold-Out Data)"))


plot(nnetRoc, type = "s", col = rgb(.2, .2, .2, .2), legacy.axes = TRUE)
plot(svmPRoc, type = "s", add = TRUE, col = rgb(.2, .2, .2, .2), legacy.axes = TRUE)
plot(knnRoc, type = "s", add = TRUE, legacy.axes = TRUE)


## Chapter 14
library(gbm)
library(lattice)
library(partykit)
library(randomForest)
library(rpart)


####################
### Section 14.1 Basic Classification Trees

ctrl <- trainControl(method = "LGOCV",
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     index = list(TrainSet = pre2008),
                     savePredictions = TRUE)

set.seed(476)
rpartFit <- train(x = training[,fullSet], 
                  y = training$Class,
                  method = "rpart",
                  tuneLength = 30,
                  metric = "ROC",
                  trControl = ctrl)
rpartFit

plot(as.party(rpartFit$finalModel))

rpart2008 <- merge(rpartFit$pred,  rpartFit$bestTune)
rpartCM <- confusionMatrix(rpartFit, norm = "none")
rpartCM
rpartRoc <- roc(response = rpartFit$pred$obs,
                predictor = rpartFit$pred$successful,
                levels = rev(levels(rpartFit$pred$obs)))

set.seed(476)
rpartFactorFit <- train(x = training[,factorPredictors], 
                        y = training$Class,
                        method = "rpart",
                        tuneLength = 30,
                        metric = "ROC",
                        trControl = ctrl)
rpartFactorFit 
plot(as.party(rpartFactorFit$finalModel))

rpartFactor2008 <- merge(rpartFactorFit$pred,  rpartFactorFit$bestTune)
rpartFactorCM <- confusionMatrix(rpartFactorFit, norm = "none")
rpartFactorCM

rpartFactorRoc <- roc(response = rpartFactorFit$pred$obs,
                      predictor = rpartFactorFit$pred$successful,
                      levels = rev(levels(rpartFactorFit$pred$obs)))

plot(rpartRoc, type = "s", print.thres = c(.5),
     print.thres.pch = 3,
     print.thres.pattern = "",
     print.thres.cex = 1.2,
     col = "red", legacy.axes = TRUE,
     print.thres.col = "red")
plot(rpartFactorRoc,
     type = "s",
     add = TRUE,
     print.thres = c(.5),
     print.thres.pch = 16, legacy.axes = TRUE,
     print.thres.pattern = "",
     print.thres.cex = 1.2)
legend(.75, .2,
       c("Grouped Categories", "Independent Categories"),
       lwd = c(1, 1),
       col = c("black", "red"),
       pch = c(16, 3))


######################
### Section 14.3 Bagged Trees
install.packages("ipred")
library(ipred)
set.seed(476)
treebagFit <- train(x = training[,fullSet], 
                    y = training$Class,
                    method = "treebag",
                    nbagg = 5,
                    metric = "ROC",
                    trControl = ctrl)
treebagFit

treebag2008 <- merge(treebagFit$pred,  treebagFit$bestTune)
treebagCM <- confusionMatrix(treebagFit, norm = "none")
treebagCM

treebagRoc <- roc(response = treebagFit$pred$obs,
                  predictor = treebagFit$pred$successful,
                  levels = rev(levels(treebagFit$pred$obs)))

plot(rpartRoc, type = "s", col = rgb(.2, .2, .2, .2), legacy.axes = TRUE)
plot(treebagRoc, type = "s", add = TRUE, print.thres = c(.5), 
     print.thres.pch = 3, legacy.axes = TRUE, print.thres.pattern = "", 
     print.thres.cex = 1.2,
     col = "red", print.thres.col = "red")
legend(.75, .2,
       c("Grouped Categories", "Independent Categories"),
       lwd = c(1, 1),
       col = c("black", "red"),
       pch = c(16, 3))

###################
### Section 14.4 Random Forests

### For the book, this model was run with only 500 trees (by
### accident). More than 1000 trees usually required to get consistent
### results.

mtryValues <- c(10, 100)
set.seed(476)
rfFit <- train(x = training[,fullSet], 
               y = training$Class,
               method = "rf",
               ntree = 50,
               tuneGrid = data.frame(mtry = mtryValues),
               importance = TRUE,
               metric = "ROC",
               trControl = ctrl)
rfFit

rf2008 <- merge(rfFit$pred,  rfFit$bestTune)
rfCM <- confusionMatrix(rfFit, norm = "none")
rfCM

rfRoc <- roc(response = rfFit$pred$obs,
             predictor = rfFit$pred$successful,
             levels = rev(levels(rfFit$pred$obs)))

plot(treebagRoc, type = "s", col = rgb(.2, .2, .2, .2), legacy.axes = TRUE)
plot(rpartRoc, type = "s", add = TRUE, col = rgb(.2, .2, .2, .2), legacy.axes = TRUE)
plot(rfRoc, type = "s", add = TRUE, print.thres = c(.5), 
     print.thres.pch = 3, legacy.axes = TRUE, print.thres.pattern = "", 
     print.thres.cex = 1.2,
     col = "red", print.thres.col = "red")
legend(.75, .2,
       c("Grouped Categories", "Independent Categories"),
       lwd = c(1, 1),
       col = c("black", "red"),
       pch = c(16, 3))


#################
### Section 14.5 Boosting

gbmGrid <- expand.grid(n.trees = c(100,500), interaction.depth = c(1, 5), 
                       n.minobsinnode = c(10), shrinkage = c(.01, .1))

set.seed(476)
gbmFit <- train(x = training[,fullSet], 
                y = training$Class,
                method = "gbm",
                tuneGrid = gbmGrid,
                metric = "ROC",
                verbose = FALSE,
                trControl = ctrl)
gbmFit

gbmFit$pred <- merge(gbmFit$pred,  gbmFit$bestTune)
gbmCM <- confusionMatrix(gbmFit, norm = "none")
gbmCM

gbmRoc <- roc(response = gbmFit$pred$obs,
              predictor = gbmFit$pred$successful,
              levels = rev(levels(gbmFit$pred$obs)))

plot(treebagRoc, type = "s", col = rgb(.2, .2, .2, .2), legacy.axes = TRUE)
plot(rpartRoc, type = "s", add = TRUE, col = rgb(.2, .2, .2, .2), legacy.axes = TRUE)
plot(gbmRoc, type = "s", print.thres = c(.5), print.thres.pch = 3, 
     print.thres.pattern = "", print.thres.cex = 1.2,
     add = TRUE, col = "red", print.thres.col = "red", legacy.axes = TRUE)

