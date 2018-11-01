library(caret)
str(fattyAcids)
library(MASS)
data(oil)
str(oilType)
table(oilType)
library(mda)

oilPreProcess <- preProcess(fattyAcids)
# oilPreProcess
scaledOil <- predict(oilPreProcess,
                     newdata = fattyAcids)
Model <- lda(x = scaledOil,
                grouping = oilType)
# oil <- predict(Model, scaledOil)
# train <- fattyAcids[train_ind, ]
# test <- fattyAcids[-train_ind, ]
# test$x = data.frame(test$x)
set.seed(476)
ctrl <- trainControl(method = "LGOCV",
                     classProbs = TRUE)
knnFit <- train(x = scaledOil,
                 y = oilType,
                 method = "knn",
                 preProc = c("center","scale"),
                 metric = "ROC",
                tuneGrid = data.frame(.k = c(4*(0:5)+1,
                                             + 20*(1:5)+1,
                                             + 50*(2:9)+1)),
                 trControl = ctrl)
plot(knnFit$results$k, knnFit$results$RMSE, type="o",xlab="# neighbors",ylab="RMSE", main="KNNs")
# knnPred = predict(knnFit, newdata=test)
confusionMatrix(data = knnFit,
                reference = oilType)
# knnPR = postResample(pred=knnPred, obs=test)
# 
# knnPR
# knnFit$pred <- merge(knnFit$pred, knnFit$bestTune)
# knnRoc <- roc(response = knnFit$pred$obs,
#               predictor = knnFit$pred$successful,
#               levels = rev(levels(knnFit$pred$obs)))
# plot(knnRoc, legacy.axes = TRUE)
# ## 75% of the sample size
# smp_size <- floor(0.75 * nrow(fattyAcids))
# set.seed(123)
# train_ind <- sample(seq_len(nrow(fattyAcids)), size = smp_size)
# 
# train <- fattyAcids[train_ind, ]
# test <- fattyAcids[-train_ind, ]
# # A K-NN model:
# set.seed(100)
# # Without specifying train control, the default is bootstrap 
# knnModel = train(x=train, grouping =oilType, method="knn",
#                  preProc=c("center","scale"),
#                  tuneLength=10)
# 
# knnModel

#Neural Network
nnetGrid <- expand.grid(.size = 1:10,
                        .decay = c(0, .1, 1, 2))
maxSize <- max(nnetGrid$.size)
numWts <- 1*(maxSize * (length(scaledOil) + 1) + maxSize + 1)
set.seed(476)
nnetFit <- train(x = scaledOil,
                 y = oilType,
                 method = "nnet",
                 metric = "ROC",
                 preProc = c("center", "scale", "spatialSign"),
                 tuneGrid = nnetGrid,
                 trace = FALSE,
                 maxit = 2000,
                 MaxNWts = numWts,
                 ## ctrl was defined in the previous chapter
                 trControl = ctrl)
confusionMatrix(data = nnetFit,
                reference = oilType)
#SVM
# set.seed(202)
sigmaRangeReduced <- sigest(as.matrix(scaledOil))
svmRGridReduced <- expand.grid(.sigma = sigmaRangeReduced[1],
                                .C = 2^(seq(-4, 4)))
set.seed(476)
svmRModel <- train(x = scaledOil, y = oilType,
                     method = "svmRadial",
                     # metric = "ROC",
                     preProc = c("center", "scale"),
                     # tuneGrid = svmRGridReduced,
                     tuneLength=20
                     # fit = FALSE,
                     # trControl = ctrl
                      )
svmRModel
confusionMatrix(data = svmRModel,
                reference = oilType)
#Nonlinear Discriminant Analysis
set.seed(476)
mdaFit <- train(x = scaledOil, y = oilType,
                  method = "mda",
                  # metric = "ROC",
                  tuneGrid = expand.grid(.subclasses = 1:8),
                  trControl = ctrl)
confusionMatrix(data = mdaFit,
                reference = oilType)
# MARS model:
#
marsGrid = expand.grid(.degree=1:2, .nprune=2:38)
set.seed(100)
marsModel = train(x=scaledOil, y=oilType, method="earth", preProc=c("center", "scale"), tuneGrid=marsGrid)

marsModel
confusionMatrix(data = marsModel,
                reference = oilType)