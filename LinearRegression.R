# install.packages("AppliedPredictiveModeling")
library(AppliedPredictiveModeling)
library(caret)
library(e1071) # misc library including skewness function
library(corrplot)
library(lattice)

## Linear regression

### Section 6.1 Case Study: Quantitative Structure- Activity
### Relationship Modeling

## training set and test set have been assigned --> solTrainX
## Also transformation has been done  --> solTrainXtrans
data(solubility)
## show all data object begins with "solT"
ls(pattern = "solT")

## show some example data by random sampling from the data  (FP represents finger print data)
set.seed(2)
sample(names(solTrainX), 12)
names(solTrainX)

### Some initial plots of the data
# Figure 6.2 (a):  weight vs. solubility (solTrainY);   plot "point"  with "grid"
xyplot(solTrainY ~ solTrainX$MolWeight, type = c("p", "g"), ylab = "Solubility (log)", main = "(a)", xlab = "Molecular Weight")

#  Rotable bonds vs. solubility 
xyplot(solTrainY ~ solTrainX$NumRotBonds, type = c("p", "g"), ylab = "Solubility (log)", xlab = "Number of Rotatable Bonds")
       
# Figure 6.2 (b): structure vs solubility; bivariate box and whisker plot
## poor visualization
xyplot(solTrainY ~ solTrainX$FP100, type = c("p", "g"), ylab = "Solubility (log)", xlab = "FP100")
dev.new()
## better visualization
bwplot(solTrainY ~ ifelse(solTrainX[,100] == 1, "structure present", "structure absent"), ylab = "Solubility (log)", main = "(b)", horizontal = FALSE)

### Find the columns that are not fingerprints (i.e. the continuous
### predictors). grep will return a list of integers corresponding to
### column names that contain the pattern "FP".

# Figure 6.3
Fingerprints <- grep("FP", names(solTrainXtrans))
featurePlot(solTrainXtrans[, -Fingerprints], solTrainY, between = list(x = 1, y = 1), type = c("g", "p", "smooth"), labels = rep("", 2))
    
# Figure 6.4  PCA 
solPCA <- prcomp(solTrainXtrans, center = TRUE, scale. = TRUE)
percentVariancePCA = solPCA$sd^2/sum(solPCA$sd^2)*100
plot(percentVariancePCA, xlab="Components", ylab="Percentage of Total Variance", type="l", main="Scree Plot of PCA Analysis")

corrplot::corrplot(cor(solTrainXtrans[, -Fingerprints]), order = "hclust", tl.cex = .8)

### Section 6.2 Linear Regression

## Use simple linear regresson with all predictors including FP

lmFitAllPredictors = lm(solTrainY ~ . , data = solTrainXtrans)
summary(lmFitAllPredictors)

lmFitNumNonHAtoms = lm(solTrainY ~solTrainXtrans$NumNonHAtoms, data = solTrainXtrans)
summary(lmFitNumNonHAtoms)

lmFitNumNonHBonds = lm(solTrainY ~solTrainXtrans$NumNonHBonds, data = solTrainXtrans)
summary(lmFitNumNonHBonds)


# Note that the results are from training data and is likely to be opmistic
# Now try on the test data

# Compue the solubility for the new test samples 
lmPred1 = predict(lmFitAllPredictors, solTestXtrans)
write.csv(solTestY, file = "st.csv")
write.csv(lmPred1, file = "pr.csv")
# Evaluate the test performance using a caret function
lmValues1 = data.frame(obs = solTestY, pred = lmPred1)
defaultSummary(lmValues1) 

## Advanced approach k folds cross validation
## Note that the resulting k learned models are averaged or otherwise combined into the final model

### Create a control function that will be used across models. 
## 10 folds CV
ctrl <- trainControl(method = "cv", number = 10)

### Linear regression model with all of the predictors. This will
### produce some warnings that a 'rank-deficient fit may be
### misleading'. This is related to the predictors being so highly
### correlated that some of the math has broken down.

set.seed(100)
lmFit1 <- train(x = solTrainXtrans, y = solTrainY, method = "lm", trControl = ctrl)

lmFit1                 
# RMSE  0.7210355  R^2 0.8768359

## Residuals analysis;  point and grid
xyplot(solTrainY ~ predict(lmFit1), type = c("p", "g"), col = "blue", xlab = "Predicted", ylab = "Observed")

xyplot(resid(lmFit1) ~ predict(lmFit1), type = c("p", "g"), col = "blue", xlab = "Predicted", ylab = "Residuals")

### Using a set of predictors reduced by unsupervised
### filtering. We apply a filter to reduce extreme between-predictor
### correlations. Note the lack of warnings.

tooHigh <- findCorrelation(cor(solTrainXtrans), .9)
trainXfiltered <- solTrainXtrans[, -tooHigh]
testXfiltered  <-  solTestXtrans[, -tooHigh]

## number of predictors reduced from 228 to 190
dim(testXfiltered )


set.seed(100)
lmFiltered <- train(x = trainXfiltered, y = solTrainY, method = "lm", trControl = ctrl)

# the results are better than the one with full 228 predictors
lmFiltered 
# RMSE 0.7113935 R^2 0.8793396 

## Now try robust linear regresson with all predictors including FP using Huber approach
## instead of sqaure error (quadratic), only square the error locally, and linear when error is large. 

library("MASS")
lmFitAllPred2 = rlm(solTrainY ~ . , data = solTrainXtrans)
summary(lmFitAllPred2)

# Note that the results are again from training data and is likely to be opmistic
# Now try on the test data

# Compue the solubility for the new test samples 
lmPred2 = predict(lmFitAllPred2, solTestXtrans)
# Evaluate the test performance using a caret function
lmValues2 = data.frame(obs = solTestY, pred = lmPred2)
defaultSummary(lmValues2) 

## Using robust linear regression with preprocess by PCA 
set.seed(100)
rlmPCA <- train(x = solTrainXtrans, y = solTrainY, method = "rlm", preProcess = "pca", trControl = ctrl)

# The results are not as good as the one with lm and without PCA
rlmPCA 

# Try on test  set 

### Save the test set results in a data frame                 
testResultsPCA <- data.frame(obs = solTestY, pred = predict(rlmPCA, solTestXtrans))

defaultSummary(testResultsPCA)

testResultsFTR <- data.frame(obs = solTestY, pred = predict(lmFiltered, testXfiltered))

defaultSummary(testResultsFTR)


### Section 6.3 Partial Least Squares (PLS)
# install.packages("pls")
library("pls")

# using plsr with 20 components
plsFit = plsr(solTrainY ~ . , data = solTrainXtrans, ncomp = 20)

## try on test data (316)
plsPred = predict(plsFit, solTestXtrans, ncomp = 20)

plsValue = data.frame(obs = solTestY, pred = plsPred[,,1])

defaultSummary(plsValue) 
## RMSE    0.7272922   Rsquared  0.8782193     

## Again, try 10 folds cross validation
set.seed(100)
pls <- train(x = solTrainXtrans, y = solTrainY, method = "pls", tuneGrid = expand.grid(ncomp = 20), trControl = ctrl)

pls
## RMSE   0.6977396  Rsquared  0.8837453  

testpls <- data.frame(obs = solTestY, pred = predict(pls, solTestXtrans))

defaultSummary(testpls)

## Model tuning
## Run PLS and PCR on solubility data and compare results with 10 folds CV

set.seed(100)
plsTune <- train(x = solTrainXtrans, y = solTrainY, method = "pls", tuneGrid = expand.grid(ncomp = 1:20), trControl = ctrl)
plsTune

testResultsPLS <- data.frame(obs = solTestY, pred = predict(plsTune, solTestXtrans))

defaultSummary(testResultsPLS)

set.seed(100)
pcrTune <- train(x = solTrainXtrans, y = solTrainY, method = "pcr", tuneGrid = expand.grid(ncomp = 1:35), trControl = ctrl)
pcrTune                  

plsResamples <- plsTune$results
plsResamples$Model <- "PLS"
pcrResamples <- pcrTune$results
pcrResamples$Model <- "PCR"
plsPlotData <- rbind(plsResamples, pcrResamples)

xyplot(RMSE ~ ncomp,
       data = plsPlotData,
       #aspect = 1,
       xlab = "# Components",
       ylab = "RMSE (Cross-Validation)",       
       col = c("blue","red"),
       auto.key = list(columns = 2),
       groups = Model,
       type = c("o", "g"))

# variable importance - PLS  Fig. 6.14
plsImp <- varImp(plsTune, scale = FALSE)
plot(plsImp, top = 25, scales = list(y = list(cex = .95)))

# variable importance - PCR
pcrImp <- varImp(pcrTune, scale = FALSE)
plot(pcrImp, top = 25, scales = list(y = list(cex = .95)))

### Section 6.4 Penalized Models

library("MASS")
install.packages("lars")
install.packages("elasticnet")
library("elasticnet")

# use enet
set.seed(10)
# lambda is the ridge regression penalty, enet works with matrix, converts the 
# data frame solTrainXtrans to a matrix. lambda = 0 performs lasso fit
lmRidge <- enet(x = as.matrix(solTrainXtrans), y = solTrainY, lambda = 0.001)

# Compute the solubility for the new test samples 
# specify the lasso parameter in terms of the fraction of full solution, with s = 1, this is ridge regression
RidgePred = predict(lmRidge, newx = as.matrix(solTestXtrans), s=1, mode="fraction", type = "fit")

## performance, try on test data (316)
RidgeValue = data.frame(obs = solTestY, pred = RidgePred$fit)

defaultSummary(RidgeValue) 
## RMSE    0.7193039  Rsquared 0.8812390       

## Experiment with different penalty lambda

enetGrid <- expand.grid(lambda = c(0, 0.01, .1), fraction = seq(.05, 1, length = 20))
set.seed(100)
enetTune <- train(x = solTrainXtrans, y = solTrainY,
                  method = "enet",
                  tuneGrid = enetGrid,
                  trControl = ctrl,
                  preProc = c("center", "scale"))
enetTune

## Figure 6.18
plot(enetTune)


