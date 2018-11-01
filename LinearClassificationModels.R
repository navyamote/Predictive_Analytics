library(AppliedPredictiveModeling)
library(caret)
library(randomForest)

### Chapter 11, p. 266 two-class example
library(MASS)
install.packages("pROC")
library(pROC)

## Simulate a two class data with two predictors
## Note that the two predictor are generated based on multivariate Gaussian as specified in quadBoundaryFunc   mean=(1,0), cov = [1 .7; .7 2]

set.seed(975)
training <- quadBoundaryFunc(500)
testing <- quadBoundaryFunc(1000)
testing$class2 <- ifelse(testing$class == "Class1", 1, 0)
testing$ID <- 1:nrow(testing)

### Fit models,   library(MASS)
qdaFit <- qda(class ~ X1 + X2, data = training)
qdaFit
##
rfFit <- randomForest(class ~ X1 + X2, data = training, ntree = 2000)
rfFit

### Predict the test set, record the class 1 probabilities 
testing$qda <- predict(qdaFit, testing)$posterior[,1]
testing$rf <- predict(rfFit, testing, type = "prob")[,1]
testing$rfclass <- predict(rfFit, testing)

head(testing)

### sensitivity and specificity 
## Let class 1 be the event of interest

sensitivity(data = testing$rfclass, reference = testing$class, positive = "Class1")
specificity(data = testing$rfclass, reference = testing$class, negative = "Class2")

## Posterior (PPV or NPV)
PPV = posPredValue(data = testing$rfclass, reference = testing$class, positive = "Class1")
PPV
NPV = negPredValue(data = testing$rfclass, reference = testing$class, positive = "Class2")
NPV
## change prior
PPV = posPredValue(data = testing$rfclass, reference = testing$class, positive = "Class1", prevalence = 0.9)
PPV
## Confusion matrix
confusionMatrix(data = testing$rfclass, reference = testing$class, positive = "Class1")

## ROC curve, 
## this function assumes the second class is the event of interest, so we reverse it
rocCurve = roc(response = testing$class, predictor = testing$rf, levels = rev(levels(testing$class)))

auc(rocCurve)         # area under the curve

# by default the x-axis go backward, use the legacy.axes=TRUE to modify it

plot(rocCurve, legacy.axes = TRUE)

### Chapter 4.5 and chapter 11
### credit rating example
### Recreate the model used in the over-fitting chapter

data(GermanCredit)

## First, remove near-zero variance predictors then get rid of a few predictors 
## that duplicate values. For example, there are two possible values for the 
## housing variable: "Rent", "Own" and "ForFree". So that we don't have linear
## dependencies, we get rid of one of the levels (e.g. "ForFree")

GermanCredit <- GermanCredit[, -nearZeroVar(GermanCredit)]
GermanCredit$CheckingAccountStatus.lt.0 <- NULL
GermanCredit$SavingsAccountBonds.lt.100 <- NULL
GermanCredit$EmploymentDuration.lt.1 <- NULL
GermanCredit$EmploymentDuration.Unemployed <- NULL
GermanCredit$Personal.Male.Married.Widowed <- NULL
GermanCredit$Property.Unknown <- NULL
GermanCredit$Housing.ForFree <- NULL

## Split the data into training (80%) and test sets (20%)
set.seed(100)
inTrain <- createDataPartition(GermanCredit$Class, p = .8)[[1]]
GermanCreditTrain <- GermanCredit[ inTrain, ]
GermanCreditTest  <- GermanCredit[-inTrain, ]
str(GermanCreditTest)

## apply Support vector machine model (SVM)
library(kernlab)
set.seed(1056)
svmFit <- train(Class ~ ., data = GermanCreditTrain, method = "svmRadial", preProc = c("center", "scale"), tuneLength = 10,trControl = trainControl(method = "repeatedcv", repeats = 5))
svmFit

## generalized linear model (glm) - logistic regression 
set.seed(1056)
logisticReg <- train(Class ~ ., data = GermanCreditTrain, method = "glm", trControl = trainControl(method = "repeatedcv", repeats = 5))
logisticReg

## Model comparison
## use "resamples" to compare models that share a common set of resampled data set
## Since both cases started with the same seed, we can do pair-wised comparison

resamp <- resamples(list(SVM = svmFit, Logistic = logisticReg))
summary(resamp)
modelDifferences <- diff(resamp)
summary(modelDifferences)

### Predict the test set
creditResults <- data.frame(obs = GermanCreditTest$Class)
creditResults$prob <- predict(logisticReg, GermanCreditTest, type = "prob")[, "Bad"]
creditResults$pred <- predict(logisticReg, GermanCreditTest)
creditResults$Label <- ifelse(creditResults$obs == "Bad", "True Outcome: Bad Credit",  "True Outcome: Good Credit")

### Create the confusion matrix from the test set.
confusionMatrix(data = creditResults$pred, reference = creditResults$obs)

### Plot the probability of bad credit
histogram(~prob|Label,
          data = creditResults,
          layout = c(2, 1),
          nint = 20,
          xlab = "Probability of Bad Credit",
          type = "count")

### Calculate and plot the calibration curve
creditCalib <- calibration(obs ~ prob, data = creditResults)
xyplot(creditCalib)

### ROC curves:

### Like glm(), roc() treats the last level of the factor as the event
### of interest so we use relevel() to change the observed class data

creditROC <- roc(relevel(creditResults$obs, "Good"), creditResults$prob)
auc(creditROC)
ci.auc(creditROC)
### Note the x-axis is reversed
#plot(creditROC)
plot(creditROC, legacy.axes = TRUE)

### Lift charts
creditLift <- lift(obs ~ prob, data = creditResults)
xyplot(creditLift)

# Grant data 
# load the data (from Kaggle)
#setwd("C:/Users//OR568_R_code")
#raw <- read.csv("unimelb_training.csv")
install.packages("lubridate")
#change the wd to your own 
#setwd("C:/Users//OR568_R_code")
source("CreateGrantData.R")

## Look at two different ways to split and resample the data. A support vector
## machine is used to illustrate the differences. The full set of predictors
## is used. 

pre2008Data <- training[pre2008,]
year2008Data <- rbind(training[-pre2008,], testing)

set.seed(552)
test2008 <- createDataPartition(year2008Data$Class, p = .25)[[1]]

allData <- rbind(pre2008Data, year2008Data[-test2008,])
holdout2008 <- year2008Data[test2008,]

## Section 12.1 SVM 
## Use a common tuning grid for both approaches. 
library(kernlab)

### Section 12.2 Logistic Regression

## This control object will be used across multiple models so that the
## data splitting is consistent, LGOCV is "Monte Cralo cross validation"

ctrl <- trainControl(method = "LGOCV",
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     index = list(TrainSet = pre2008),
                     savePredictions = TRUE)

set.seed(476)
lrFit <- train(x = training[,reducedSet], 
               y = training$Class,
               method = "glm",
               metric = "ROC",
               trControl = ctrl)
lrFit

## Get the confusion matrices for the hold-out set
lrCM <- confusionMatrix(lrFit, norm = "none")
lrCM

## Get the area under the ROC curve for the hold-out set
lrRoc <- roc(response = lrFit$pred$obs,
             predictor = lrFit$pred$successful,
             levels = rev(levels(lrFit$pred$obs)))
plot(lrRoc, legacy.axes = TRUE)
lrImp <- varImp(lrFit, scale = FALSE)
lrImp

### Section 12.3 Linear Discriminant Analysis
## Fit the model to the reduced set
set.seed(476)
ldaFit <- train(x = training[,reducedSet], 
                y = training$Class,
                method = "lda",
                preProc = c("center","scale"),
                metric = "ROC",
                trControl = ctrl)
ldaFit

ldaFit$pred <- merge(ldaFit$pred,  ldaFit$bestTune)
ldaCM <- confusionMatrix(ldaFit, norm = "none")
ldaCM
ldaRoc <- roc(response = ldaFit$pred$obs,
              predictor = ldaFit$pred$successful,
              levels = rev(levels(ldaFit$pred$obs)))
plot(lrRoc, type = "s", col = rgb(.2, .2, .2, .2), legacy.axes = TRUE)
plot(ldaRoc, add = TRUE, type = "s", legacy.axes = TRUE)

################################################################################
### Section 12.4 Partial Least Squares Discriminant Analysis

## This model uses all of the predictors
set.seed(476)
install.packages("pls")
library(pls)
plsFit <- train(x = training[,fullSet], 
                y = training$Class,
                method = "pls",
                tuneGrid = expand.grid(ncomp = 1:10),
                preProc = c("center","scale"),
                metric = "ROC",
                trControl = ctrl)
plsFit


plsImpGrant <- varImp(plsFit, scale = FALSE)

bestPlsNcomp <- plsFit$results[best(plsFit$results, "ROC", maximize = TRUE), "ncomp"]
bestPlsROC <- plsFit$results[best(plsFit$results, "ROC", maximize = TRUE), "ROC"]

## Only keep the final tuning parameter data
plsFit$pred <- merge(plsFit$pred,  plsFit$bestTune)

plsRoc <- roc(response = plsFit$pred$obs,
              predictor = plsFit$pred$successful,
              levels = rev(levels(plsFit$pred$obs)))

### PLS confusion matrix information
plsCM <- confusionMatrix(plsFit, norm = "none")
plsCM


################################################################################
### Section 12.5 Penalized Models

## The glmnet model
glmnGrid <- expand.grid(alpha = c(0,  .1,  .2, .4, .6, .8, 1), lambda = seq(.01, .2, length = 40))
set.seed(476)
install.packages("glmnet")
library(glmnet)
glmnFit <- train(x = training[,fullSet], 
                 y = training$Class,
                 method = "glmnet",
                 tuneGrid = glmnGrid,
                 preProc = c("center", "scale"),
                 metric = "ROC",
                 trControl = ctrl)
glmnFit

glmnet2008 <- merge(glmnFit$pred,  glmnFit$bestTune)
glmnetCM <- confusionMatrix(glmnFit, norm = "none")
glmnetCM

glmnetRoc <- roc(response = glmnet2008$obs,
                 predictor = glmnet2008$successful,
                 levels = rev(levels(glmnet2008$obs)))
glmnetRoc

## Sparse logistic regression
install.packages("sparseLDA")
library(sparseLDA)
set.seed(476)
spLDAFit <- train(x = training[,fullSet], 
                  y = training$Class,
                  "sparseLDA",
                  tuneGrid = expand.grid(lambda = c(.1),
                                         NumVars = c(1, 5, 10, 15, 20, 50, 100, 250, 500, 1000)),
                  preProc = c("center", "scale"),
                  metric = "ROC",
                  trControl = ctrl)
spLDAFit

spLDA2008 <- merge(spLDAFit$pred,  spLDAFit$bestTune)
spLDACM <- confusionMatrix(spLDAFit, norm = "none")
spLDACM

spLDARoc <- roc(response = spLDA2008$obs,
                predictor = spLDA2008$successful,
                levels = rev(levels(spLDA2008$obs)))

update(plot(spLDAFit, scales = list(x = list(log = 10))),
       ylab = "ROC AUC (2008 Hold-Out Data)")

plot(plsRoc, type = "s", col = rgb(.2, .2, .2, .2), legacy.axes = TRUE)
plot(glmnetRoc, type = "s", add = TRUE, col = rgb(.2, .2, .2, .2), legacy.axes = TRUE)
plot(ldaRoc, type = "s", add = TRUE, col = rgb(.2, .2, .2, .2), legacy.axes = TRUE)
plot(lrRoc, type = "s", col = rgb(.2, .2, .2, .2), add = TRUE, legacy.axes = TRUE)
plot(spLDARoc, type = "s", add = TRUE, legacy.axes = TRUE)
