# 7.4)In order to make a parallel comparison to the results in Exercise 6.2, we need to perform the same pre-precessing steps and set up the identical validation approach.  The following syntax provides the same pre-processing, data partition into training and testing sets, and validation set-up.
library(AppliedPredictiveModeling)
library(caret)
data(permeability)
#Identify and remove NZV predictors
nzvFingerprints = nearZeroVar(fingerprints)
noNzvFingerprints <- fingerprints[,-nzvFingerprints]
#Split data into training and test sets
set.seed(614)
trainingRows <- createDataPartition(permeability, 
                                    p = 0.75, 
                                    list = FALSE)

trainFingerprints <- noNzvFingerprints[trainingRows,]
trainPermeability <- permeability[trainingRows,]

testFingerprints <- noNzvFingerprints[-trainingRows,]
testPermeability <- permeability[-trainingRows,]

set.seed(614)
ctrl <- trainControl(method = "LGOCV")
# Next, we will find optimal tuning parameters for MARS, SVM (radial basis function), and K-NN models.
set.seed(614)

marsPermGrid <- expand.grid(degree = 1:2, nprune = seq(2,14,by=2))
marsPermTune <- train(x = trainFingerprints, y = log10(trainPermeability),
                      method = "earth",
                      tuneGrid = marsPermGrid,
                      trControl = ctrl)


RSVMPermGrid <- expand.grid(sigma = c(0.0005,0.001,0.0015),
                            C = seq(1,49,by=6))
RSVMPermTune <- train(x = trainFingerprints, y = log10(trainPermeability),
                      method = "svmRadial",
                      trControl = ctrl, 
                      tuneGrid = RSVMPermGrid)

knnPermTune <- train(x = trainFingerprints, y = log10(trainPermeability),
                     method = "knn",
                     tuneLength = 10)
# indicates that the optimal degree and number of terms that maximize $R^2$ are
marsPermTune$results$degree[best(marsPermTune$results, "Rsquared", maximize = TRUE)]
marsPermTune$results$nprune[best(marsPermTune$results, "Rsquared", maximize = TRUE)]
round(marsPermTune$results$Rsquared[best(marsPermTune$results, "Rsquared", maximize = TRUE)],2)
plotTheme <- bookTheme()
trellis.par.set(plotTheme)
plot(marsPermTune,metric="Rsquared")
# indicates that the optimal cost and sigma that maximize $R^2$ are
RSVMPermTune$results$C[best(RSVMPermTune$results, "Rsquared", maximize = TRUE)]
RSVMPermTune$results$sigma[best(RSVMPermTune$results, "Rsquared", maximize = TRUE)]
round(RSVMPermTune$results$Rsquared[best(RSVMPermTune$results, "Rsquared", maximize = TRUE)],2)
plotTheme <- bookTheme()
trellis.par.set(plotTheme)
plot(RSVMPermTune,metric="Rsquared", scales = list(x = list(log = 2)))
# indicates that the optimal number of nearest neighbors that maximize $R^2$ are
knnPermTune$results$k[best(knnPermTune$results, "Rsquared", maximize = TRUE)]
round(knnPermTune$results$Rsquared[best(knnPermTune$results, "Rsquared", maximize = TRUE)],2)
plotTheme <- bookTheme()
trellis.par.set(plotTheme)
plot(knnPermTune,metric="Rsquared")
# For these three non-linear models, the radial basis function SVM model performs best with a leave-group out cross-validated $R^2$ value of
round(RSVMPermTune$results$Rsquared[best(RSVMPermTune$results, "Rsquared", maximize = TRUE)],2)
# This is worse than the elastic net model tuned in Exercise 6.2 which had a leave-group out cross-validated $R^2$ value of 0.58.  These results indicate that the underlying relationship between the predictors and the response is likely best described by a linear structure in a reduced dimension of the original space.
# For the models tuned thus far, we would recommend the elastic net model.
# and the performance results of these models are provided in Table \ref{T:ex4Performance}.  Not surprisingly, the single tree performs the worst.  The randomness and iterative process incorporated using Random Forest improves predictive ability when using just this one predictor.  For the Cubist models, a couple of trends can be seen.  First, the no neighbor models perform better than the corresponding models that were tuned using multiple neighbors.  At the same time, using multiple committees slightly improves the predictive ability of the models.  Still, the best Cubist model (multiple committees and no neighbors) performs slightly worse than the random forest model.
# 8.4)
data(solubility)
# library("xtable")
solTrainMW <- subset(solTrainXtrans,select="MolWeight")
solTestMW <- subset(solTestXtrans,select="MolWeight")

set.seed(100)
rpartTune <- train(solTrainMW, solTrainY,
                   method = "rpart2",
                   tuneLength = 5)
rpartTest <- data.frame(Method = "RPart",Y=solTestY,
                        X=predict(rpartTune,solTestMW))
rfTune <- train(solTrainMW, solTrainY,
                method = "rf",
                tuneLength = 5)
rfTest <- data.frame(Method = "RF",Y=solTestY,
                     X=predict(rfTune,solTestMW))
rpartPerf <- data.frame(Method = "Recursive Partitioning", 
                        R2 = round(rpartTune$results$Rsquared[best(rpartTune$results, "Rsquared", maximize = TRUE)],3))
rfPerf <- data.frame(Method = "Random Forest", 
                     R2 = round(rfTune$results$Rsquared[best(rfTune$results, "Rsquared", maximize = TRUE)],3))
ex4Results <- rbind(rpartPerf,rfPerf)
print(xtable(ex4Results,
             align=c("ll|r"),
             caption = "Model performance using only Molecular Weight as a predictor.",
             label = "T:ex4Performance"),
      include.rownames=FALSE
)
cubistEx4Test <- rbind(rpartTest,
                       rfTest)
scatterTheme <- caretTheme()

scatterTheme$plot.line$col <- c("blue")
scatterTheme$plot.line$lwd <- 2

scatterTheme$plot.symbol$col <- rgb(0, 0, 0, .3)
scatterTheme$plot.symbol$cex <- 0.8
scatterTheme$plot.symbol$pch <- 16

scatterTheme$add.text <- list(cex = 0.6)
trellis.par.set(scatterTheme)
xyplot(X ~ Y | Method,
       cubistEx4Test,
       layout = c(2,3),
       panel = function(...) {
         theDots <- list(...)
         panel.xyplot(..., type = c("p", "g"))
         corr <- round(cor(theDots$x, theDots$y), 2)
         panel.text(44,
                    min(theDots$y),
                    paste("corr:", corr))
       },
       ylab = "Predicted",
       xlab = "Observed")
