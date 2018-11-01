
# install.packages("AppliedPredictiveModeling")
library(AppliedPredictiveModeling)
library(caret)
library(e1071) # misc library including skewness function

### Section 3.1 Case Study: Cell Segmentation in High-Content Screening
data(segmentationOriginal)
?segmentationOriginal
segmentationOriginal #Navya
## Retain the original training set
segTrain <- subset(segmentationOriginal, Case == "Train")

## Remove the first three columns (identifier columns)
segTrainX <- segTrain[, -(1:3)]
segTrainClass <- segTrain$Class

## The column VarIntenCh3 measures the standard deviation of the intensity
## of the pixels in the actin filaments

max(segTrainX$VarIntenCh3)/min(segTrainX$VarIntenCh3)
skewness(segTrainX$VarIntenCh3)

# Example:  skewness(rnorm(100000))

## Use caret's preProcess function to transform for skewness
# preProcess estimates the transformation (centering, scaling etc.) function from the training data and can be applied to any data set with the same variables.
segPP <- preProcess(segTrainX, method = "BoxCox")

## Apply the transformations
# predict is a generic function for predictions from the results of various model fitting functions. The function invokes particular methods which depend on the class of the first argument.

segTrainTrans <- predict(segPP, segTrainX)

## Results for a single predictor
segPP$bc$VarIntenCh3   # since estimated lambda = 0.1, use 0 instead --> (log transformation)

histogram(segTrainX$VarIntenCh3, xlab = "Natural Units", type = "count")   
skewness(segTrainX$VarIntenCh3)
dev.new()
histogram(log(segTrainX$VarIntenCh3), xlab = "Log Units", type = "count")
skewness(log(segTrainX$VarIntenCh3))
skewness(segTrainTrans$VarIntenCh3)

## Results for a single predictor
segPP$bc$PerimCh1

histogram(segTrainX$PerimCh1, xlab = "Natural Units", type = "count")
skewness(segTrainX$PerimCh1)
dev.new()
histogram(segTrainTrans$PerimCh1, xlab = "Transformed Data", type = "count")
skewness(segTrainTrans$PerimCh1)

### Section 3.3 Data Transformations for Multiple Predictors

## R's prcomp is used to conduct PCA on a subset of two correlated predictors:
## channel 1 average pixel intensity and entropy

## prcomp performs a PCA analysis on the two predictors and returns the results as an object of class prcomp.
## "~" means "corresponding to" the data frame specified in data
pr <- prcomp(~ AvgIntenCh1 + EntropyIntenCh1, data = segTrainTrans, scale. = TRUE)

transparentTheme(pchSize = .7, trans = .6)

# two response types: poor segment (PS), well segment (WS)
xyplot(AvgIntenCh1 ~ EntropyIntenCh1, data = segTrainTrans, groups = segTrain$Class, xlab = "Channel 1 Avg Intensity", ylab = "Intensity Entropy Channel 1", auto.key = list(columns = 2), type = c("p", "g"), main = "Original Data", aspect = 1)

xyplot(PC2 ~ PC1, data = as.data.frame(pr$x), groups = segTrain$Class, xlab = "Principal Component #1", ylab = "Principal Component #2", main = "Transformed", xlim = extendrange(pr$x), ylim = extendrange(pr$x), type = c("p", "g"), aspect = 1)

# percentage of variance
percentVariance = pr$sd^2/sum(pr$sd^2)*100
percentVariance

## Apply PCA to the entire set of predictors.

## There are a few predictors with only a single value, so we remove these first
## (since PCA uses variances, which would be zero)

isZV <- apply(segTrainX, 2, function(x) length(unique(x)) == 1)
segTrainX <- segTrainX[, !isZV]

# Apply Boxcox, center, and scale transformations
segPP <- preProcess(segTrainX, c("BoxCox", "center", "scale"))
segTrainTrans <- predict(segPP, segTrainX)

# Apply PCA on all predictors
segPCA <- prcomp(segTrainTrans, center = TRUE, scale. = TRUE)

## Plot a scatterplot matrix of the first three components
transparentTheme(pchSize = .8, trans = .3)

panelRange <- extendrange(segPCA$x[, 1:3])
splom(as.data.frame(segPCA$x[, 1:3]), groups = segTrainClass, type = c("p", "g"), as.table = TRUE, auto.key = list(columns = 2), prepanel.limits = function(x) panelRange)

## compute the percentage of variance for each component
percentVariancePCA = segPCA$sd^2/sum(segPCA$sd^2)*100

percentVariancePCA[1:4]   # first 4 components account for 42% of variance
plot(percentVariancePCA, xlab="Component", ylab="Percentage of Total Variance", type="l", main="PCA")

 ## show the transformed values  
 head(segPCA$x[,1:5])
 
### Section 3.5 Removing Variables

## To filter on correlations, we first get the correlation matrix for the predictor set

segCorr <- cor(segTrainTrans)

library(corrplot)
corrplot(segCorr, order = "hclust", tl.cex = .35)

## caret's findCorrelation function is used to identify columns to remove.
highCorr <- findCorrelation(segCorr, cutoff = .75)  # correlation coefficient > 0.75
filteredSegData = segTrainTrans[, -highCorr]

# new plot (down from 114 predictors to 71 predictors)
corrplot(cor(filteredSegData), order = "hclust", tl.cex = .35)

# install.packages("mlbench")
library(mlbench)
data(Glass)
str(Glass)
?Glass #Added by Navya on 02/04/2018

segTrainClass <- Glass$Type

# Look at all pairwise scatter plots:
pairs(Glass)

# Look at all the correlation between all predictors:
cor( Glass[,-10] ) # drop the factor variable (the last 10th one)

# Look at the correlation of each predictor with the class label:
cor( Glass[,-10], as.numeric( Glass[,10] ) )

# Visually display all data; some have outliers and skewed

par(mfrow=c(1,1))
boxplot( Glass)

# Visually display how Na, Si, and Ca depend on the glass type:
par(mfrow=c(1,3)) 
boxplot( Glass$Na ~ Glass$Type )
boxplot( Glass$Si ~ Glass$Type )
boxplot( Glass$Ca ~ Glass$Type )
boxplot( Glass$RI ~ Glass$Type ) #Navya
boxplot( Glass$Mg ~ Glass$Type ) #Navya
boxplot( Glass$Al ~ Glass$Type ) #Navya
boxplot( Glass$K ~ Glass$Type ) #Navya
boxplot( Glass$Ba ~ Glass$Type ) #Navya
boxplot( Glass$Fe ~ Glass$Type ) #Navya

# Visually display how Mg, AL, and K depend on the glass type:
par(mfrow=c(1,3)) 
boxplot( Glass$Mg ~ Glass$Type, main="Mg" )
boxplot( Glass$Al ~ Glass$Type, main ="Al" )
boxplot( Glass$K ~ Glass$Type, main="K" )

par(mfrow=c(1,1))

# Use the "corrplot" command:
# install.packages("corrplot")
library(corrplot)
corrplot( cor( Glass[,-10] ), order="hclust" )

# Compute the skewness of each feature:
library(e1071)
apply( Glass[,-10], 2, skewness )

# Look at histograms of some of the skewed predictors:
par(mfrow=c(1,3))
hist( Glass$K ) # Looks like a data error in that we have only two  samples with a very large K value 
hist( Glass$Ba ) # Looks like a skewed distribution
hist( Glass$Mg ) # Looks multimodal
hist( Glass$RI ) #Navya
hist( Glass$Na ) #Navya
hist( Glass$Al ) #Navya
hist( Glass$Si ) #Navya
hist( Glass$Ca ) #Navya
hist( Glass$Fe ) #Navya

Glass #Navya
Glass$Mg = Glass$Mg + 1.e-6
segPP <- preProcess(Glass, method = "BoxCox")
segTrainTrans <- predict(segPP, Glass)
segPP$bc$RI
segPP$bc$Na
segPP$bc$Mg
segPP$bc$Al
segPP$bc$Si
segPP$bc$K
segPP$bc$Ca
segPP$bc$Ba
segPP$bc$Fe

histogram(Glass$RI, xlab = "Natural Units", type = "count")   
skewness(Glass$RI)
dev.new()
histogram(log(Glass$RI), xlab = "Log Units", type = "count")
skewness(log(Glass$RI))
skewness(segTrainTrans$RI)

histogram(Glass$Na, xlab = "Natural Units", type = "count")   
skewness(Glass$Na)
dev.new()
histogram(log(Glass$Na), xlab = "Log Units", type = "count")
skewness(log(Glass$Na))
skewness(segTrainTrans$Na)

histogram(Glass$Mg, xlab = "Natural Units", type = "count")   
skewness(Glass$Mg)
dev.new()
histogram(log(Glass$Mg), xlab = "Log Units", type = "count")
skewness(log(Glass$Mg))
skewness(segTrainTrans$Mg)

histogram(Glass$Al, xlab = "Natural Units", type = "count")   
skewness(Glass$Al)
dev.new()
histogram(log(Glass$Al), xlab = "Log Units", type = "count")
skewness(log(Glass$Al))
skewness(segTrainTrans$Al)

histogram(Glass$Si, xlab = "Natural Units", type = "count")   
skewness(Glass$Si)
dev.new()
histogram(log(Glass$Si), xlab = "Log Units", type = "count")
skewness(log(Glass$Si))
skewness(segTrainTrans$Si)

histogram(Glass$K, xlab = "Natural Units", type = "count")   
skewness(Glass$K)
dev.new()
histogram(log(Glass$K), xlab = "Log Units", type = "count")
skewness(log(Glass$K))
skewness(segTrainTrans$K)

histogram(Glass$Ca, xlab = "Natural Units", type = "count")   
skewness(Glass$Ca)
dev.new()
histogram(log(Glass$Ca), xlab = "Log Units", type = "count")
skewness(log(Glass$Ca))
skewness(segTrainTrans$Ca)

histogram(Glass$Ba, xlab = "Natural Units", type = "count")   
skewness(Glass$Ba)
dev.new()
histogram(log(Glass$Ba), xlab = "Log Units", type = "count")
skewness(log(Glass$Ba))
skewness(segTrainTrans$Ba)

histogram(Glass$Fe, xlab = "Natural Units", type = "count")   
skewness(Glass$Fe)
dev.new()
histogram(log(Glass$Fe), xlab = "Log Units", type = "count")
skewness(log(Glass$Fe))
skewness(segTrainTrans$Fe)

pr <- prcomp(~ RI + Na, data = segTrainTrans, scale. = TRUE)

transparentTheme(pchSize = .7, trans = .6)

# two response types: poor segment (PS), well segment (WS)
xyplot(RI ~ Na, data = segTrainTrans, groups = Glass$Type, xlab = "RI", ylab = "Na", auto.key = list(columns = 2), type = c("p", "g"), main = "Original Data", aspect = 1)

## Apply PCA to the entire set of predictors.

## There are a few predictors with only a single value, so we remove these first
## (since PCA uses variances, which would be zero)

isZV <- apply(Glass, 2, function(x) length(unique(x)) == 1)
Glass <- Glass[, !isZV]

# Apply Boxcox, center, and scale transformations
segPP <- preProcess(Glass, c("BoxCox", "center", "scale"))
segTrainTrans <- predict(segPP, Glass)
segTrainTrans$Type = as.numeric(segTrainTrans$Type)
# Apply PCA on all predictors
segPCA <- prcomp(segTrainTrans, center = TRUE, scale. = TRUE)

## Plot a scatterplot matrix of the first three components
transparentTheme(pchSize = .8, trans = .3)

panelRange <- extendrange(segPCA$x[, 1:3])
splom(as.data.frame(segPCA$x[, 1:3]), groups = Glass$Type, type = c("p", "g"), as.table = TRUE, auto.key = list(columns = 2), prepanel.limits = function(x) panelRange)

## compute the percentage of variance for each component
percentVariancePCA = segPCA$sd^2/sum(segPCA$sd^2)*100

percentVariancePCA[1:9]   # first 4 components account for 42% of variance
plot(percentVariancePCA, xlab="Component", ylab="Percentage of Total Variance", type="l", main="PCA")

## show the transformed values  
head(segPCA$x[,1:5])

# Begin of code additions by Navya on 01/31/2018
library(mlbench)
data(Soybean)
?Soybean
# End of additions by Navya on

