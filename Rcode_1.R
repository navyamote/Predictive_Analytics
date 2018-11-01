# install.packages("mlbench")
library(AppliedPredictiveModeling)
library(caret)
library(e1071)
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
Glass$K = Glass$K + 1.e-6
Glass$Ba = Glass$Ba + 1.e-6
Glass$Fe = Glass$Fe + 1.e-6
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