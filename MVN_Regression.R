# install.packages("AppliedPredictiveModeling")
library(AppliedPredictiveModeling)
library(caret)
library(e1071) # misc library including skewness function
library(corrplot)

# Plot Bivariate normal contour
# install.packages("mvtnorm")
library(mvtnorm)

# two MVN examples
par(mfrow = c(2,3))

mu <- c(0,0) ## means
rho <- 0.5
S <- matrix(c(1,rho,rho,1), ncol=2)

# Generate multivariate normal random vectors
x <- rmvnorm(n=10000, mean=mu, sigma=S)
plot(x)

x <- seq(-3*S[1]+mu[1], 3*S[1]+mu[1], length = 50)
y <- seq(-3*S[4]+mu[2], 3*S[4]+mu[2], length = 50)
f <- function(x,y) { dmvnorm(cbind(x,y), mean=mu, sigma=S) }

# outer product of vectors x and y with the elements calculated by the defined function
z <- outer(x, y, f)

# 2D contour
contour(x,y,z, main="Bivariate Normal (corr = 0.5)")

# 3D density
persp(x, y, z, theta = 20, phi = 30, expand = 0.5)

rho <- -0.95 
S <- matrix(c(1,rho,rho,1), ncol=2)
x <- rmvnorm(n=10000, mean=mu, sigma=S)
plot(x)

x <- seq(-5*S[1]+mu[1], 5*S[1]+mu[1], length = 30)
y <- seq(-5*S[4]+mu[2], 5*S[4]+mu[2], length = 30)
f <- function(x,y) { dmvnorm(cbind(x,y), mean=mu, sigma=S) }
z <- outer(x, y, f)

# 2D contour
contour(x,y,z, main="Bivariate Normal (corr = -0.95)", color.palette=topo.colors)

# 3D density
persp(x, y, z, theta = 20, phi = 30, expand = 0.5)


## Multivariate t distribution 
# install.packages("sn")
library(sn)
n = 2500
df =  3
set.seed(20)

# Generate random multivariate t distribution vectors
# Note that rmst means random muttivariiate skewed t-dist, ahpha = [0 0] means no skew
x = rmst(n,xi=rep(0,2),nu=df,alpha=rep(0,2),Omega=diag(c(1,1)))

# generate two random t distribution numbers
y1 = rt(n,df=df)
y2 = rt(n,df=df)

par(mfrow=c(1,2))
plot(x[,1],x[,2],main="(a) Multivariate-t", xlab = "X1", ylab = "X2")
abline(h=0); abline(v=0)

plot(y1,y2,main="(b) Independent t", xlab = "X1", ylab = "X2")
abline(h=0); abline(v=0)

