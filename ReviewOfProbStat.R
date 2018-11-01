# Shape, right or left skewed 

par(mfrow = c(3,1))

x = seq(0,2, by = 0.01)
plot(x,dlnorm(x,meanlog=0, sdlog=.35), lty = 1, type = "l", lwd = 2)

plot(2-x,dlnorm(x,meanlog=0, sdlog=.35), lty = 1, type = "l", lwd = 2)

plot(x,dnorm(x-1, sd=0.25), lty = 1, type = "l", lwd = 2)

# Binomial distribution

n = 10
k <- seq(0, n, by = 1)

par(mfrow=c(2,2))

p = 0.9
Sk = (1-2*p)/sqrt(n*p*(1-p)) # skewness formula
K = 3 + (1-6*p*(1-p))/(n*p*(1-p)) # kurtosis formula 
barplot(dbinom(k, n, p), 4 , ylab = "P(X=x)", xlab ="x", names.arg = as.character(k))
text(20, 0.35, paste("p=",p,"Skewness=",round(Sk,2),"Kurtosis=",round(K,2)))

p = 0.5
Sk = (1-2*p)/sqrt(n*p*(1-p))
K = 3 + (1-6*p*(1-p))/(n*p*(1-p))
barplot(dbinom(k, n, p), 4 , ylab = "P(X=x)", xlab ="x", names.arg = as.character(k))
text(40, 0.22, paste("p=",p,"Sk=",round(Sk,2),"K=",round(K,2)))

p = 0.2
Sk = (1-2*p)/sqrt(n*p*(1-p))
K = 3 + (1-6*p*(1-p))/(n*p*(1-p))
barplot(dbinom(k, n, p), 4 , ylab = "P(X=x)", xlab ="x", names.arg = as.character(k))
text(35, 0.25, paste("p=",p,"Sk=",round(Sk,2),"K=",round(K,2)))

p = 0.02
Sk = (1-2*p)/sqrt(n*p*(1-p))
K = 3 + (1-6*p*(1-p))/(n*p*(1-p))
barplot(dbinom(k, n, p), 4 , ylab = "P(X=x)", xlab ="x", names.arg = as.character(k))
text(25, 0.6, paste("p=",p,"Sk=",round(Sk,2),"K=",round(K,2)))

# Jarque-bera test

install.packages("tseries")
library(tseries)

set.seed(1234)

x <- rnorm(1000)  # normal
jarque.bera.test(x)

x <- runif(1000)  # uniform
jarque.bera.test(x)

x <- rbinom(1000,10,.25)  # binomial (#, n, p)
jarque.bera.test(x)

# t-distribution example

x = rt(10000, df = 5)
mean(x) 
var(x) 

install.packages("e1071")
library(e1071) 
skewness(x)
kurtosis(x)

par(mfrow=c(2,2))
x=seq(-6,15,by=.01)
plot(x,dnorm(x,sd=sqrt(3.4)),type="l",lwd=2,ylim=c(0,.4), xlab="x",ylab="density",main="(a) densities")
lines(x,.9*dnorm(x)+.1*dnorm(x,sd=5),lwd=2,lty=5)
legend("topright",c("normal","mixture"),lwd=2,lty=c(1,5))

plot(x,dnorm(x,sd=sqrt(3.4)),type="l",lwd=2,ylim=c(0,.025),xlim=c(4,15),xlab="x",ylab="density",main="(b) densities")
lines(x,.9*dnorm(x)+.1*dnorm(x,sd=5),lwd=2,lty=5)
legend("topright",c("normal","mixture"),lwd=2,lty=c(1,5))

set.seed("7953")
y1 = rnorm(200,sd=sqrt(3.4))
qqnorm(y1,datax=T,main="(c) QQ plot, normal",xlab="theoretical quantiles", ylab="sample quantiles")
qqline(y1,datax=T)

# generate a random number based on binomial distribution binomial(200,0.9)
n2=rbinom(1,200,.9)
y2 = c(rnorm(n2),rnorm(200-n2,sd=5))
qqnorm(y2,datax=T,main="(d) QQ plot, mixture",xlab="theoretical quantiles", ylab="sample quantiles")
qqline(y2,datax=T)

## Fit a data set 

Yn = rnorm(100,mean=30,sd=3)
Yln = 28.97 + rlnorm(100,meanlog=-1.096,sdlog=1.5)
n2=rbinom(1,100,.9)
Ymix = c(rnorm(n2,mean=30,sd=3),rnorm(100-n2,mean=60,sd=3))

install.packages("MASS")
library(MASS)
fitdistr(Yn,"normal")
fitdistr(Yln,"normal")
fitdistr(Ymix,"normal")



