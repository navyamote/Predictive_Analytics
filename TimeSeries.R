############################
###  Time series models and forecasting
############################

library(tseries)

data(Mishkin, package="Ecdat")

# pai1 = one-month inflation rate in percent, annual rate) 
x= as.vector(Mishkin[,1])  
n = length(x)
year = seq(1950 + 1/12,1990+11/12,1/12)

#####  Time series plots
par(mfrow=c(2,1))
plot(year,x,ylab="inflation rate",type="l",xlab="year",cex.lab=1.5,
   cex.axis=1.5,cex.main=1.3,main="(a)")
plot(year[2:n],diff(x),ylab="change in rate",type="l",xlab="year",cex.lab=1.5,
   cex.axis=1.5,cex.main=1.2,main="(b)")
   
# pai1 = one-month inflation rate  (in percent, annual rate) 
x= as.vector(Mishkin[,1])  
year = seq(1950 + 1/12,1990+11/12,1/12)
n = length(year)
logn = log(n)

par(mfrow=c(1,2))
acf(x,cex.axis=1.5,cex.lab=1.5,cex.main=1.2,main="Inflation rate")
acf(diff(x),cex.axis=1.5,cex.lab=1.5,cex.main=1.2,
main="Change in inflation rate")

Box.test(diff(x),lag=10)

####### Autocorrelation functions

par(mfrow = c(2,2))
phi = 0.95
x = seq(1,15)
y = phi^x
plot(x,y, ylim= c(-1,1), type = "o",xlab = 'h',ylab=expression(paste(phi,'(h)')), main=expression(paste(phi," = 0.95")))
abline(0,0)

phi = 0.75
x = seq(1,15)
y = phi^x
plot(x,y, ylim= c(-1,1), type = "o",xlab = 'h',ylab=expression(paste(phi,'(h)')), main=expression(paste(phi," = 0.75")))
abline(0,0)

phi = 0.2
x = seq(1,15)
y = phi^x
plot(x,y, ylim= c(-1,1), type = "o",xlab = 'h',ylab=expression(paste(phi,'(h)')), main=expression(paste(phi," = 0.2")))

abline(0,0)
phi = -0.9
x = seq(1,15)
y = phi^x
plot(x,y, ylim= c(-1,1), type = "o",xlab = 'h',ylab=expression(paste(phi,'(h)')), main=expression(paste(phi," = -0.9")))
abline(0,0)


set.seed(8716)
e = rnorm(200)
x1 = e
for (i in 2:200)
{
x1[i] = .98*x1[i-1]+e[i]
}
x2 = e
for (i in 2:200)
{
x2[i] = -.6*x2[i-1]+e[i]
}
x3 = e
x3 = cumsum(e)
x4 = e
for (i in 2:200)
{
x4[i] = 1.01*x4[i-1]+e[i]
}

par(mfrow=c(2,2),cex.axis=1.15,cex.lab=1.15,cex.main=1.15)

plot(x1,type="l",xlab="Time",ylab=expression(Y[t]),
   main=expression(paste(phi," = 0.98")))

plot(x2,type="l",xlab="Time",ylab=expression(Y[t]),
  main=expression(paste(phi == - 0.6)))

plot(x3,type="l",xlab="Time",ylab=expression(Y[t]),
   main=expression(paste(phi," = 1")))

plot(x4,type="l",,xlab="Time",ylab=expression(Y[t]),
   main=expression(paste(phi," = 1.01")))
   

data(bmw,package="evir")
bmw = as.vector(bmw)
par(mfrow=c(2,1))
plot(bmw,type="l",xlab="Time")
# Check ACF 
acf(bmw, main='BMW log returns', lag = 10)
# test 
Box.test(bmw, type = "Ljung-Box", lag = 5)

# Fit AR(1) model
fitAR1 = arima(bmw, order = c(1,0,0))
fitAR1
### Test residuals, for arima(p,i,q), fitdf = p+q
Box.test(residuals(fitAR1), type = "Ljung-Box", lag = 5, fitdf = 1)


data(bmw,package="evir")
bmw = as.vector(bmw)

fitAR1 = arima(bmw, order = c(1,0, 0))

par(mfrow=c(1,3),cex.axis=1.15,cex.lab=1.15)
acf( residuals(fitAR1),lag.max=20 , main="")
qqnorm(residuals(fitAR1),datax=T,main="AR(1) resid")
qqline(residuals(fitAR1),datax=T)
plot(residuals(fitAR1),ylab="Residual")

# pai1 = one-month inflation rate  (in percent, annual rate) 
x= as.vector(Mishkin[,1])  

fit = arima(x,c(1,0,0))

par(mfrow=c(1,2))
acf(x,main="Inflation rate")
acf(fit$resid,main="Residuals from AR(1)")

Box.test(fit$resid,type = "Ljung-Box", lag = 10, fitdf = 1)

# How about on change of  inflation rate??

fitd = arima(diff(x),c(1,0,0))

par(mfrow=c(1,2))
acf(diff(x),main="Change of Inflation rate")
acf(fitd$resid,main="Residuals from AR(1)")

# Still not good enough
Box.test(fit$resid,type = "Ljung-Box", lag = 10, fitdf = 1)

####### (B) Figure 9.9 Theoretical ACF for any ARMA model

x1 = ARMAacf(ar=c(.5,-.3),lag.max=10)
x1 = as.vector(x1)

x2 = ARMAacf(ar=c(.5,.15),lag.max=10)
x2 = as.vector(x2)

x3 = ARMAacf(ar=c(.15,.8), lag.max=10)
x3 = as.vector(x3)

plot(x1,xlab="lag",ylab=expression(paste("ACF(",phi,'1,',phi,'2)')), main= "ACF of three AR(2) processes",cex.axis=1.5, cex.lab=1.2, cex=2,cex.main=1.5,pch="*",type="b",ylim=c(-.5,1))
lines(x2,cex.axis=1.5, cex.lab=1.5, cex=2,pch="o",type="b")
lines(x3,cex.axis=1.5, cex.lab=1.5, cex=2,pch="x",type="b")
abline(h=0)
legend(6,-.1,c("(0.5, -0.3)", "(0.5, 0.15)","(0.15, 0.8)") , pch=c("*","o","x"),cex=1.5,box.lty=0 )


# pai1 = one-month inflation rate  (in percent, annual rate) 
x= as.vector(Mishkin[,1])  
year = seq(1950 + 1/12,1990+11/12,1/12)
n=length(year)
logn=log(n)

#####  Fitting AR models
resultsdiff = matrix(0,nrow=20,ncol=3)
for (i in 1:20)
{
fit = arima(diff(x),order=c(i,0,0))
resultsdiff[i,2] = fit$aic
## BIC score
resultsdiff[i,3] = resultsdiff[i,2] + (logn-2)*i
resultsdiff[i,1] = i
}

plot(resultsdiff[,1],resultsdiff[,2],xlab="p",ylab="criterion", cex.lab=1.35,cex.axis=1.35, main="AR fits to changes in inflation rate", cex.main=1.35,cex=2,pch="*",ylim=c(2440,2560))
points(resultsdiff[,1],resultsdiff[,3],pch="o",cex=2)
legend(12,2565,c("AIC","BIC"),pch=c("*","o"),cex=2,box.lty=0)

# Automatically find the best model for inflation rate change 

# install.packages("forecast")
library(forecast)

# pai1 = one-month inflation rate  (in percent, annual rate) 
x= as.vector(Mishkin[,1])  

# Automatically find the best AR model with minimum BIC score for inflation rate change 
fit = auto.arima(diff(x), max.p = 10, max.q = 0, ic = "bic")
fit

# check whiteness
acf(fit$resid)     # same as acf(residuals(fit))
# Box test also confirms that we can't reject the null hypothesis that the residuals are white noise
Box.test(fit$resid, lag  = 10)

# What about inflation rate

fit = auto.arima(x, max.p = 10, max.q = 0, ic = "aic")
fit
# check whiteness
acf(fit$resid)     # acf(residuals(fit))
# Box test also confirms that we can't reject the null hypothesis
Box.test(fit$resid, lag  = 10)


# pai1 = one-month inflation rate  (in percent, annual rate) 
x= as.vector(Mishkin[,1])  
year = seq(1950 + 1/12,1990+11/12,1/12)
n=length(year)
logn=log(n)

#####  Fitting AR models
resultsdiff = matrix(0,nrow=10,ncol=3)
for (i in 1:10)
{
fit = arima(diff(x),order=c(0,0,i))
resultsdiff[i,2] = fit$aic
resultsdiff[i,3] = resultsdiff[i,2] + (logn-2)*i
resultsdiff[i,1]=i
}

plot(resultsdiff[,1],resultsdiff[,2],xlab="q",ylab="criterion", cex.lab=1.35,cex.axis=1.35, main="MA fits to changes in inflation rate", cex.main=1.35,cex=2,pch="*",ylim=c(2440,2510))
points(resultsdiff[,1],resultsdiff[,3],pch="o",cex=2)
legend(1,2510,c("AIC","BIC"),pch=c("*","o"),cex=2,box.lty=0)

# MA model checking

fitma = arima(diff(x), order = c(0,0,3))
# check whiteness
acf(fitma$resid)     # acf(residuals(fit))
# Box test also confirms that we can't reject the null hypothesis
Box.test(fitma$resid, lag  = 10)

# This will return the same results automatically
fit = auto.arima(diff(x), max.p = 0, max.q = 10, ic = "aic")
fit

######### fit AMRA(1,1) with inflation rate change

fitAM1 = arima(diff(x), order = c(1, 0, 1))

par(mfrow=c(1,3),cex.axis=1.15,cex.lab=1.15)
acf( residuals(fitAM1),lag.max=20 , main="")
qqnorm(residuals(fitAM1),datax=T,main="ARMA(1,1) resid")
qqline(residuals(fitAM1),datax=T)
plot(residuals(fitAM1),ylab="Residual")

Box.test(fitAM1$resid, lag  = 10)

# How about auto fitting?  Same results
fit = auto.arima(diff(x), max.p = 10, max.q = 10, ic = "aic")

# Unit root test, note that inflation rate is difficult to fit, let's fit a stationary (d=0) inflation rate to ARMA model
fit = auto.arima(x, max.p = 10, max.d = 0, max.q = 10, ic = "aic")
fit

# polyroot finds all solutions (zeros)
polyroot( c( 1, - fit$coef[1], -fit$coef[2]))

# Note that uniroot only solves for a unique solution within an interval
uniroot(function(r) fit$coef[2]*r^2 + fit$coef[1] * r - 1, c(0.7,1.5))
# The two solutions are: 1.02153, 4.376856; one of them is close to 1


## Fit Inflation data with a stationary ARMA(p,0,q) model 

data(Mishkin,package="Ecdat")

# pai1 = one-month inflation rate  (in percent, annual rate) 
x= as.vector(Mishkin[,1])  

fit = auto.arima(x, max.p = 10, max.d = 0, max.q = 10, ic = "aic")
fit
## Finds all solutions (zeros) of the polynomial
polyroot( c( 1, - fit$coef[1], -fit$coef[2]))

## Unit root test: Dickey-Fuller test, (null hypothesis: there is a unit root)
adf.test(x)
## Unit root test: Phillips-Perron test, (null hypothesis: there is a unit root)
pp.test(x)
## KSPP test (null hypoithesis: the process is stationary)
kpss.test(x)

# Forecasting
fit = arima(diff(x), order = c(1,0, 1))
pp = predict(fit, 10)

plot(1:10,pp$pred,type="b",lty=2,ylim=c(0,4),pch="*",cex=3,
   xlab="k",ylab="forecast",main="Inflation Rate Change", cex.axis=1.15,cex.lab=1.15,lwd=3)
abline(h = mean(diff(x)), lty = 2, lwd = 2)

# Plotting the confidence interval
X = c(1:10)
plot(X,pp$pred,type="l",ylim=c(-10,10),col="blue")
lines(X,pp$pred+1.96*pp$se,col="red")
lines(X,pp$pred-1.96*pp$se,col="red")

