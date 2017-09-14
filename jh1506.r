# 06 Simple SVM Stepest Desecnt

set.seed(10)
randn = function(...){array(rnorm(prod(...)),dim=list(...))}
rand = function(...){array(runif(prod(...)),dim=list(...))}
zeros = function(...){array(0,dim=list(...))}
ones = function(...){array(1,dim=list(...))}
mod = function(a,b){a%%b}

#%%%%%%%%%%% Constants %%%%%%%%%%%%%%%%%%%%
N=30        #%% Sample number %%% 10 20 30
NOISEL=0.0  #%% NOISE LEVEL in Training Samples %%% 0.0 0.1 0.2 0.3
CCC=20000   #%% PARAMETR for soft margin %%%
SUPPORT=0.1 #%% Judgement of support vector %%% 0.1 1 10 100
CYCLE=20000 #%% Cycle of optimization process
MODCYC=1000 #%% Drawing cycle
ETA0=0.05   #%% Optimization coefficient
LLL=100     #%% Parameter for Condition, dot(ydata,alpha)=0
#%%%%%%%%%%%%%%%%%%%%%%%%%Ture parameter %%%%%%%%%%%%%%%%%%%%
w0=c(1,1)
b0=-1
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Make Inputs and Outputs %%%%%%%%%%%%%%%%%%%%%%%%%%%
xdata=rand(2,N)     #%% (2,N)   Inputs of samples
yvalue=w0 %*% xdata+b0+NOISEL*(1-2*rand(1,N))
ydata=sign(yvalue)  #%% (1,N) Outputs of samples
yflag=(ydata+1)/2
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
par(mfrow = c(1,1))
par(mar = c(3, 3, 2, 1))
plot(xdata[1,],xdata[2,],col=c("red","blue")[yflag+1],type="p",pch=20)
plot(function(x){-w0[1]/w0[2]*x-b0/w0[2]},0,1,add=T,lty=2)
getGraphicsEvent(prompt="Waiting for input", onMouseDown=function(buttons,x,y){"ok"})
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Kernel%%%%%%%%%%%%%%%%%%%%%%%%%%
KXX=t(xdata) %*% xdata   #%% (N,N)  (x,x)
KYY=t(ydata) %*% ydata   #%% (N,N)   yy
KKK=KYY*KXX              #%% (N,N) yy(x,x)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Initialize Optimization%%%%%%%%%%%%%%%%%%%%%%%%%%%%
alpha=t(solve(KKK+diag(N)+LLL*KYY) %*% ones(N))
#alpha=zeros(N)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Optimization%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%% Dual Parameters are optimized by steepest descent %%%%%%%%%%%%%%
DLOSS=zeros(CYCLE/MODCYC)
RTERM=zeros(CYCLE/MODCYC)
for(cycle in 1:CYCLE){
  ETA=ETA0*(1-(cycle-1)/CYCLE)
  anew=alpha+ETA*(1-alpha %*% t(KKK)-LLL*sum(ydata*alpha)*ydata) #%% Steepest Descent
  anew=anew-mean(ydata*anew)*ydata #%% make (dot(anew,ydata) = 0)
  anew=ifelse(anew>=0,anew,0)      #%% make (anew >= 0)
  anew=ifelse(anew>CCC,CCC,anew)   #%% make (anew <= CCC)
  alpha=anew                       #%% Update alpha
  if(mod(cycle,MODCYC)==0){
    DLOSS[cycle/MODCYC]=sum(alpha)-0.5*alpha %*% KKK %*% t(alpha)
    RTERM[cycle/MODCYC]=sum(ydata*alpha)
  }
}
sprintf('Optimization Completed')
sprintf('dot(ydata,alpha)=%e sufficiently small.',sum(ydata*alpha)) #%%%% dot(ydata,alpha)=0
par(mfrow = c(3,1))
par(mar = c(3, 3, 2, 1))
plot(DLOSS,type="l",main="Dual Loss")
plot(RTERM,type="l",main="Regularization Term")
barplot(alpha,names.arg=1:N,main="Optimized Values of Dual Variables")
getGraphicsEvent(prompt="Waiting for input", onMouseDown=function(buttons,x,y){"ok"})
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%w1 and b1 are calculated%%%%%%%%%%%%%%%%%%%%
s=which(alpha[1,] > SUPPORT)
w1=(alpha*ydata) %*% t(xdata)
b1=mean(ydata[1,s] - w1 %*% xdata[,s])
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Display Results%%%%%%%%%%%%%%
par(mfrow = c(1,1))
par(mar = c(3, 3, 2, 1))
pchs = ifelse(alpha > SUPPORT, 20, 21)
plot(xdata[1,],xdata[2,],type="p",pch=pchs,col=c("red","blue")[yflag+1])
plot(function(x){-w0[1]/w0[2]*x-b0/w0[2]},0,1,add=T,lty=2)
plot(function(x){-w1[1,1]/w1[1,2]*x-b1/w1[1,2]},0,1,add=T,lty=2,col="red")
alpha[s]
w1 %*% xdata[,s]+b1
xdata[,s]
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Generalization Error
ntest=1000
xtest=rand(ntest)
ytest=rand(ntest)
ztrue=w0[1]*xtest+w0[2]*ytest+b0
ztest=w1[1,1]*xtest+w1[1,2]*ytest+b1
sprintf('Number of Support Vectors = %g',length(s))
sprintf('Generalization: Recognition Rate = %f',mean((sign(ztrue*ztest)+1)/2))
