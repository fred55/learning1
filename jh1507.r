# 07 Gaussian SVM

set.seed(10)
randn = function(...){array(rnorm(prod(...)),dim=list(...))}
rand = function(...){array(runif(prod(...)),dim=list(...))}
zeros = function(...){array(0,dim=list(...))}
ones = function(...){array(1,dim=list(...))}
mod = function(a,b){a%%b}

#%%%%%%%%%%% Constants %%%%%%%%%%%%%%%%%%%%
N=50        #%% Training Sample number
BETA=5      #%% 1 5 25 125 Parameter of Gaussian Kernel
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
NOISEY=0.0  #%% NOISE LEVEL in Training Samples
CCC=1000    #%% PARAMETR for soft margin
CYCLE=20000 #%% Cycle of optimization process
MODCYC=1000 #%% Drawing cycle
ETA0=0.01   #%% Optimization coefficient
LLL=1       #%% Parameter for Condition dot(ydata,alpha)=0
SUPPORT=0.01#%% Judgement of support vector
COMP=3      #%% complexity of the true distribution
#%%%%%%%%%%%%%%%%%%%%%%%%% Ture parameter %%%%%%%%%%%%%%%%%%%%
w0=c(1,1)
b0=-1
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Make Inputs and Outputs %%%%%%%%%%%%%%%%%%%%%%%%%%%
my_true = function(x,y){0.5*w0[1]*(1+sin(COMP*pi*x))+w0[2]*y+b0}
xtest = seq(from=0,to=1,by=0.01)
ytest = seq(from=0,to=1,by=0.01)
ztrue=outer(xtest,ytest,function(x,y){my_true(x,y)})
xdata=rand(2,N)     #%% (2,N)   Inputs of samples
yvalue=my_true(xdata[1,],xdata[2,])+NOISEY*(1-2*rand(1,N))
ydata=sign(yvalue)  #%% (1,N) Outputs of samples
yflag=(ydata+1)/2
par(mfrow = c(1,1))
par(mar = c(3, 3, 2, 1))
plot(xdata[1,],xdata[2,],col=c("blue","red")[yflag+1],type="p",pch=20,main="Blue -1 : Red : +1")
contour(xtest,ytest,ztrue,levels=0,add=T,col="red")
getGraphicsEvent(prompt="Waiting for input", onMouseDown=function(buttons,x,y){"ok"})
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Kernel%%%%%%%%%%%%%%%%%%%%%%%%%%
KXX=outer(1:N,1:N,function(i,j){
  dx=xdata[,i]-xdata[,j]
  return(colSums(dx^2))
})
KXX=exp(-BETA*KXX)     #%% (N,N)  (x,x)
KYY=t(ydata) %*% ydata #%% (N,N)   yy
KKK=KYY*KXX            #%% (N,N) yy(x,x)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Initialize Optimization%%%%%%%%%%%%%%%%%%%%%%%%%%%%
alpha=t(solve(KKK+diag(N)+LLL*KYY) %*% ones(N))
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
sprintf('dot(ydata,alpha)=%e should be sufficiently small.',sum(ydata*alpha)) #%%%% dot(ydata,alpha)=0
par(mfrow = c(3,1))
par(mar = c(3, 3, 2, 1))
plot(DLOSS,type="l",main="Dual Loss")
plot(RTERM,type="l",main="Regularization Term")
barplot(alpha,names.arg=1:N,main="Optimized Values of Dual Variables")
getGraphicsEvent(prompt="Waiting for input", onMouseDown=function(buttons,x,y){"ok"})
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% w and b1 are calculated %%%%%%%%%%%%%%%%%%%%
s=which(alpha[1,]>SUPPORT)
b1=mean(ydata[1,s] - (alpha[1,s] * ydata[1,s]) %*% KXX[s,s])
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Display Results %%%%%%%%%%%%%%
par(mfrow = c(1,1))
par(mar = c(3, 3, 2, 1))
pchs = ifelse(alpha > SUPPORT, 20, 21)
plot(xdata[1,],xdata[2,],type="p",pch=pchs,col=c("blue","red")[yflag+1],main="Red: True,  Black: Estimated")
my_fun = function(x){
  tx=xdata[1,s]-x[1]
  ty=xdata[2,s]-x[2]
  tf=exp(-BETA*(tx^2+ty^2))*ydata[1,s]*alpha[1,s]
  return(sum(tf)+b1)
}
contour(xtest,ytest,ztrue,levels=0,add=T,col="red")
ztest=array(apply(expand.grid(xtest, ytest),1,my_fun),dim=c(length(xtest),length(ytest)))
contour(xtest,ytest,ztest,levels=0,add=T,col="black")
alpha[s]
(alpha[1,s] * ydata[1,s]) %*% KXX[s,s] + b1
xdata[,s]
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Generalization Error
sprintf('Number of Support Vectors = %g',length(s))
sprintf('Generalization: Recognition Rate = %f',mean((sign(ztrue*ztest)+1)/2))
