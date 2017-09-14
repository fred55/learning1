# 11 Variational Bayes of Normal Mixture

set.seed(10)
randn = function(...){array(rnorm(prod(...)),dim=list(...))}
rand = function(...){array(runif(prod(...)),dim=list(...))}
zeros = function(...){array(0,dim=list(...))}
ones = function(...){array(1,dim=list(...))}
mod = function(a,b){a%%b}

K0=3          #%% True clusters
STDTRUE=0.3   #%% True Standard deviation of each clusters
K=3           #%% Components of learning clusters
STD=0.3       #%% 0.1 0.2 0.3 0.4 0.5 %%% Standard deviation in learning machine
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N=100         #%% Number of samples
D=2           #%% Dimension
CYCLE=100     #%% Number of recursive process
AK0=0.01      #%% Hyperparameter of mixture ratio : 3/2 Kazuho's critical point
BK0=0.01      #%% 1/BK0 = Variance of Prior
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
gauss2d=function(x1,x2,b1,b2,s2){exp(-((x1-b1)^2+(x2-b2)^2)/2/s2)/(2*pi*s2)}
#%%%%%%%%%%%%%%%%%%%%%% True mixture ratios %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
A0=c(0.2,0.3,0.5)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% make samples %%%%%%%%%%%%%%%%%%%%%%%%%%%
truecase=1
if(truecase==1){
  M0=matrix(c(0, 0, 0, 1, 1, 1),2,3)
}
if(truecase==2){
  M0=matrix(c(0, 0, 0, 0.5, 0.5, 0),2,3)
}
if(truecase==3){
  M0=matrix(c(0.5, 0.5, 0.5, 0.5, 0.5, 0.5),2,3)
}
xdata=STDTRUE*randn(D,N)
for(i in 1:N){
  r=rand(1)
  if(r>A0[1]+A0[2]){
    xdata[,i]=xdata[,i]+M0[,3]
  }else if(r>A0[1]){
    xdata[,i]=xdata[,i]+M0[,2]
  }else{
    xdata[,i]=xdata[,i]+M0[,1]
  }
}
#%%%%%%%%%%%%%%%%%%%%%%% make data end %%%%%%%%%%%%%%
#%%%%%%%%% Initialize VB
ak =N/K*rep(1,K)+AK0
bk =N/K*rep(1,K)+BK0
mk =matrix(rowMeans(xdata),D,K)+0.1*randn(D,K)
FRE=zeros(CYCLE)
#%%%%%%%%% Recursive VB Start
for(cycle in 1:CYCLE){
  rho=t(sapply(1:K,function(k){colSums((xdata-mk[,k])^2)/(2*STD*STD)}))
  rho=digamma(ak)-digamma(sum(ak))-D/2/bk-rho
  rho=apply(rho,2,function(x){exp(x-max(x))/sum(exp(x-max(x)))})

  ak=AK0+rowSums(rho)
  bk=BK0+rowSums(rho)
  rx=xdata %*% t(rho)
  mk=rx/matrix(bk,D,K,byrow=T)
  #%%%%%%%%%%%%%%%%Free Energy
  FF1=lgamma(sum(ak))-sum(lgamma(ak))
  FF2=lgamma(K*AK0)-K*lgamma(AK0)
  FF3=D/2*sum(log(bk))-D/2*K*log(BK0)
  FF4=sum((mk^2*matrix(bk,D,K,byrow=T))/(2*STD^2))-sum((xdata^2)/(2*STD^2))
  FF5=N*D/2*log(2*pi*STD^2)
  SSS=sum(rho*log(rho))
  FreeEnergy = FF1-FF2+FF3-FF4+FF5+SSS
  FRE[cycle]=FreeEnergy
}
#%%%%%%%%%%%%%%%%
am=ak/sum(ak)
FreeEnergy
am
mk
par(mfrow = c(1,1))
par(mar = c(3, 3, 2, 1))
plot(FRE,type="l",main="Free Energy")
getGraphicsEvent(prompt="Waiting for input", onMouseDown=function(buttons,x,y){"ok"})
#%%%%%%%%%%%%%%%%%%% plot samples %%%%%%%%%%%%%
par(mfrow = c(2,2))
par(mar = c(3, 3, 2, 1))
plot(xdata[1,],xdata[2,],type="p",pch=21,col="blue",xlim=c(-1,2),ylim=c(-1,2),main="Samples")
plot(M0[1,],M0[2,],col="red",type="p",pch=22,xlim=c(-1,2),ylim=c(-1,2),main="True:Red Squares,  Estimated:Brue +")
lines(mk[1,],mk[2,],type="p",pch=3,col="blue")
#%%%%%%%%%%%%%%%%%%% plot true %%%%%%%%%%%%%
xd=seq(from=-1,to=2,by=0.1)
yd=seq(from=-1,to=2,by=0.1)
s2=2*STDTRUE*STDTRUE
ztrue=outer(xd,yd,function(x,y){
  A0[1]*gauss2d(x,y,M0[1,1],M0[2,1],s2)+
  A0[2]*gauss2d(x,y,M0[1,2],M0[2,2],s2)+
  A0[3]*gauss2d(x,y,M0[1,3],M0[2,3],s2)
})
contour(xd,yd,ztrue,col="red",main="True Probability Density Function")
#%%%%%%%%%%%%%%%%%%% plot estimated %%%%%%%%%%%%%
s2=2*STD*STD
zd=outer(xd,yd,function(x,y){
  am[1]*gauss2d(x,y,mk[1,1],mk[2,1],s2)+
  am[2]*gauss2d(x,y,mk[1,2],mk[2,2],s2)+
  am[3]*gauss2d(x,y,mk[1,3],mk[2,3],s2)
})
contour(xd,yd,zd,col="blue",main="Estimated Probability Density Function")
