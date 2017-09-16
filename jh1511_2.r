# 11 Variational Bayesian Learning
#    True 5 components
#    Learner 3,4,5,6,7

set.seed(10)
randn = function(...){array(rnorm(prod(...)),dim=list(...))}
rand = function(...){array(runif(prod(...)),dim=list(...))}
zeros = function(...){array(0,dim=list(...))}
ones = function(...){array(1,dim=list(...))}
mod = function(a,b){a%%b}

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
STD0=5
STD=5         #%% Standard deviation in learning machine
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N=500         #%% Number of samples
D=1           #%% Dimension
CYCLE=200     #%% Number of recursive process
AK0=3         #%% Hyperparameter of mixture ratio : 3/2 Kazuho's critical point
BK0=1
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
gauss1d=function(x,b,s2){exp(-(x-b)^2/2/s2)/sqrt(2*pi*s2)}
#%%%%%%%%%%%%%%%%%%%%%% True mixture ratios %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%% Set True
A0=c(0.1,0.15,0.25,0.35,0.15)
M0=c(10,30,50,70,90)
#%%%%%%%%%%%%%%%%%%%%%%%%%% Sample generation
xdata=zeros(D,N)
for(i in 1:N){
  r=rand(1)
  if(r<sum(A0[1:1])){
    mn=M0[1]
  }else if(r<sum(A0[1:2])){
    mn=M0[2]
  }else if(r<sum(A0[1:3])){
    mn=M0[3]
  }else if(r<sum(A0[1:4])){
    mn=M0[4]
  }else{
    mn=M0[5]
  }
  xdata[D,i]=mn+STD0*randn(1)
}
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
result=list()
for(K in 3:7){ #%% Components of learning clusters
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
  result=c(result,list(list(K=K,am=am,mk=mk,FreeEnergy=FreeEnergy,FRE=FRE)))
}
#%%%%%%%%%%%%%%%%%%%%%% plot samples %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
par(mfrow = c(2,3))
par(mar = c(3, 3, 2, 1))
hist(xdata,breaks=50,xlab="",ylab="",main="Sample Histogram")
xd=1:100
ytrue=zeros(100)
for(k in 1:5){
  ytrue=ytrue+A0[k]*gauss1d(xd,M0[k],STD^2)
}
for(rt in result){
  yd=zeros(100)
  for(k in 1:rt$K){
    yd=yd+rt$am[k]*gauss1d(xd,rt$mk[k],STD^2)
  }
  ymax=max(yd,ytrue)
  plot(xd,yd,type="l",col="blue",ylim=c(0,ymax),main="Red: true, Blue: Estimated")
  lines(xd,ytrue,type="l",col="red")
  legend("topright",legend=sprintf("K=%g",rt$K))
  legend("topleft",legend=sprintf("F=%g",rt$FreeEnergy))
}
