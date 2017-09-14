# 16 Expectation Maximization

set.seed(10)
randn = function(...){array(rnorm(prod(...)),dim=list(...))}
rand = function(...){array(runif(prod(...)),dim=list(...))}
zeros = function(...){array(0,dim=list(...))}
ones = function(...){array(1,dim=list(...))}
mod = function(a,b){a%%b}

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
K=9              #%% Components of learning clusters
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N=360            #%% Number of samples
D=2              #%% Dimension
CYCLE=400
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
gauss2d=function(x1,x2,b1,b2,s2){exp(-((x1-b1)^2+(x2-b2)^2)/(2*s2))/(2*pi*s2)}
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% make samples %%%%%%%%%%%%%%%%%%%%%%%%%%%
mk0=zeros(D,9)
for(j in 1:9){
  mk0[1,j]=floor((j-1)/3)
  mk0[2,j]=mod(j-1,3)
}
xdata=zeros(D,N)
for(i in 1:N){
  xdata[,i]=0.3*randn(2,1)+mk0[,mod(i-1,9)+1]
}
#xdata=2*rand(D,N)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tstart <- proc.time()
result=list()
for(ttt in 1:11){
  #%%%%%%%%% Initialize
  ak=1/K*ones(K)
  mk=rowMeans(xdata)+0.2*randn(2,K)
  s2=(var(xdata[1,])+var(xdata[2,]))/2*ones(K)
  an=zeros(K)
  mn=zeros(D,K)
  s2n=zeros(K)
  r=zeros(K,N)
  lk=zeros(CYCLE)
  #%%%%%%%%% Recursive EM Start
  for(cycle in 1:CYCLE){
    for(i in 1:N){
      rho=ak*gauss2d(xdata[1,i],xdata[2,i],mk[1,],mk[2,],s2)
      r[,i]=rho/sum(rho)
    }
    for(k in 1:K){
      nk=sum(r[k,])
      an[k]=nk/N
      mn[,k]=(xdata%*%r[k,])/nk
      s2n[k]=sum(r[k,]*((xdata[1,]-mk[1,k])^2+(xdata[2,]-mk[2,k])^2))/(D*nk)
    }
    ak=an
    mk=mn
    s2=s2n
    pd = outer(1:K,1:N,function(k,i){
      ak[k]*gauss2d(xdata[1,i],xdata[2,i],mk[1,k],mk[2,k],s2[k])
    })
    lk[cycle]=sum(log(colSums(pd)))
  }
  result=c(result,list(list(ak=ak,mk=mk,s2=s2,loglike=lk)))
}
proc.time() - tstart   # 38sec
#%%%%%%%%%%%%%%%%%%%%%% plot results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
par(mfrow = c(3,4))
par(mar = c(3, 3, 2, 1))
plot(xdata[1,],xdata[2,],type="p",pch=20)
lines(mk0[1,],mk0[2,],type="p",col="red",pch=4)
xd=seq(from=-0.5,to=2.5,by=0.1)
yd=seq(from=-0.5,to=2.5,by=0.1)
for(rt in result){
  zd=outer(xd,yd,function(x,y){
    z=0
    for(k in 1:K){
      z=z+rt$ak[k]*gauss2d(x,y,rt$mk[1,k],rt$mk[2,k],rt$s2[k])
    }
    return(z)
  })
  contour(xd,yd,zd,main=rt$loglike[CYCLE])
  lines(rt$mk[1,],rt$mk[2,],type="p",col="red",pch=8)
}
getGraphicsEvent(prompt="Waiting for input", onMouseDown=function(buttons,x,y){"ok"})
#%%%%%%%%%%%%%%%%%%%%%% plot loglike %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
par(mfrow = c(3,4))
par(mar = c(3, 3, 2, 1))
plot(xdata[1,],xdata[2,],type="p",pch=20)
lines(mk0[1,],mk0[2,],type="p",col="red",pch=4)
for(rt in result){
  plot(rt$loglike,type="l",ylab="",main="")
}
lapply(result,function(x){x$loglike[CYCLE]})
lapply(result,function(x){sort(x$ak)})
