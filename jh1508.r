# 08 K Means

library("deldir")

set.seed(10)
randn = function(...){array(rnorm(prod(...)),dim=list(...))}
rand = function(...){array(runif(prod(...)),dim=list(...))}
zeros = function(...){array(0,dim=list(...))}
ones = function(...){array(1,dim=list(...))}
mod = function(a,b){a%%b}

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N=200          #%% Training samples
K=8            #%% Learning Components
CYCLE=20       #%% Repeated Time
#%%%%%%%%%%%%%%%%%%%%%%%% Choose SAMPLE Distributions %%%%%%%%%%%%
TYPESET=3      #%% Training Sample Type
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if(TYPESET==1){ #%%%%%%%%% (1) Uniform
  x=rand(2,N)
}
if(TYPESET==2){ #%%%%%%%%% (2) Gaussian
  x=array(c(0.1,0.12,0.12,0.1),dim=c(2,2)) %*% randn(2,N)+0.5
}
if(TYPESET==3){ #%%%%%%%%% (3) Gaussian mixtue
  x0=t(array(c(0,1,0,1,0.5,0,0,1,1,0.5),dim=c(5,2)))
  x=zeros(2,N)
  for(i in 1:N){
    j=1+floor(5*rand(1,1))
    x[,i]=x0[,j]
  }
  x=x+0.05*randn(2,N)
}
if(TYPESET==4){ #%%%%%%%%% (4) line %%%
  x=zeros(2,N)
  x[1,]=rand(1,N)
  x[2,]=4*(0.5-x[1,])^2
}
if(TYPESET==5){ #%%%%%%%%% (5) circle %%%
  x=zeros(2,N)
  x[1,]=rand(1,N)
  x[2,]=sign(2*rand(1,N)-1)*sqrt(0.25-(x[1,]-0.5)^2)+0.5
}
if(TYPESET==6){ #%%%%%%%%% (6) sin %%%
  x=zeros(2,N)
  x[1,]=rand(1,N)
  x[2,]=0.3*sin(2*pi*x[1,])+0.05*randn(1,N)
}
#%%%%%%%%%%%%%%%%%%%%% Learning Machine and Record %%%%%%%%%%
recordy=zeros(CYCLE+1,2,K)   #%% Recording
y=x[,floor(runif(K,1,N))]
recordy[1,,]=y
par(mfrow = c(1,1))
par(mar = c(3, 3, 2, 1))
plot(x[1,],x[2,],type="p",pch=20)
lines(y[1,],y[2,],type="p",pch=4,col="red")
getGraphicsEvent(prompt="Waiting for input", onMouseDown=function(buttons,x,y){"ok"})
#%%%%%%%%%%%%%%%%%%%%% Learning Begin %%%%%%%%%%%%%%%
ERR=zeros(CYCLE)
for(cycle in 1:CYCLE){
  c=zeros(N)
  err=0
  for(i in 1:N){
    dist=colSums((y-x[,i])^2)
    k=which.min(dist)
    c[i]=k
    err=err+dist[k]
  }
  ERR[cycle]=err
  ynew=aggregate(t(x),by=list(c),mean)
  y[,ynew[,1]]=t(ynew[,-1])
  recordy[cycle+1,,]=y
}
plot(ERR,type="l",main="sum of squared distance")
getGraphicsEvent(prompt="Waiting for input", onMouseDown=function(buttons,x,y){"ok"})
#%%%%%%%%%%%%%%%%%%%%% Patition Numbers %%%%%%%%%%%%%%%%
table(c)
#%%%%%%%%%%%%%%%%%%%%% Learning End %%%%%%%%%%%%%%%%%%%%
vtess <- deldir(y[1,], y[2,])
plot(x[1,],x[2,],type="p",pch=20,main="K-Mean and Voronoi")
plot(vtess, wlines="tess", wpoints="none", number=FALSE, add=TRUE, lty=1)
lines(y[1,],y[2,],type="p",pch=4,col="red")
