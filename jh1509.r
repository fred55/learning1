# 09 Image compression by K-Means

library(jpeg)
library(rgl)

set.seed(10)
randn = function(...){array(rnorm(prod(...)),dim=list(...))}
rand = function(...){array(runif(prod(...)),dim=list(...))}
zeros = function(...){array(0,dim=list(...))}
ones = function(...){array(1,dim=list(...))}
mod = function(a,b){a%%b}

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
K=8          #%% K-means Components K=4, 8, 12, 16, 20
CYCLE=50     #%% Repeated Time
REDUCE=3     #%% Reduce Image
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
A <- readJPEG("suzukake.jpg")
TATE=dim(A)[1]
YOKO=dim(A)[2]
IRO=dim(A)[3]
par(mfrow = c(1,1))
par(mar = c(3, 3, 2, 1))
plot(0,0,xlim=c(1,YOKO),ylim=c(1,TATE),type="n",asp=1,xlab="",ylab="")
rasterImage(A,1,1,YOKO,TATE)
getGraphicsEvent(prompt="Waiting for input", onMouseDown=function(buttons,x,y){"ok"})
TT=floor(TATE/REDUCE)
YY=floor(YOKO/REDUCE)
sprintf('pixels reduced: (%g,%g)-->(%g,%g)',TATE,YOKO,TT,YY)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Draw Image %%%%%%%%%%%%%%%%%%%
B=zeros(TT,YY,IRO)
for(i in 1:TT){
 for(j in 1:YY){
   B[i,j,]=A[REDUCE*i,REDUCE*j,]
 }
}
x=zeros(IRO,TT*YY)
for(i in 1:TT){
 for(j in 1:YY){
   x[,YY*(i-1)+j]=B[i,j,]
 }
}
par(mfrow = c(1,2))
par(mar = c(3, 3, 2, 1))
plot(0,0,xlim=c(1,YOKO),ylim=c(1,TATE),type="n",asp=1,xlab="",ylab="")
rasterImage(A,1,1,YOKO,TATE)
plot(0,0,xlim=c(1,YY),ylim=c(1,TT),type="n",asp=1,xlab="",ylab="")
rasterImage(B,1,1,YY,TT)
getGraphicsEvent(prompt="Waiting for input", onMouseDown=function(buttons,x,y){"ok"})
#%%%%%%%%%%%%%%% K Means %%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%% Learning Machine and Record %%%%%%%%%%
recordy=zeros(CYCLE+1,IRO,K)
y=x[,floor(runif(K,1,TT*YY))]
recordy[1,,]=y
#%%%%%%%%%%%%%%%%%%%%% Learning Begin %%%%%%%%%%%%%%%
n=TT*YY        #%% Training samples
mk=zeros(n)
ERR=zeros(CYCLE)
for(cycle in 1:CYCLE){
  err=0
  for(i in 1:n){
    dist=colSums((y-x[,i])^2)
    k=which.min(dist)
    mk[i]=k
    err=err+dist[k]
  }
  ERR[cycle]=err
  ynew=aggregate(t(x),by=list(mk),mean)
  y[,ynew[,1]]=t(ynew[,-1])
  recordy[cycle+1,,]=y
}
#%%%%%%%%%%%%%%%%%%%%% Patition Numbers %%%%%%%%%%%%%%%%
par(mfrow = c(1,2))
par(mar = c(3, 3, 2, 1))
plot(ERR,type="l",main="sum of squared distance")
table(mk)
barplot(table(mk))
getGraphicsEvent(prompt="Waiting for input", onMouseDown=function(buttons,x,y){"ok"})
#%%%%%%%%%%%%%%%%%%%%% Learning End %%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Region judge
C=zeros(TT,YY,IRO)
for(i in 1:TT){
  for(j in 1:YY){
    C[i,j,]=y[,mk[YY*(i-1)+j]]
  }
}
par(mfrow = c(2,2))
par(mar = c(3, 3, 2, 1))
plot(0,0,xlim=c(1,YOKO),ylim=c(1,TATE),type="n",asp=1,xlab="",ylab="")
rasterImage(A,1,1,YOKO,TATE)
plot(0,0,xlim=c(1,YY),ylim=c(1,TT),type="n",asp=1,xlab="",ylab="")
rasterImage(B,1,1,YY,TT)
plot(0,0,xlim=c(1,YY),ylim=c(1,TT),type="n",asp=1,xlab="",ylab="")
rasterImage(C,1,1,YY,TT)
#%%%%%%%%%%%%%%%%%%%%% Draw graph %%%%%%%%%%%%%%%%%%%
plot3d(x[1,],x[2,],x[3,],xlab="RED",ylab="GREEN",zlab="BLUE",col=gray(0.5))
plot3d(y[1,],y[2,],y[3,],xlab="RED",ylab="GREEN",zlab="BLUE",col="red",size=10,add=T)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Rate=(TT*YY*log2(K)+K*log2(3))/(TT*YY*3*8)
sprintf('Error=%f, Compression Rate=%f',ERR[CYCLE],Rate)
