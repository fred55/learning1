# 10 Kohonen's Self Orginizing Map (SOM)
#    One dimension

set.seed(100)
randn = function(...){array(rnorm(prod(...)),dim=list(...))}
rand = function(...){array(runif(prod(...)),dim=list(...))}
zeros = function(...){array(0,dim=list(...))}
ones = function(...){array(1,dim=list(...))}
mod = function(a,b){a%%b}

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N=100       #%% Training samples
K=20        #%% Learning Components
CYCLE=500   #%% Repeated Time
DATACASE=1  #%%%% 1 2 3
epsilon=0.002
#%%%%%%%%%%%%%%%% Training Data %%%%%%%%%%%%%%%
x=zeros(N,2)
if(DATACASE==1){ #%%%%%%%%% (1) quadratic
  x[,1]=4*rand(N)-2
  x[,2]=x[,1]^2+0.5*randn(N)+2
}else if(DATACASE==2){ #%%%%%%%%% (2) circle
  r=rand(N)
  x[,1]=3*cos(2*pi*r)+0.2*randn(N)
  x[,2]=3*sin(2*pi*r)+0.2*randn(N)
}else if(DATACASE==3){ #%%%%%%%%% (3) cross
  for(i in 1:(N/2)){
    x[i,1]=4*(-(N/4)+i)/(N/2)+0.1*randn(1)
    x[i,2]=0.1*randn(1)
  }
  for(i in 1:(N/2)){
    x[i+N/2,1]=0.1*randn(1)
    x[i+N/2,2]=4*(-(N/4)+i)/(N/2)+0.1*randn(1)
  }
}
#%%%%%%%%%%%%%%%%%%% Initialization %%%%%%%%%%%
y=zeros(K,2)
for(k in 1:K){
  y[k,1]=(-k+K/2)/K
  y[k,2]=0.5
}
#%%%%%%%%%%%%%%%%%%% Distance %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
distance=function(x,y){rowSums((matrix(x,nrow(y),ncol(y),byrow=T)-y)^2)}
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
result=list(list(y=y))
DIST=zeros(CYCLE)
for(cycle in 1:CYCLE){
  #%%%%%%% SOM Learning
  for(i in 1:N){
    dd=distance(x[i,],y)
    k=which.min(dd)
    y[k,]=y[k,]+2*epsilon*(x[i,]-y[k,])
    if(k>1){
      y[k-1,]=y[k-1,]+epsilon*(x[i,]-y[k-1,])
    }
    if(k>2){
      y[k-2,]=y[k-2,]+epsilon*(x[i,]-y[k-2,])
    }
    if(k<K){
      y[k+1,]=y[k+1,]+epsilon*(x[i,]-y[k+1,])
    }
    if(k<K-1){
      y[k+2,]=y[k+2,]+epsilon*(x[i,]-y[k+2,])
    }
  }
  #%%%%%%%%% smoothing
  w=cycle/2
  z=zeros(K,2)
  for(k in 2:(K-1)){
    z[k,]=(y[k-1,]+w*y[k,]+y[k+1,])/(w+2)
  }
  z[1,]=(w*y[1,]+y[2,])/(w+1)
  z[K,]=(y[K-1,]+w*y[K,])/(w+1)
  y=z
  #%%%%%%% Total Distance
  total=0
  for(i in 1:N){
    dd=distance(x[i,],y)
    total=total+min(dd)
  }
  DIST[cycle]=total
  result=c(result,list(list(y=y)))
}
#%%%%%%%%%%%%%%%%%%%% Figure %%%%%%%%%%%%%%%%%%%%%%%%
dmin=min(log(DIST))
dmax=max(log(DIST))
h=(dmax-dmin)/10
nos=sapply(0:10,function(i){sum(log(DIST)>=(dmax-i*h))})
par(mfrow = c(3,4))
par(mar = c(3, 3, 2, 1))
plot(DIST,type="l",ylim=c(0,max(DIST)))
abline(h=0,lty=2)
for(cycle in nos){
  yp=result[[cycle]]$y
  xmin=min(x[,1],yp[,1])
  xmax=max(x[,1],yp[,1])
  ymin=min(x[,2],yp[,2])
  ymax=max(x[,2],yp[,2])
  plot(x[,1],x[,2],type="p",pch=20,xlim=c(xmin,xmax),ylim=c(ymin,ymax),main="SOM: Training Process")
  lines(yp[,1],yp[,2],type="b",pch=20,col="red")
}
