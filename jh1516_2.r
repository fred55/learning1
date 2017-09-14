# 16 Variational Bayes

set.seed(10)
randn = function(...){array(rnorm(prod(...)),dim=list(...))}
rand = function(...){array(runif(prod(...)),dim=list(...))}
zeros = function(...){array(0,dim=list(...))}
ones = function(...){array(1,dim=list(...))}
mod = function(a,b){a%%b}

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
K=9              #%% Components of learning clusters
D=2              #%% Dimension
a0=1
b0=0.1
u0=10
w0=1
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N=360            #%% Number of samples
CYCLE=1000
MODCYC=20
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
gauss2d=function(x1,x2,b1,b2,s2){exp(-((x1-b1)^2+(x2-b2)^2)/(2*s2))/(2*pi*s2)}
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% make samples %%%%%%%%%%%%%%%%%%%%%%%%%%%
mk0=zeros(2,9)
for(j in 1:9){
  mk0[1,j]=floor((j-1)/3)
  mk0[2,j]=mod(j-1,3)
}
xdata=zeros(2,N)
for(i in 1:N){
  xdata[,i]=0.3*randn(2,1)+mk0[,mod(i-1,9)+1]
}
#xdata=t(scale(faithful))
#N=ncol(xdata)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%% Initialize
m0=rowMeans(xdata)
result=list()
tstart <- proc.time()
for(try in 1:11){
  #rki=1/K+0.01*randn(K,N)
  rki=rand(K,N)/K*2
  nk=rowSums(rki)
  xk=xdata%*%t(rki)
  xk=xk/matrix(nk,D,K,byrow=T)
  ak=a0+nk
  bk=b0+nk
  uk=u0+D/2*nk
  mk=xdata%*%t(rki)+b0*matrix(m0,D,K)
  mk=mk/matrix(bk,D,K,byrow=T)
  wk=zeros(K)
  for(k in 1:K){
    w1=(xdata[1,]-xk[1,k])^2+(xdata[2,]-xk[2,k])^2
    w2=sum(rki[k,]*w1)
    w3=b0*nk[k]/(b0+nk[k])*sum((xk[,k]-m0)^2)
    wk[k]=1/(1/w0+w2/2+w3/2)
  }
  lk=zeros(CYCLE/MODCYC)
  #%%%%%%%%% Recursive VB Start
  for(cycle in 1:CYCLE){
    for(i in 1:N){
      w4=mk-xdata[,i]
      rho=digamma(ak)-digamma(sum(ak))+D/2*(digamma(uk)+log(wk))-D/2/bk-uk*wk/2*colSums(w4^2)
      rki[,i]=exp(rho-max(rho))/sum(exp(rho-max(rho)))
    }
    nk=rowSums(rki)
    xk=xdata%*%t(rki)
    xk=xk/matrix(nk,D,K,byrow=T)
    ak=a0+nk
    bk=b0+nk
    uk=u0+D/2*nk
    mk=xdata%*%t(rki)+b0*matrix(m0,D,K)
    mk=mk/matrix(bk,D,K,byrow=T)
    for(k in 1:K){
      w1=colSums((xdata-matrix(xk[,k],D,N))^2)
      w2=sum(rki[k,]*w1)
      w3=b0*nk[k]/(b0+nk[k])*sum((xk[,k]-m0)^2)
      wk[k]=1/(1/w0+w2/2+w3/2)
    }
    if(mod(cycle,MODCYC)==0){
      loglike=lgamma(K*a0)-K*lgamma(a0)-lgamma(sum(ak))+sum(lgamma(ak))
      loglike=loglike-sum(rki*log(rki+0.001))
      loglike=loglike-K*lgamma(u0)-K*u0*log(w0)
      loglike=loglike+sum(lgamma(uk))+sum(uk*log(wk))
      loglike=loglike+K*D/2*log(b0)-D/2*sum(log(bk))
      loglike=loglike-N*D/2*log(2*pi)
      lk[cycle/MODCYC]=loglike
    }
  }
  result=c(result,list(list(nk=nk,ak=ak,bk=bk,mk=mk,uk=uk,wk=wk,loglike=lk)))
}
proc.time() - tstart   # 120sec
#%%%%%%%%%%%%%%%%%%%%%% plot results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
par(mfrow = c(3,4))
par(mar = c(3, 3, 2, 1))
plot(xdata[1,],xdata[2,],type="p",pch=20)
lines(mk0[1,],mk0[2,],type="p",col="red",pch=8)
xmin=min(xdata[1,])
xmax=max(xdata[1,])
ymin=min(xdata[2,])
ymax=max(xdata[2,])
xd=seq(from=xmin,to=xmax,by=(xmax-xmin)/20)
yd=seq(from=ymin,to=ymax,by=(ymax-ymin)/20)
for(rt in result){
  aa=rt$ak/sum(rt$ak)
  s2=1/(rt$uk*rt$wk)
  zd=outer(xd,yd,function(x,y){
    z=0
    for(k in 1:K){
      z=z+aa[k]*gauss2d(x,y,rt$mk[1,k],rt$mk[2,k],s2[k])
    }
    return(z)
  })
  id=which(aa>1/K/2)
  contour(xd,yd,zd,main=sprintf("%d/%.3f",length(id),rt$loglike[CYCLE/MODCYC]))
  lines(rt$mk[1,id],rt$mk[2,id],type="p",col="red",pch=8)
}
getGraphicsEvent(prompt="Waiting for input", onMouseDown=function(buttons,x,y){"ok"})
#%%%%%%%%%%%%%%%%%%%%%% plot results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
plot(xdata[1,],xdata[2,],type="p",pch=20)
lines(mk0[1,],mk0[2,],type="p",col="red",pch=8)
for(rt in result){
  plot(rt$loglike,type="l",xlab="Cycle",main=rt$loglike[CYCLE/MODCYC])
}
