# 02 Learning by a Tensor Machine

library(rgl)

set.seed(10)
randn = function(...){array(rnorm(prod(...)),dim=list(...))}
rand = function(...){array(runif(prod(...)),dim=list(...))}
zeros = function(...){array(0,dim=list(...))}
ones = function(...){array(1,dim=list(...))}
mod = function(a,b){a%%b}

#%%%%%%%%%%%%%%%%%%%%% Excersize %%%%%%%%%%%%%%%%%%%%%%%
N=100    #%% N=100, 1000,   Number of Training samples
CASE=1   #%% CASE= 1, 2,     Case of a true function
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
NOISE=0.1  #%% Output noise
Hyperparameter=0.0001
#%%%%%%%%%%%%%%%%%%%% Definition of a True function %%%%%%%%%%%%%%%%
if(CASE==1){
  ftrue=function(xdata,ydata){xdata*ydata}
}
if(CASE==2){
  ftrue=function(xdata,ydata){2*exp(-3*(xdata^2+ydata^2))}
}
#%%%%%%%%%%%%%%%%%%%%%%%%% Figure 1 : True function %%%%%%%%%%%%%%%%%%%%%%
xt=seq(from=-1,to=1,by=0.05)
yt=seq(from=-1,to=1,by=0.05)
ztrue=outer(xt,yt,function(x,y){
  ftrue(x,y)
})
persp3d(xt,yt,ztrue,col="lightblue",front="filled",back="lines",lit=T)
#%%%%%%%%%%%%%%%%%%%% Figure2 : Training Samples %%%%%%%%%%%%%%%%%
xdata=2*rand(N)-1
ydata=2*rand(N)-1
ztrain=ftrue(xdata,ydata)+NOISE*randn(N)
plot3d(xdata,ydata,ztrain,add=TRUE,col="red",type="p")
#%%%%%%%%%%%%%%%%%%%%%%%%%%% Definition of a Tensor Learning Machine  %%%%%%%%%%%%%%
fxy=function(m,x,y){x^(mod(m-1,4)) * y^(floor((m-1)/4))}
fpoly=function(x,y,w){rowSums(sapply(1:16,function(m){w[m]*fxy(m,x,y)}))}
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Learning Mathematics %%%%%%%%%%%
#%% f= sum_k w(k)*e_k(x)
#%% E = sum_i {  sum_k w(k)*e_k(x(i))-y(i)  }^2
#%% dE/dw(j)=sum_i (sum_k w(k)*e_k(x(i))-y(i))*e_j(x(i))=0
#%% sum_k w(k)*{ sum_i e_k(x(i))*e_j(x(i)) } =sum_i y(i)*e_j(x(i))
#%% A(k,j)=sum_i e_k(x(i))*e_j(x(i))
#%% b(j)=sum_i y(i)*e_j(x(i))
#%% w=A^{-1}*b
#%%%%%%%%%%%%%%%%%%%%% Learning Process %%%%%%%%%%%%%%%%%%%%%%%
A=zeros(16,16)
for(j in 1:16){
  for(k in 1:16){
    A[j,k]=sum(fxy(j,xdata,ydata)*fxy(k,xdata,ydata))
  }
}
B=zeros(16,1)
for(j in 1:16){
  B[j,1]=sum(ztrain*fxy(j,xdata,ydata))
}
ww=solve(A+Hyperparameter) %*% B
#%%%%%%%%%%%%%%%%%%% Figure 3: Output of Trained Learning Machine %%%%%
zp=outer(xt,yt,function(x,y){
  fpoly(x,y,ww)
})
persp3d(xt,yt,ztrue,col="lightblue",front="lines",back="lines",lit=F)
persp3d(xt,yt,zp,add=T,col="red",front="lines",back="lines",lit=F)
#%%%%%%%%%%%%%%%%%% Figure 4: Generalization Error %%%%%%%%%%%%%%%%%%%%
zerror=zp-ztrue
persp3d(xt,yt,zerror,col="red",front="lines",back="lines",lit=F)
sprintf('CASE=%g, N=%g',CASE,N)
sprintf('Training Error=%f',mean((fpoly(xdata,ydata,ww)-ztrain)^2))
sprintf('Generalization Error=%f',NOISE^2+mean(zerror^2))
