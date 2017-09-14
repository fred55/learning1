# 14 Convolution neural network
#    for time series prediction

set.seed(10)
randn = function(...){array(rnorm(prod(...)),dim=list(...))}
rand = function(...){array(runif(prod(...)),dim=list(...))}
zeros = function(...){array(0,dim=list(...))}
ones = function(...){array(1,dim=list(...))}
mod = function(a,b){a%%b}

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
CONV=0           #%%%% Convolution Network 0 1 2 (0: all)
RIDGE=0          #%%%% Ridge 0 1
LASSO=0          #%%%% Lasso 0 1
#%%%%%%%%%%%%%%%%% Training Cycles %%%%%%%%%%%%%%%%
CYCLE=2000
MODCYC=10
Err1=zeros(CYCLE/MODCYC)
Err2=zeros(CYCLE/MODCYC)
#%%%%%%%%%%%%%%%%%%%%%% DATA READING %%%%%%%%%%%%%%
dt <- read.table('hakusai.txt', sep="\t", header=F, stringsAsFactors=F)
A0 = dt
#%% Hakusai.txt は
#%%「政府統計の総合窓口」のデータを使用しています。
#%%  http://www.e-stat.go.jp/SG1/estat/eStatTopPortal.do
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
T1=dim(A0)[1]
T2=dim(A0)[2]
A=A0[,2]
meanval=mean(A)
maxval=max(abs(A-meanval))
A=0.49*(A-meanval)/maxval+0.5
M=27
N=T1-(M+1)
Ntrain=floor(N/2)
Ntest=N-Ntrain
ydata=A[(1+M):(N+M)]
xdata=zeros(M,N)
for(i in 1:N){
  xdata[,i]=A[i:(i+M-1)]
}
#%%%%%%%%%%%%%% RIDGE, LASSO, Network %%%%%%%%%%%%%
HYPERPARAMETER1=0.000004     #%%%% RIDGE Hyperparameter %%%%%%%%%%
HYPERPARAMETER2=0.000002     #%%%% LASSO Hyperparameter %%%%%%%%%%
if(RIDGE==1){
  fff=function(a){HYPERPARAMETER1*a}
}else if(LASSO==1){
  fff=function(a){HYPERPARAMETER2*sign(a)}
}else{
  fff=function(a){0}
}
#%%%%%%%%%%%%%%%%%%%%%%% Neural Network Calculation %%%%%%%%%%
sigmoid=function(t){1/(1+exp(-t))}
out=function(w,t,h){
  sigmoid(w%*%h+matrix(t,nrow(t),ncol(h)))
}
#%%%%%%%%%%%%%%%% Neural Network Architecture %%%
#%%%% M=H3 -> H2 -> H1 -> H0=D
D=1
H0=1
H1=3
H2=9
H3=M
#%%%%%%%%%%%%%%%%%%%%%%% Training Paramaters %%%%%%%%%%%%%%%%%
ETA0=0.5
ALPHA=0.1
EPSILON=0.01
#%%%%%%%%%%%%%%%%%%% input, hidden, output %%%%%%%%%%%%%%
h0=zeros(H0,1)
h1=zeros(H1,1)
h2=zeros(H2,1)
#%%%%%%%%%%%%%%%%%%%%% Training Initialization %%%%%%%%
w0=0.1*randn(H0,H1)
w1=0.1*randn(H1,H2)
w2=0.1*randn(H2,H3)
th0=0.01*randn(H0,1)
th1=0.01*randn(H1,1)
th2=0.01*randn(H2,1)
#%%%%%%%%%% Initial weight is determined by linear prediction
xtrain=xdata[,1:Ntrain]
ytrain=ydata[1:Ntrain]
xtest=xdata[,(Ntrain+1):N]
ytest=ydata[(Ntrain+1):N]
syx=xtrain %*% ytrain
sxx=xtrain %*% t(xtrain)
wlinear=solve(sxx) %*% syx
Tlinear=sum((ytrain-t(wlinear)%*%xtrain)^2)
Glinear=sum((ytest-t(wlinear)%*%xtest)^2)
sprintf('[Linear Pred]:Training Err=%f, Test Err=%f',Tlinear/Ntest,Glinear/Ntrain)
for(i in 1:H2){
  w2[i,3*i-2]=tanh(5*wlinear[3*i-2,1])
  w2[i,3*i-1]=tanh(5*wlinear[3*i-1,1])
  w2[i,3*i-0]=tanh(5*wlinear[3*i-0,1])
}
#%% Convolution network
#%% CONV==0 --> all weights are used. Not convolution
#%% CONV==1 --> Convoltion (-1,0,1)
#%% CONV==2 --> Convolution (-2,-1,0,1,2)
if(CONV==0){
  wmask1=ones(H1,H2)
  wmask2=ones(H2,H3)
}else{
  wmask1=zeros(H1,H2)
  wmask2=zeros(H2,H3)
  for(i in 1:H1){
    for(j in 1:H2){
      if(abs(3*i-j-1)<CONV+1){
        wmask1[i,j]=1
      }
    }
  }
  for(i in 1:H2){
    for(j in 1:H3){
      if(abs(3*i-j-1)<CONV+1){
        wmask2[i,j]=1
      }
    }
  }
}
#%%%%%%%%%%%%%%%%%%% Accelerator
dw0=zeros(H0,H1)
dw1=zeros(H1,H2)
dw2=zeros(H2,H3)
dth0=zeros(H0,1)
dth1=zeros(H1,1)
dth2=zeros(H2,1)
#%%%%%%%%%%%%%%%%%%% Backpropagation Learning %%%%%%%%%%%%
tstart <- proc.time()
for(cycle in 1:CYCLE){
  ETA=ETA0*sqrt(100/(100+cycle))
  for(i in 1:Ntrain){
    h3=xdata[,i,drop=F]
    t0=ydata[i]
    w2=wmask2*w2
    w1=wmask1*w1
    h2=out(w2,th2,h3)
    h1=out(w1,th1,h2)
    h0=out(w0,th0,h1)
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    delta0=(h0-t0)*(h0*(1-h0)+EPSILON)
    delta1=(t(w0)%*%delta0)*(h1*(1-h1)+EPSILON)
    delta2=(t(w1)%*%delta1)*(h2*(1-h2)+EPSILON)
    #%%%%%%%%%%%%%%%%% gradient %%%%%%%%%%%
    dw0=-ETA*delta0%*%t(h1)+ALPHA*dw0
    dth0=-ETA*delta0+ALPHA*dth0
    dw1=-ETA*delta1%*%t(h2)+ALPHA*dw1
    dth1=-ETA*delta1+ALPHA*dth1
    dw2=-ETA*delta2%*%t(h3)+ALPHA*dw2
    dth2=-ETA*delta2+ALPHA*dth2
    #%%%%%%%%%%%%%%%%%% steepest descent %%%%%%%%%%
    w0=w0+dw0-fff(w0)
    th0=th0+dth0-fff(th0)
    w1=w1+dw1-fff(w1)
    th1=th1+dth1-fff(th1)
    w2=w2+dw2-fff(w2)
    th2=th2+dth2-fff(th2)
    w2=wmask2*w2
    w1=wmask1*w1
  }
  #%%%%%%%%%%%%% Calculation of Training and Generalization Errors %%%%
  if(mod(cycle,MODCYC)==0){
    h3=xdata[,1:Ntrain]
    t0=ydata[1:Ntrain]
    h2=out(w2,th2,h3)
    h1=out(w1,th1,h2)
    h0=out(w0,th0,h1)
    Err1[cycle/MODCYC]=mean((t0-h0)^2)
    h3=xdata[,(Ntrain+1):N]
    t0=ydata[(Ntrain+1):N]
    h2=out(w2,th2,h3)
    h1=out(w1,th1,h2)
    h0=out(w0,th0,h1)
    Err2[cycle/MODCYC]=mean((t0-h0)^2)
  }
}
proc.time() - tstart   # 43sec
#%%%%%%%%%%%%%%%%%%%%%%% Backpropagation End %%%%%%%%%%%%%%%%
par(mfrow = c(2,2))
par(mar = c(3, 3, 2, 1))
plot(A,type="l",main="Hakusai")
#%%%%% Draw Results 1 : Training and Generalization Errors
plot(Err1,type="l",col="blue",xlab="Cycle",ylab="Error",main="Blue: Training Error, Red: Test Error.")
lines(Err2,type="l",col="red")
#%%%%% Draw Results 2 : Training and Unknwon Time Series Predictions
h3=xdata[,1:Ntrain]
h2=out(w2,th2,h3)
h1=out(w1,th1,h2)
h0=out(w0,th0,h1)
true1=ydata[1:Ntrain]
ans1=h0
h3=xdata[,(Ntrain+1):N]
h2=out(w2,th2,h3)
h1=out(w1,th1,h2)
h0=out(w0,th0,h1)
true2=ydata[(Ntrain+1):N]
ans2=h0
plot(true1,type="l",col="blue",ylim=c(0,1),xlab="Cycle",ylab="",main="Trained Data: red:true, blue:predicttion")
lines(ans1[1,],type="l",col="red")
plot(true2,type="l",col="blue",ylim=c(0,1),xlab="Cycle",ylab="",main="Unknown: red:true, blue:predicttion")
lines(ans2[1,],type="l",col="red")
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
