# 13 5-layer Neural Network by Pre-Bottle-Neck
#    Firstly, A 5-layer neural network is trained by a bottle-neck method.
#%%%%%%%%%%%%%%% Neural Network Architecture %%%
#%%%%%%%%%%%%%%% (1) M -> H03 -> H02 -> H03 -> M
#%%%%%%%%%%%%%%% (2) M => H03 => H02 -> H01 -> D
#%%%%%%%%%%%%%%%     => is a copy of above ->

set.seed(10)
randn = function(...){array(rnorm(prod(...)),dim=list(...))}
rand = function(...){array(runif(prod(...)),dim=list(...))}
zeros = function(...){array(0,dim=list(...))}
ones = function(...){array(1,dim=list(...))}
mod = function(a,b){a%%b}

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
RIDGE=0                    #%%%% Ridge %%%%%%%%%%%
LASSO=1                    #%%%% Lasso %%%%%%%%%%%
HYPERPARAMETER1=0.00001    #%%%% Hyperparameter %%%%%%%%%%
HYPERPARAMETER2=0.000001   #%%%% Hyperparameter %%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
#%%%%%%%%%%%%%%%%%%%%%%%% Input:M, Output: D %%%%%%%%
PIX=5
M=PIX*PIX
D=2                #%%%% Output units %%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N=2000             #%%%%%%%%%%%%%%%% Training set
Ntest=2000         #%%%%%%%%%%%%%%%% Test set
#%%%%%%%%%%%%%%%%%%% Training Data reading %%%%%%%%
dt <- read.table('char_train.txt', sep=" ", header=F, stringsAsFactors=F)
xdata = t(dt[,-26])
ydata=zeros(D,N)
for(i in 1:N){
  if(i<1001){
    ydata[1,i]=1
  }else{
    ydata[2,i]=1
  }
}
#%%%%%%%%%%%%%%%%%% Test data
dt <- read.table('char_test.txt', sep=" ", header=F, stringsAsFactors=F)
xtest = t(dt[,-26])
ytest=zeros(D,Ntest)
for(i in 1:Ntest){
  if(i<1001){
    ytest[1,i]=1
  }else{
    ytest[2,i]=1
  }
}
#%%%%%%%%%%%%%%%%%%%%%%%%%%%% Training Recorder %%%%%%%%%%%%%
CYCLE=200
MODCYC=5
Err1=zeros(CYCLE/MODCYC)
Err2=zeros(CYCLE/MODCYC)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%% Bottle-Neck Neural Network  %%%
#%%%% M=H4 -> H3 -> H2 -> H3 -> H0=M
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
H0=M
H1=8
H2=6
H3=8
H4=M
#%%%%%%%%%%%%%%%%%%% input, hidden, output %%%%%%%%%%%%%%
h0=zeros(H0,1)
h1=zeros(H1,1)
h2=zeros(H2,1)
h3=zeros(H3,1)
h4=zeros(H4,1)
#%%%%%%%%%%%%%%%%%%%%% Training Initialization %%%%%%%%
w0=0.01*randn(H0,H1)
w1=0.01*randn(H1,H2)
w2=0.01*randn(H2,H3)
w3=0.01*randn(H3,H4)
th0=0.01*randn(H0,1)
th1=0.01*randn(H1,1)
th2=0.01*randn(H2,1)
th3=0.01*randn(H3,1)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dw0=zeros(H0,H1)
dw1=zeros(H1,H2)
dw2=zeros(H2,H3)
dw3=zeros(H3,H4)
dth0=zeros(H0,1)
dth1=zeros(H1,1)
dth2=zeros(H2,1)
dth3=zeros(H3,1)
ETA0=0.8
ALPHA=0.3
EPSILON=0.0001
#%%%%%%%%%%%%%%%%%%% Backpropagation Learning %%%%%%%%%%%%
tstart <- proc.time()
for(cycle in 1:CYCLE){
  ETA=ETA0*CYCLE/(CYCLE+10*cycle)
  for(i in 1:N){
    ii=(N/2)*mod(i-1,2)+floor((i+1)/2)
    h4=xdata[,ii,drop=F]
    t0=xdata[,ii]
    h3=out(w3,th3,h4)
    h2=out(w2,th2,h3)
    h1=out(w1,th1,h2)
    h0=out(w0,th0,h1)
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    delta0=(h0-t0)*(h0*(1-h0)+EPSILON)
    delta1=(t(w0)%*%delta0)*(h1*(1-h1)+EPSILON)
    delta2=(t(w1)%*%delta1)*(h2*(1-h2)+EPSILON)
    delta3=(t(w2)%*%delta2)*(h3*(1-h3)+EPSILON)
    #%%%%%%%%%%%%%%%%% gradient %%%%%%%%%%%
    dw0=-ETA*delta0%*%t(h1)+ALPHA*dw0
    dth0=-ETA*delta0+ALPHA*dth0
    dw1=-ETA*delta1%*%t(h2)+ALPHA*dw1
    dth1=-ETA*delta1+ALPHA*dth1
    dw2=-ETA*delta2%*%t(h3)+ALPHA*dw2
    dth2=-ETA*delta2+ALPHA*dth2
    dw3=-ETA*delta3%*%t(h4)+ALPHA*dw3
    dth3=-ETA*delta3+ALPHA*dth3
    #%%%%%%%%%%%%%%%%%% steepest descent %%%%%%%%%%
    w0=w0+dw0-fff(w0)
    th0=th0+dth0-fff(th0)
    w1=w1+dw1-fff(w1)
    th1=th1+dth1-fff(th1)
    w2=w2+dw2-fff(w2)
    th2=th2+dth2-fff(th2)
    w3=w3+dw3-fff(w3)
    th3=th3+dth3-fff(th3)
  }
  #%%%%%%%%%%%%% Calculation of Training and Generalization Errors %%%%
  if(mod(cycle,MODCYC)==0){
    h4=xdata
    t0=h4
    h3=out(w3,th3,h4)
    h2=out(w2,th2,h3)
    h1=out(w1,th1,h2)
    h0=out(w0,th0,h1)
    Err1[cycle/MODCYC]=mean((t0-h0)^2)
    h4=xtest
    t0=h4
    h3=out(w3,th3,h4)
    h2=out(w2,th2,h3)
    h1=out(w1,th1,h2)
    h0=out(w0,th0,h1)
    Err2[cycle/MODCYC]=mean((t0-h0)^2)
  }
}
proc.time() - tstart   # 89sec
result=list(list(Err1=Err1,Err2=Err2))
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%% Neural Network for Character Recognition %%%
#%%%% M=H4 -> H3 -> H2 -> H1 -> H0=D
H0=D
H1=4
H2=6
H3=8
H4=M
#%%%%%%%%%%%%%%%%%%% input, hidden, output %%%%%%%%%%%%%%
h0=zeros(H0,1)
h1=zeros(H1,1)
h2=zeros(H2,1)
h3=zeros(H3,1)
h4=zeros(H4,1)
#%%%%%%%%%%%%%%%%%%%%% Training Initialization %%%%%%%%
w0=0.01*randn(H0,H1)
w1=0.01*randn(H1,H2)
#%% trained w2 in bottle-neck is used
#%% trained w3 in bottle-neck is used
th0=0.01*randn(D,1)
th1=0.01*randn(H1,1)
#%% trained th2 in bottle-neck is used
#%% trained th3 in bottle-neck is used
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dw0=zeros(H0,H1)
dw1=zeros(H1,H2)
dw2=zeros(H2,H3)
dw3=zeros(H3,H4)
dth0=zeros(H0,1)
dth1=zeros(H1,1)
dth2=zeros(H2,1)
dth3=zeros(H3,1)
ETA0=0.8
ALPHA=0.3
EPSILON=0.0001
#%%%%%%%%%%%%%%%%%%% Backpropagation Learning %%%%%%%%%%%%
tstart <- proc.time()
for(cycle in 1:CYCLE){
  ETA=ETA0*CYCLE/(CYCLE+10*cycle)
  for(i in 1:N){
    ii=(N/2)*mod(i-1,2)+floor((i+1)/2)
    h4=xdata[,ii,drop=F]
    t0=ydata[,ii]
    h3=out(w3,th3,h4)
    h2=out(w2,th2,h3)
    h1=out(w1,th1,h2)
    h0=out(w0,th0,h1)
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    delta0=(h0-t0)*(h0*(1-h0)+EPSILON)
    delta1=(t(w0)%*%delta0)*(h1*(1-h1)+EPSILON)
    delta2=(t(w1)%*%delta1)*(h2*(1-h2)+EPSILON)
    delta3=(t(w2)%*%delta2)*(h3*(1-h3)+EPSILON)
    #%%%%%%%%%%%%%%%%% gradient %%%%%%%%%%%
    dw0=-ETA*delta0%*%t(h1)+ALPHA*dw0
    dth0=-ETA*delta0+ALPHA*dth0
    dw1=-ETA*delta1%*%t(h2)+ALPHA*dw1
    dth1=-ETA*delta1+ALPHA*dth1
    dw2=-ETA*delta2%*%t(h3)+ALPHA*dw2
    dth2=-ETA*delta2+ALPHA*dth2
    dw3=-ETA*delta3%*%t(h4)+ALPHA*dw3
    dth3=-ETA*delta3+ALPHA*dth3
    #%%%%%%%%%%%%%%%%%% steepest descent %%%%%%%%%%
    w0=w0+dw0-fff(w0)
    th0=th0+dth0-fff(th0)
    w1=w1+dw1-fff(w1)
    th1=th1+dth1-fff(th1)
    w2=w2+dw2-fff(w2)
    th2=th2+dth2-fff(th2)
    w3=w3+dw3-fff(w3)
    th3=th3+dth3-fff(th3)
  }
  #%%%%%%%%%%%%% Calculation of Training and Generalization Errors %%%%
  if(mod(cycle,MODCYC)==0){
    h4=xdata
    t0=ydata
    h3=out(w3,th3,h4)
    h2=out(w2,th2,h3)
    h1=out(w1,th1,h2)
    h0=out(w0,th0,h1)
#    Err1[cycle/MODCYC]=(N-sum((h0>0.5)*(t0>0/5)))/N
    Err1[cycle/MODCYC]=mean((t0-h0)^2)
    h4=xtest
    t0=ytest
    h3=out(w3,th3,h4)
    h2=out(w2,th2,h3)
    h1=out(w1,th1,h2)
    h0=out(w0,th0,h1)
#    Err2[cycle/MODCYC]=(Ntest-sum((h0>0.5)*(t0>0/5)))/Ntest
    Err2[cycle/MODCYC]=mean((t0-h0)^2)
  }
}
proc.time() - tstart   # 84sec
result=c(result,list(list(Err1=Err1,Err2=Err2)))
#%%%%%%%%%%%%%%%%%%%%%%%%% Graph Result %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
par(mfrow = c(1,1))
par(mar = c(3, 3, 2, 1))
plot(Err1,type="n",ylim=c(0,0.05),xlab="Cycle",ylab="Error",main="Blue: Training Error, Red: Test Error.")
ltys=2:1
for(i in 1:2){
  rt=result[[i]]
  lines(rt$Err1,type="l",col="blue",lty=ltys[i])
  lines(rt$Err2,type="l",col="red",lty=ltys[i])
}
legend("topright",col="red",lty=ltys,legend=1:2)
#%%%%%%%%%%%%%%%%%%%%%%%%% Trained  Data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
h4=xdata
t0=ydata
h3=out(w3,th3,h4)
h2=out(w2,th2,h3)
h1=out(w1,th1,h2)
h0=out(w0,th0,h1)
counter1=N-sum((h0>0.5)*(t0>0/5))
sprintf('Error/TRAINED = %g/%g = %f',counter1,N,counter1/N)
#%%%%%%%%%%%%%%%%%%%%%%%%% Test Data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
h4=xtest
t0=ytest
h3=out(w3,th3,h4)
h2=out(w2,th2,h3)
h1=out(w1,th1,h2)
h0=out(w0,th0,h1)
counter2=Ntest-sum((h0>0.5)*(t0>0/5))
sprintf('Error/TEST = %g/%g = %f',counter2,Ntest,counter2/Ntest)
