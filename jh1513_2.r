# 13 5-layer neural network by Sequential Training
#    3-layer ==> 4-layer ==> 5-layer
#%%%%%%%%%%%%%%% Neural Network Architecture %%%
#%%%%%%%%%%%%%%% (1) M -> H03 -> D
#%%%%%%%%%%%%%%% (2) M => H03 -> H02 -> D
#%%%%%%%%%%%%%%% (3) M => H03 => H02 -> H01 -> D
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
HYPERPARAMETER2=0.000002   #%%%% Hyperparameter %%%%%%%%%%
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
#%%%%%%%%%%%%%%%% Final Neural NetworkX Architecture %%%
#%%%%%%%%%%%%%%%% M=H4 -> H3 -> H2 -> H1 -> H0=D
H0=D
H01=4
H02=6
H03=8
H4=M
#%%%%%%%%%%%%%%%%%%%%%%%%%%% 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%% M= H2 -> H1 -> H0=D
#%%%%%%%%%%%%%%%%%%%%%%% Hyperparameters %%%%%%%%%%%%%%%%%%%%%
ETA0=0.8
ALPHA=0.3
EPSILON=0.0001
#%%%%%%%%%%%%%%%% Neural Network for Character Recognition %%%
H0=D
H1=H03
H2=M
H3=0
H4=0
#%%%%%%%%%%%%%%%%%%% input, hidden, output %%%%%%%%%%%%%%
h0=zeros(H0,1)
h1=zeros(H1,1)
h2=zeros(H2,1)
#%%%%%%%%%%%%%%%%%%%%% Training Initialization %%%%%%%%
w0=0.1*randn(H0,H1)
th0=0.1*randn(H0,1)
w1=0.1*randn(H1,H2)
th1=0.1*randn(H1,1)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dw0=zeros(H0,H1)
dth0=zeros(H0,1)
dw1=zeros(H1,H2)
dth1=zeros(H1,1)
#%%%%%%%%%%%%%%%%%%% Backpropagation Learning %%%%%%%%%%%%
tstart <- proc.time()
for(cycle in 1:CYCLE){
  ETA=ETA0*CYCLE/(CYCLE+10*cycle)
  for(i in 1:N){
    ii=(N/2)*mod(i-1,2)+floor((i+1)/2)
    h2=xdata[,ii,drop=F]
    t0=ydata[,ii]
    h1=out(w1,th1,h2)
    h0=out(w0,th0,h1)
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    delta0=(h0-t0)*(h0*(1-h0)+EPSILON)
    delta1=(t(w0)%*%delta0)*(h1*(1-h1)+EPSILON)
    #%%%%%%%%%%%%%%%%% gradient %%%%%%%%%%%
    dw0=-ETA*delta0%*%t(h1)+ALPHA*dw0
    dth0=-ETA*delta0+ALPHA*dth0
    dw1=-ETA*delta1%*%t(h2)+ALPHA*dw1
    dth1=-ETA*delta1+ALPHA*dth1
    #%%%%%%%%%%%%%%%%%% steepest descent %%%%%%%%%%
    w0=w0+dw0-fff(w0)
    th0=th0+dth0-fff(th0)
    w1=w1+dw1-fff(w1)
    th1=th1+dth1-fff(th1)
  }
  #%%%%%%%%%%%%% Calculation of Training and Generalization Errors %%%%
  if(mod(cycle,MODCYC)==0){
    h2=xdata
    t0=ydata
    h1=out(w1,th1,h2)
    h0=out(w0,th0,h1)
    Err1[cycle/MODCYC]=(N-sum((h0>0.5)*(t0>0/5)))/N
#    Err1[cycle/MODCYC]=mean((t0-h0)^2)
    h2=xtest
    t0=ytest
    h1=out(w1,th1,h2)
    h0=out(w0,th0,h1)
    Err2[cycle/MODCYC]=(Ntest-sum((h0>0.5)*(t0>0/5)))/Ntest
#    Err2[cycle/MODCYC]=mean((t0-h0)^2)
  }
}
proc.time() - tstart   # 37sec
result=list(list(Err1=Err1,Err2=Err2))
#%%%%%%%%%%%%%%%%%%%%%%%%%% 2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%% M=H3 -> H2 -> H1 -> H0=D
#%%%%%%%%%%%%%%%%%%%%%%% Hyperparameters %%%%%%%%%%%%%%%%%%%%%
ETA0=0.6
ALPHA=0.2
EPSILON=0.0001
#%%%%%%%%%%%%%%%% Neural Network for Character Recognition %%%
H0=D
H1=H02
H2=H03
H3=M
H4=0
#%%%%%%%%%%%%%%%%%%% input, hidden, output %%%%%%%%%%%%%%
h0=zeros(H0,1)
h1=zeros(H1,1)
h2=zeros(H2,1)
h3=zeros(H3,1)
#%%%%%%%%%%%%%%%%%%%%% Training Initialization %%%%%%%%
w2=w1
th2=th1
w1=0.1*randn(H1,H2)
th1=0.1*randn(H1,1)
w0=0.1*randn(H0,H1)
th0=0.1*randn(H0,1)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dw0=zeros(H0,H1)
dth0=zeros(H0,1)
dw1=zeros(H1,H2)
dth1=zeros(H1,1)
dw2=zeros(H2,H3)
dth2=zeros(H2,1)
#%%%%%%%%%%%%%%%%%%% Backpropagation Learning %%%%%%%%%%%%
tstart <- proc.time()
for(cycle in 1:CYCLE){
  ETA=ETA0*CYCLE/(CYCLE+10*cycle)
  for(i in 1:N){
    ii=(N/2)*mod(i-1,2)+floor((i+1)/2)
    h3=xdata[,ii,drop=F]
    t0=ydata[,ii]
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
  }
  #%%%%%%%%%%%%% Calculation of Training and Generalization Errors %%%%
  if(mod(cycle,MODCYC)==0){
    h3=xdata
    t0=ydata
    h2=out(w2,th2,h3)
    h1=out(w1,th1,h2)
    h0=out(w0,th0,h1)
    Err1[cycle/MODCYC]=(N-sum((h0>0.5)*(t0>0/5)))/N
#    Err1[cycle/MODCYC]=mean((t0-h0)^2)
    h3=xtest
    t0=ytest
    h2=out(w2,th2,h3)
    h1=out(w1,th1,h2)
    h0=out(w0,th0,h1)
    Err2[cycle/MODCYC]=(Ntest-sum((h0>0.5)*(t0>0/5)))/Ntest
#    Err2[cycle/MODCYC]=mean((t0-h0)^2)
  }
}
proc.time() - tstart   # 58sec
result=c(result,list(list(Err1=Err1,Err2=Err2)))
#%%%%%%%%%%%%%%%%%%%%%%%%%% 3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%% M=H4 -> H3 -> H2 -> H1 -> H0=D
#%%%%%%%%%%%%%%%%%%%%%%% Hyperparameters %%%%%%%%%%%%%%%%%%%%%
ETA0=0.3
ALPHA=0.05
EPSILON=0.0001
#%%%%%%%%%%%%%%%% Neural Network for Character Recognition %%%
H0=D
H1=H01
H2=H02
H3=H03
H4=M
#%%%%%%%%%%%%%%%%%%% input, hidden, output %%%%%%%%%%%%%%
h0=zeros(H0,1)
h1=zeros(H1,1)
h2=zeros(H2,1)
h3=zeros(H3,1)
h4=zeros(H4,1)
#%%%%%%%%%%%%%%%%%%%%% Training Initialization %%%%%%%%
w3=w2
th3=th2
w2=w1
th2=th1
w1=0.1*randn(H1,H2)
th1=0.1*randn(H1,1)
w0=0.1*randn(H0,H1)
th0=0.1*randn(H0,1)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dw0=zeros(H0,H1)
dth0=zeros(H0,1)
dw1=zeros(H1,H2)
dth1=zeros(H1,1)
dw2=zeros(H2,H3)
dth2=zeros(H2,1)
dw3=zeros(H3,H4)
dth3=zeros(H3,1)
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
    Err1[cycle/MODCYC]=(N-sum((h0>0.5)*(t0>0/5)))/N
#    Err1[cycle/MODCYC]=mean((t0-h0)^2)
    h4=xtest
    t0=ytest
    h3=out(w3,th3,h4)
    h2=out(w2,th2,h3)
    h1=out(w1,th1,h2)
    h0=out(w0,th0,h1)
    Err2[cycle/MODCYC]=(Ntest-sum((h0>0.5)*(t0>0/5)))/Ntest
#    Err2[cycle/MODCYC]=mean((t0-h0)^2)
  }
}
proc.time() - tstart   # 78sec
result=c(result,list(list(Err1=Err1,Err2=Err2)))
#%%%%%%%%%%%%%%%%%%%%%%%%% Graph Result %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
par(mfrow = c(1,1))
par(mar = c(3, 3, 2, 1))
plot(Err1,type="n",ylim=c(0,0.05),xlab="Cycle",ylab="Error",main="Blue: Training Error, Red: Test Error.")
ltys=3:1
for(i in 1:3){
  rt=result[[i]]
  lines(rt$Err1,type="l",col="blue",lty=ltys[i])
  lines(rt$Err2,type="l",col="red",lty=ltys[i])
}
legend("topright",col="red",lty=ltys,legend=1:3)
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
