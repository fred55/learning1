# 04 Neural Network for Classification

set.seed(10)
randn = function(...){array(rnorm(prod(...)),dim=list(...))}
rand = function(...){array(runif(prod(...)),dim=list(...))}
zeros = function(...){array(0,dim=list(...))}
ones = function(...){array(1,dim=list(...))}
mod = function(a,b){a%%b}

#%%%%%%%%%%%%% True Classfication Rule %%%%%%%%%%%%%%%%%%%%%%%%%%%%
true_rule=function(x1,x2){(x2-x1^3+2*x1)}
true_func=function(x1,x2){0.01+0.98*(sign(true_rule(x1,x2))+1)/2}
#%%%%%%%%%%%%%%%%% Neural Network Sigmoid Function %%%%%%%%%%%%
sigmoid = function(x){1/(1+exp(-x))}
#%%%%%%%%%%%%%%%%%%%%%% Neural Network Architecture %%%%%%%%%%%
D=1          #%% output Units
H=8          #%% hidden Units
M=2          #%% input Units
#%%%%%%%%%%%%%%%%%%%% Generate Training Data %%%%%%%%%%%%%%%%%%%%%
N=50         #%%%% Number of Training samples
xdata=-4*rand(2,N)+2
ydata=true_func(xdata[1,],xdata[2,])
#%%%%%%%%%%%%%%%%%%%% Generate Test Data %%%%%%%%%%%%%%%%%%%%%
xd=seq(from=-2,to=2,by=0.1)
yd=seq(from=-2,to=2,by=0.1)
grid=expand.grid(xd,yd)
xtest=t(grid)
ytest=true_func(xtest[1,],xtest[2,])
#%%%%%%%%%%%%%%%%%%%% Draw Test Samples %%%%%%%%%%%%%%%%%%%%
par(mfrow = c(1,2))
par(mar = c(3, 3, 2, 1))
cols = ifelse(ytest > 0.5, "red", "blue")
plot(xtest[1,],xtest[2,],type="p",col=cols,pch=20,main="Test data: Red:1, Blue:0")
plot(function(x){x^3-2*x}, -2, 2, add=TRUE, lty=2)
#%%%%%%%%%%%%%%%%%%%% Draw Training Samples %%%%%%%%%%%%%%%%%%%%
cols = ifelse(ydata > 0.5, "red", "blue")
plot(xdata[1,],xdata[2,],type="p",col=cols,pch=20,main="Training data: Red:1, Blue:0")
plot(function(x){x^3-2*x}, -2, 2, add=TRUE, lty=2)
getGraphicsEvent(prompt="Waiting for input", onMouseDown=function(buttons,x,y){"ok"})
#%%%%%%%%%%%%%%%%%%%%%% Training Conditions %%%%%%%%%%%%%%%%%
CYCLE=5000   #%% training cycles
MODCYC=50    #%%%%%%%%% 2 5 50
#%%%%%%%%%%%%%%%%%%%%%%% Hyperparameters %%%%%%%%%%%%%
HYPERPARAMETER=0.0000      #%%%% 0.00000  0.00002
fff=function(a){HYPERPARAMETER*a}
#%%%%%%%%%%%%%%%%%%%%%% Training Parameters %%%%%%%%%%%%%%%%%
ETA=0.1      #%% gradient constant
ALPHA=0.3    #%% accelerator
EPSILON=0.01 #%% regularization
#%%%%%%%%%%%%%%%%%%%%% Training Initialization %%%%%%%%
u=0.3*randn(D,H)   #%% weight from hidden to output
w=0.3*randn(H,M)   #%% weight from input to hidden
ph=0.3*randn(D,1)  #%% bias of output
th=0.3*randn(H,1)  #%% bias of hidden
mu=zeros(D,H)      #%% weight from hidden to output
mw=zeros(H,M)      #%% weight from input to hidden
mph=zeros(D,1)     #%% bias of output
mth=zeros(H,1)     #%% bias of hidden
#%%%%%%%%%%%%%%%%%%% Backpropagation Learning %%%%%%%%%%%%
training_err=zeros(CYCLE/MODCYC)
test_err=zeros(CYCLE/MODCYC)
tstart <- proc.time()
for(cycle in 1:CYCLE){
  for(i in 1:N){
    xt=xdata[,i,drop=F];
    tt=ydata[i,drop=F];
    pt = w %*% xt + th
    ot = sigmoid(pt)
    gt = u %*% ot + ph
    ft = sigmoid(gt)
    df = ft - tt
    #%%%%%%%%%%%%%%%%% gradient %%%%%%%%%%%
    dg = df * (ft * (1-ft) + EPSILON)
    du = dg %*% t(ot)
    dph = rowSums(dg)
    do = t(u) %*% dg
    dp = do * (ot * (1-ot) + EPSILON)
    dw = dp %*% t(xt)
    dth = rowSums(dp)
    mu =du +ALPHA*mu
    mph=dph+ALPHA*mph
    mw =dw +ALPHA*mw
    mth=dth+ALPHA*mth
    #%%%%%%%%%%%%%%%%%% stochastic steepest descent %%%%%%%%%%
    u = u - ETA*mu-fff(u)
    ph=ph - ETA*mph
    w = w - ETA*mw-fff(w)
    th=th - ETA*mth
  }
  #%%%%%%%%% Draw Trained Results %%%%%%%%%%%%%%%%
  if(mod(cycle,MODCYC)==0){
    pt = w %*% xdata + matrix(th,H,ncol(xdata))
    ot = sigmoid(pt)
    gt = u %*% ot + matrix(ph,D,ncol(ot))
    ft = sigmoid(gt)
    df = ft - ydata
    training_e = mean(df*df)
    training_err[cycle/MODCYC]=training_e
    pt = w %*% xtest + matrix(th,H,ncol(xtest))
    ot = sigmoid(pt)
    gt = u %*% ot + matrix(ph,D,ncol(ot))
    ft = sigmoid(gt)
    df = ft - ytest
    test_e = mean(df*df)
    test_err[cycle/MODCYC]=test_e
  }
}
proc.time() - tstart # 22sec
#%%%%%%%%%%%%%%%%%%%% Draw Training Samples %%%%%%%%%%%%%%%%%%%%
par(mfrow = c(1,3))
par(mar = c(3, 3, 2, 1))
cols = ifelse(ydata > 0.5, "red", "blue")
plot(xdata[1,],xdata[2,],type="p",col=cols,pch=20,main="Training data: Red:1, Blue:0",xlim=c(-2,2),ylim=c(-2,2))
plot(function(x){x^3-2*x}, -2, 2, add=TRUE, lty=2)
#%%%%%%%%%%%%%%%%%%% Trained Neural Network Output %%%%%%%%
cols = ifelse(ft > 0.5, "red", "blue")
plot(xtest[1,],xtest[2,],type="p",col=cols,pch=20,main="Trained Neural Network Output")
plot(function(x){x^3-2*x}, -2, 2, add=TRUE, lty=2)
#%%%%%%%%%%%%%%%%%%% Training and Test Errors %%%%%%%%%%%%%%
ymin=min(training_err,test_err,0)
ymax=max(training_err,test_err)
plot(training_err,type="l",col="blue",ylim=c(ymin,ymax),
 main="Blue:Training Error, Red:Test Error")
lines(test_err,type="l",col="red")
