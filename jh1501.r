# 01 Neural Network for Character Recongition

set.seed(10)
randn = function(...){array(rnorm(prod(...)),dim=list(...))}
rand = function(...){array(runif(prod(...)),dim=list(...))}
zeros = function(...){array(0,dim=list(...))}
ones = function(...){array(1,dim=list(...))}
mod = function(a,b){a%%b}

######################################################################
PIX=5
M=PIX*PIX
H=8   ########## 1 --- 10
D=2
#################### Training Data reading ########
dt <- read.table('jh1501_train.txt', sep=" ", header=F, stringsAsFactors=F)
xdata=t(as.matrix(dt[,-26]))
N=ncol(xdata)
ydata=zeros(D,N)
for(i in 1:N){
  if(i<201){
    ydata[1,i]=1
  }else{
    ydata[2,i]=1
  }
}
####################### Test data
dt <- read.table('jh1501_test.txt', sep=" ", header=F, stringsAsFactors=F)
xtest=t(as.matrix(dt[,-26]))
Ntest=ncol(xtest)
ytest=zeros(D,N)
for(i in 1:Ntest){
  if(i<201){
    ytest[1,i]=1
  }else{
    ytest[2,i]=1
  }
}
######################################################################
########### Choose the optimal hyperparameter of Lasso
########### that minimizes the generalization error.
######################################################################
Lasso=0.0003   ### LassoHyperparameter=0.0003
#%%%%%%%%%%%%%%%%% Neural Network Sigmoid Function %%%%%%%%%%%%
sigmoid = function(x){1/(1+exp(-x))}
####################### Training Conditions ######
CYCLE=5000
MODCYC=50
Err1=zeros(CYCLE/MODCYC)
Err2=zeros(CYCLE/MODCYC)
###################### Training Initialization ########
u=0.1*randn(D,H)
w=0.1*randn(H,M)
ph=0.1*randn(D,1)
th=0.1*randn(H,1)
mu=zeros(D,H)
mw=zeros(H,M)
mph=zeros(D,1)
mth=zeros(H,1)
ETA0=0.005
ALPHA=0.0
EPSILON=0.001
#################### Backpropagation Learning ############
tstart <- proc.time()
for(cycle in 1:CYCLE){
  ETA=ETA0*CYCLE/(CYCLE+5*cycle)
  for(i in 1:N){
    ii=(N/2)*mod(i-1,2)+floor((i+1)/2)
    xt=xdata[,ii,drop=F]
    tt=ydata[,ii,drop=F]
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
    mu = du + ALPHA*mu
    mph = dph + ALPHA*mph
    mw = dw + ALPHA*mw
    mth = dth + ALPHA*mth
    #%%%%%%%%%%%%%%%%%% stochastic steepest descent %%%%%%%%%%
    u = u - ETA*(mu +Lasso*sign(u))
    ph=ph - ETA*(mph+Lasso*sign(ph))
    w = w - ETA*(mw +Lasso*sign(w))
    th=th - ETA*(mth+Lasso*sign(th))
  }
  ### Calculation of Training and Generalization Errors ####
  if(mod(cycle,MODCYC)==0){
    pt = w %*% xdata + matrix(th,H,ncol(xdata))
    ot = sigmoid(pt)
    gt = u %*% ot + matrix(ph,D,ncol(ot))
    ft = sigmoid(gt)
    df = ft - ydata
    #training_e = mean(df*df)
    training_e = sum(ifelse(ft<0.5,1,0)*ydata)/N
    Err1[cycle/MODCYC]=training_e
    pt = w %*% xtest + matrix(th,H,ncol(xtest))
    ot = sigmoid(pt)
    gt = u %*% ot + matrix(ph,D,ncol(ot))
    ft = sigmoid(gt)
    df = ft - ytest
    #test_e = mean(df*df)
    test_e = sum(ifelse(ft<0.5,1,0)*ytest)/Ntest
    Err2[cycle/MODCYC]=test_e
  }
}
proc.time() - tstart # 200sec
par(mfrow = c(1,1))
par(mar = c(3, 3, 2, 1))
ymin=min(Err1,Err2,0)
ymax=max(Err1,Err2)
main="Neural Network for Character Recongition"
plot(Err1,type="l",col="blue",xlab="Training Cycle",ylab="",main=main,ylim=c(ymin,ymax))
lines(Err2,type="l",col="red")
legend("topright",col=c("blue","red"),legend=c("Training Error","Test Error"),lwd=1)
getGraphicsEvent(prompt="Waiting for input", onMouseDown=function(buttons,x,y){"ok"})
########################## Trained  ###############################
sprintf('Lasso=%g, ETA0=%g, ALPHA=%g, EPSILON=%g',Lasso,ETA0,ALPHA,EPSILON)
pt = w %*% xdata + matrix(th,H,ncol(xdata))
ot = sigmoid(pt)
gt = u %*% ot + matrix(ph,D,ncol(ot))
ft = sigmoid(gt)
df = ft - ydata
training_e = mean(df*df)
error = sum(ft[1,1:200]<ft[2,1:200])+sum(ft[1,201:400]>ft[2,201:400])
sprintf('Error/TRAINED = %g/%g = %g',error,N,error/N)
########################## Test ###############################
pt = w %*% xtest + matrix(th,H,ncol(xtest))
ot = sigmoid(pt)
gt = u %*% ot + matrix(ph,D,ncol(ot))
ft = sigmoid(gt)
df = ft - ytest
test_e = mean(df*df)
error = sum(ft[1,1:200]<ft[2,1:200])+sum(ft[1,201:400]>ft[2,201:400])
sprintf('Error/TEST = %g/%g = %g',error,Ntest,error/Ntest)
#%%%%%%%%%%%%%%%%%%% Display Training image %%%%%%%%%%%%%%%%%%%%%%%%%%%%
par(mfrow = c(4,5))
par(mar = c(3, 3, 2, 1))
for(i in 1:10){
  mat = matrix(xdata[,i],PIX,PIX,byrow=F)
  mat = mat[,(PIX:1)]
  image(mat,col=gray((32:0)/32),xaxt="n",yaxt="n",main="0")
}
for(i in 1:10){
  mat = matrix(xdata[,200+i],PIX,PIX,byrow=F)
  mat = mat[,(PIX:1)]
  image(mat,col=gray((32:0)/32),xaxt="n",yaxt="n",main="6")
}
