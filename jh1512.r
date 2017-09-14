# 12 Restricted Boltzman machine

set.seed(10)
randn = function(...){array(rnorm(prod(...)),dim=list(...))}
rand = function(...){array(runif(prod(...)),dim=list(...))}
zeros = function(...){array(0,dim=list(...))}
ones = function(...){array(1,dim=list(...))}
mod = function(a,b){a%%b}

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dt <- read.table('char_learn03.txt', sep=" ", header=F, stringsAsFactors=F)
A0 = as.matrix(dt[,-31])
N0=dim(A0)[1]
D=dim(A0)[2]
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%% 5 times Input data are made by adding noises
A=rbind(A0,A0,A0,A0,A0)
N=dim(A)[1]
A=0.8*A+0.1
A=A+0.1*randn(N,D)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%% N = number of input vectors
#%% D = dimension of input
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
YOKO=5      #%% Character size
CHARN=N0    #%% Number of Input Patterns
H=8         #%% hidden variables
ETA=0.01    #%% Coefficient of learning
CYCLE=500   #%% cycle of learning
MCMC=20     #%% MCMC process 1 5 20 50 100
MCMCTEST=50 #%% MCMC for TEST
NSEE=10     #%% Visible units in testy 5 10 15 20 25
NOISETEST=0.1
#%%%%%%%%%%%%%%%%%%% sigmoid function %%%%%%%%%%%%%%%%%%
sigmoid=function(t){1/(1+exp(-t))}
#%%%%%%%%%%%%%%%%%%% Initial parameters %%%%%%%%%%%%%%%%%
w=0.2*randn(H,D)
th1=0.2*randn(H,1)
th2=0.2*randn(D,1)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for(cycle in 1:CYCLE){
  for(i in 1:N){
    S=0.5*ones(H+D,1)
    SUMS=zeros(H+D,1)
    COVS=zeros(H+D,H+D)
    for(mcmc in 1:MCMC){
      S[1:H,1]=floor(sigmoid((w %*% S[(H+1):(H+D),1])+th1)+rand(H,1))
      S[(H+1):(H+D),1]=floor(sigmoid((t(w) %*% S[1:H,1])+th2)+rand(D,1))
      SUMS=SUMS+S
      COVS=COVS+S %*% t(S)
    }
    SUMS=SUMS/MCMC
    COVS=COVS/MCMC
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    hid_given=sigmoid((w %*% A[i,])+th1)
    w=w+ETA*((hid_given %*% A[i,])-COVS[1:H,(H+1):(H+D)])
    th1=th1+ETA*(hid_given-SUMS[1:H,1])
    th2=th2+ETA*(A[i,]-SUMS[(H+1):(H+D),1])
  }
}
#%%%%%%%%%%%%%%%%%% Trained DATA %%%%%%%%%%%%%%%%%%%%%%
xx=zeros(6,5)
par(mfrow = c(3,4))
par(mar = c(3, 3, 2, 1))
for(i in 1:N0){
  for(j in 1:D){
    xx[floor((j-1)/YOKO)+1,mod(j-1,YOKO)+1]=A[i,j]
  }
  image(255*(1-0.5*xx),col=gray((32:0)/32),xaxt="n",yaxt="n",main="Trained DATA")
}
#%%%%%%%%%%%%%%%% TEST DATA %%%%%%%%%%%
TESTA=A0+NOISETEST*randn(N0,D)
for(i in 1:N0){
  for(j in (NSEE+1):D){
    TESTA[i,j]=0.5
  }
  for(j in 1:D){
    xx[floor((j-1)/YOKO)+1,mod(j-1,YOKO)+1]=TESTA[i,j]
  }
  image(255*(1-0.5*xx),col=gray((32:0)/32),xaxt="n",yaxt="n",main="TEST DATA")
}
#%%%%%%%%%%%%%%%%% Results of Test data %%%%%%%%%%%%%%%%%%%%%%%
for(i in 1:N0){
  S=0.5*ones(H+D,1)
  SUMS=zeros(H+D,1)
  for(mcmc in 1:MCMCTEST){
    S[(H+1):(H+NSEE),1]=t(TESTA[i,1:NSEE])
    S[1:H,1]=floor(sigmoid((w %*% S[(H+1):(H+D)])+th1)+rand(H,1))
    S[(H+1):(H+D),1]=floor(sigmoid((t(w) %*% S[1:H])+th2)+rand(D,1))
    SUMS=SUMS+S
  }
  for(j in 1:D){
    xx[floor((j-1)/YOKO)+1,mod(j-1,YOKO)+1]=SUMS[H+j,1]/MCMCTEST
  }
  image(255*(1-0.5*xx),col=gray((32:0)/32),xaxt="n",yaxt="n",main="Results")
}
