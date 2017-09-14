# 10 Kohonen's Self Orginizing Map (SOM)
#    自己組織化写像を用いた日本の市区町村の考察

#% 謝辞：独立行政法人統計センターのデータを用いた。
#% http://www.e-stat.go.jp/SG1/estat/eStatTopPortal.do
#% データの著作権は独立行政法人統計センターのページをご覧ください。
#% http://www.e-stat.go.jp/estat/html/spec.html
#% このデータは２０１２年の市区町村の人口である。
#% 欠損データを含む市区町村は（利用者により）除外されています。
#% 全人口,15歳未満,15-64歳,65以上,出生,死亡,転入,転出,昼間人口,結婚,離婚

set.seed(10)
randn = function(...){array(rnorm(prod(...)),dim=list(...))}
rand = function(...){array(runif(prod(...)),dim=list(...))}
zeros = function(...){array(0,dim=list(...))}
ones = function(...){array(1,dim=list(...))}
mod = function(a,b){a%%b}

dt <- read.table('jh1510_2dat2.csv', sep=",", header=T, stringsAsFactors=F)
estat=as.matrix(dt[,1:11])

#%%%%%%%%%%%%%%% データの正規化 %%%%%%%%%%%%%
N=dim(estat)[1]    #%% N 市の数
DIM=dim(estat)[2]  #%% DIM 特徴量の次元
#%%%%%%%%%%%% 人口でないデータを人口比で正規化
estat[,-1]=estat[,-1]/estat[,1]
#%%%%%%%%%%%%  データの次元ごとに平均と標準偏差で正規化
estat=scale(estat)
#%%%%%%%%%%%%%%%%% 自己組織化写像
#%%%%%%%%%%%%%%%%% EPSILON2=0のとき競合学習
CYCLE=500
MODCYC=20
EPSILON1=0.005
EPSILON2=0.003
TATE=4
YOKO=4
#%%%%%%%%%%%%%%%%%%% Distance %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
distance=function(x,y){rowSums((y-matrix(x,nrow(y),ncol(y),byrow=T))^2)}
yxtoi=function(y,x){(y-1)*YOKO+x}
itoy=function(i){floor((i-1)/YOKO)+1}
itox=function(i){mod(i-1,YOKO)+1}
#%%%%%%%%%%%%%%%%%%% Initialization %%%%%%%%%%%
SOM=0.02*randn(TATE*YOKO,DIM)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SOM 学習 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
DIST=zeros(CYCLE/MODCYC)
for(cycle in 1:CYCLE){
  mindistsum=0
  for(i in 1:N){
    #%%%%%% find the nearest reference vector
    dd=distance(estat[i,],SOM)
    mi=which.min(dd)
    my=itoy(mi)
    mx=itox(mi)
    mindistsum=mindistsum+dd[mi]
    #%%%%%%%%%%% update SOM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    SOM[yxtoi(my,mx),]=SOM[yxtoi(my,mx),]+EPSILON1*(estat[i,]-SOM[yxtoi(my,mx),])
    if(my>1){
      SOM[yxtoi(my-1,mx),]=SOM[yxtoi(my-1,mx),]+EPSILON2*(estat[i,]-SOM[yxtoi(my-1,mx),])
    }
    if(my<TATE){
      SOM[yxtoi(my+1,mx),]=SOM[yxtoi(my+1,mx),]+EPSILON2*(estat[i,]-SOM[yxtoi(my+1,mx),])
    }
    if(mx>1){
      SOM[yxtoi(my,mx-1),]=SOM[yxtoi(my,mx-1),]+EPSILON2*(estat[i,]-SOM[yxtoi(my,mx-1),])
    }
    if(mx<YOKO){
      SOM[yxtoi(my,mx+1),]=SOM[yxtoi(my,mx+1),]+EPSILON2*(estat[i,]-SOM[yxtoi(my,mx+1),])
    }
  }
  if(mod(cycle,MODCYC)==0){
    DIST[cycle/MODCYC]=mindistsum
  }
}
par(mfrow = c(1,1))
par(mar = c(3, 3, 2, 1))
plot(DIST,type="l",ylim=c(0,max(DIST)))
abline(h=0,lty=2)
getGraphicsEvent(prompt="Waiting for input", onMouseDown=function(buttons,x,y){"ok"})
#%%%%%%%%%%%%%%%%%%%%%%%%% 代表点に近いデータの個数 %%%%%%%%%%%%%%%%%%%%%%
times=zeros(TATE,YOKO)
for(i in 1:N){
  dd=distance(estat[i,],SOM)
  mi=which.min(dd)
  my=itoy(mi)
  mx=itox(mi)
  times[my,mx]=times[my,mx]+1
}
times
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% グラフ作成 %%%%%%%%%%%%%%%%%%%%%%%%%%%%
par(mfrow = c(TATE,YOKO))
par(mar = c(3, 3, 2, 1))
ymin=min(SOM)
ymax=max(SOM)
for(my in 1:TATE){
  for(mx in 1:YOKO){
    y=SOM[yxtoi(my,mx),]
    barplot(y,ylim=c(ymin,ymax),main=times[my,mx])
  }
}
getGraphicsEvent(prompt="Waiting for input", onMouseDown=function(buttons,x,y){"ok"})
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 具体例 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%% 1全人口,2子供,3労働者,4老人,5出生,6死亡,7転入,8転出,9昼間人口,10結婚,11離婚
que=c(27,70,141,154,171)
labels=dt[que,12]
par(mfcol = c(2,5))
par(mar = c(3, 3, 2, 1))
for(j in 1:5){
  barplot(estat[que[j],],ylim=c(-2,5),xlab="",main=labels[j])
  dd=distance(estat[que[j],],SOM)
  mi=which.min(dd)
  my=itoy(mi)
  mx=itox(mi)
  pos=sprintf('%s(TATE=%g,YOKO=%g)',labels[j],my,mx)
  barplot(SOM[mi,],ylim=c(-2,5),xlab="",main=pos)
}
