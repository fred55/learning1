# 15 Model selection

#%% 政府統計の総合窓口（ｅ－Ｓｔａｔ）のデータを使わせて頂きました。
#%% 「ご利用にあたって」のページをお読みの上で著作権に十分に注意してください。
#%% http://www.e-stat.go.jp/SG1/estat/eStatTopPortal.do

set.seed(10)
randn = function(...){array(rnorm(prod(...)),dim=list(...))}
rand = function(...){array(runif(prod(...)),dim=list(...))}
zeros = function(...){array(0,dim=list(...))}
ones = function(...){array(1,dim=list(...))}
mod = function(a,b){a%%b}

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dt <- read.table('hakusai_negi.txt', sep="\t", header=F, stringsAsFactors=F)
data=dt
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
NUM=dim(data)[1]    #%% 年月数
D=5                 #%% 比較するモデルの個数
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N=60                #%% 20 25 30 60 120
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X=data[1:N,2]       #%% 白菜の値段
Y=data[1:N,3]       #%% ねぎの値段
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X0=seq(from=(min(X)-5),to=(max(X)+5),by=1)
xmean=mean(X)
xscale=max(abs(X-xmean))
xt=(X-xmean)/xscale
xt0=(X0-xmean)/xscale
xa=t(sapply(1:D,function(d){xt^(d-1)}))
xa0=t(sapply(1:D,function(d){xt0^(d-1)}))
cols=rainbow(D)
par(mfrow = c(1,1))
par(mar = c(3, 3, 2, 1))
plot(X,Y,type="p",pch=20,xlab="Hakusai",ylab="Negi",main="")
result=t(sapply(1:D,function(d){
  ar=xa[1:d,,drop=F]
  xx=ar%*%t(ar)
  xy=ar%*%Y
  b=solve(xx)%*%xy
  Y0=t(b)%*%xa0[1:d,]
  lines(X0,Y0,type="l",col=cols[d])
  YY=t(b)%*%xa[1:d,]
  rss=sum((Y-YY)^2)
  aic=log(rss/N)*N+N+N*log(2*pi)+2*(d+1)
  bic=log(rss/N)*N+N+N*log(2*pi)+log(N)*(d+1)
  c(d=d,rss=rss,aic=aic,bic=bic)
}))
legend("topleft",col=cols,lwd=1,legend=1:D)
result
#######
d=3
x1=xt
x2=xt^2
rt=lm(Y~x1+x2)
AIC(rt)
BIC(rt)
e2=sum((predict(rt)-Y)^2)
log(e2/N)*N+N+N*log(2*pi)+2*(d+1)
log(e2/N)*N+N+N*log(2*pi)+log(N)*(d+1)
