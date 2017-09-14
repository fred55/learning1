# 10 Kohonen's Self Orginizing Map (SOM)
#    ���ȑg�D���ʑ���p�������{�̎s�撬���̍l�@

#% �ӎ��F�Ɨ��s���@�l���v�Z���^�[�̃f�[�^��p�����B
#% http://www.e-stat.go.jp/SG1/estat/eStatTopPortal.do
#% �f�[�^�̒��쌠�͓Ɨ��s���@�l���v�Z���^�[�̃y�[�W���������������B
#% http://www.e-stat.go.jp/estat/html/spec.html
#% ���̃f�[�^�͂Q�O�P�Q�N�̎s�撬���̐l���ł���B
#% �����f�[�^���܂ގs�撬���́i���p�҂ɂ��j���O����Ă��܂��B
#% �S�l��,15�Ζ���,15-64��,65�ȏ�,�o��,���S,�]��,�]�o,���Ԑl��,����,����

set.seed(10)
randn = function(...){array(rnorm(prod(...)),dim=list(...))}
rand = function(...){array(runif(prod(...)),dim=list(...))}
zeros = function(...){array(0,dim=list(...))}
ones = function(...){array(1,dim=list(...))}
mod = function(a,b){a%%b}

dt <- read.table('jh1510_2dat2.csv', sep=",", header=T, stringsAsFactors=F)
estat=as.matrix(dt[,1:11])

#%%%%%%%%%%%%%%% �f�[�^�̐��K�� %%%%%%%%%%%%%
N=dim(estat)[1]    #%% N �s�̐�
DIM=dim(estat)[2]  #%% DIM �����ʂ̎���
#%%%%%%%%%%%% �l���łȂ��f�[�^��l����Ő��K��
estat[,-1]=estat[,-1]/estat[,1]
#%%%%%%%%%%%%  �f�[�^�̎������Ƃɕ��ςƕW���΍��Ő��K��
estat=scale(estat)
#%%%%%%%%%%%%%%%%% ���ȑg�D���ʑ�
#%%%%%%%%%%%%%%%%% EPSILON2=0�̂Ƃ������w�K
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
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SOM �w�K %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
#%%%%%%%%%%%%%%%%%%%%%%%%% ��\�_�ɋ߂��f�[�^�̌� %%%%%%%%%%%%%%%%%%%%%%
times=zeros(TATE,YOKO)
for(i in 1:N){
  dd=distance(estat[i,],SOM)
  mi=which.min(dd)
  my=itoy(mi)
  mx=itox(mi)
  times[my,mx]=times[my,mx]+1
}
times
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% �O���t�쐬 %%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ��̗� %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%% 1�S�l��,2�q��,3�J����,4�V�l,5�o��,6���S,7�]��,8�]�o,9���Ԑl��,10����,11����
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
