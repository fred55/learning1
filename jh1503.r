# 03 Steepest Descent Dynamics

library(rgl)

set.seed(10)
randn = function(...){array(rnorm(prod(...)),dim=list(...))}
rand = function(...){array(runif(prod(...)),dim=list(...))}
zeros = function(...){array(0,dim=list(...))}
ones = function(...){array(1,dim=list(...))}
mod = function(a,b){a%%b}

#%%%%%%%%%%%%%%%%%%%%%%% Constants %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
A = 5.0      #%%
B = 0.02     #%%
C = 2        #%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
my_func = function(x,y){
  z1 = A*sin(x)-y
  z2 = z1*z1 + B*x^4 - C*x^2
  return(z2)
}
my_func_x = function(x,y){
  z1 = A*sin(x)-y
  z1x = A*cos(x)
  z2x = 2*z1*z1x + B*4*x^3 - C*2*x
  return(z2x)
}
my_func_y = function(x,y){
  z1 = A*sin(x)-y
  z1y = -1
  z2y = 2*z1*z1y
  return(z2y)
}
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
xd=seq(from=-9,to=9,by=0.5)
yd=seq(from=-9,to=9,by=0.5)
zd=outer(xd,yd,function(x,y){
  my_func(x,y)
})
persp3d(xd,yd,zd,col="lightblue",front="filled",back="lines",lit=T)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
VECT=0.01
grid=expand.grid(xd,yd)
x1=grid[,1]
y1=grid[,2]
px1 = my_func_x(x1,y1)
py1 = my_func_y(x1,y1)
x1p = x1 + px1*VECT
y1p = y1 + py1*VECT
par(mfrow = c(1,1))
par(mar = c(3, 3, 2, 1))
plot(x1,y1,type="n")
arrows(x1p,y1p,x1,y1,length=0.05)
getGraphicsEvent(prompt="Waiting for input", onMouseDown=function(buttons,x,y){"ok"})
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ETA=0.01          #%%%% homework     0.01   0.1   0.2
ALPHA=0.0         #%%%% homework     0.0    0.3   0.8
LANGEVIN=0.0      #%%%% homework     0.0    0.5   1.0
NNN=LANGEVIN*sqrt(2.0*ETA)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x0 = 1.4          #%% initial point
y0 = -2.5         #%% initial point 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
CYCLE=100
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fdata=zeros(CYCLE+1)
xdata=zeros(CYCLE+1)
ydata=zeros(CYCLE+1)
xp = x0
yp = y0
mx = 0.0
my = 0.0
fdata[1] = my_func(xp,yp)
xdata[1] = xp
ydata[1] = yp
for(cycle in 1:CYCLE){
  dx = my_func_x(xp,yp)
  dy = my_func_y(xp,yp)
  xn = xp - ETA*dx + ALPHA*mx + NNN*randn(1)
  yn = yp - ETA*dy + ALPHA*my + NNN*randn(1)
  mx = xn-xp
  my = yn-yp
  xp = xn
  yp = yn
  fdata[cycle+1] = my_func(xp,yp)
  xdata[cycle+1] = xp
  ydata[cycle+1] = yp
}
par(mfrow = c(2,1))
par(mar = c(3, 3, 2, 1))
plot(x1,y1,type="n")
arrows(x1p,y1p,x1,y1,length=0.05)
lines(xdata,ydata,type="l",col="red")
plot(fdata,type="l",xlab="Cycle",ylab="Evaluation")
