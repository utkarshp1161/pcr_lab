import pandas as pd
import numpy as np

df= pd.read_csv('input_utkarsh.csv')


freq=df['f']
eps=df['e1']
epsdp=df['e2']
#eps = 15 ;
#epsdp = 0.6;
#epsr = eps-epsdp;
epsr=eps.subtract(epsdp)


#miu = xlsread(f,'D2:D202');
miu0 = 1.26e-6
miu = 1
miudp = 0
#t = 0.001
t = float(input("enter value of t"))




#%paramater - Obtain x+iy form from r,theta form raw data
#%miu   = xlsread(f,'D2:D439');
#%miudp = xlsread(f,'E2:E439');
#miur  = miu-1i.*miudp;
miur = miu - miudp





#% S-parameters
#%S11 = xlsread(f,'K2:K202');
from decimal import *
getcontext().prec = 20
pi=np.pi
w = 2*pi*freq
eps0 = 8.854187817e-12
miu0 = 1.26e-6
sigma = w*eps0*epsdp;
Z0 = 377
c = 3e8
delta = np.sqrt(2/abs((miu*miu0*w*sigma)))


#%SEa = 8.68.*t./delta;
SEa= 8.68*t*np.sqrt(abs(pi*miu*miu0*freq*sigma))


Zin = Z0*np.sqrt(miur/epsr)*np.tanh((2*pi*freq*t/c)*np.sqrt(miur*epsr))
X = (Zin-Z0)/(Zin+Z0)
#%modX = abs(X)
RLoss = 20*np.log(abs(X))
Z = abs(np.sqrt(miur/epsr)*np.sqrt(miu0/eps0))
#%RLoss = 20.*log(abs(S11));


D = (1.41421356237*pi*freq)/c
alpha = D*np.sqrt((miudp*epsdp-miu*eps)+np.sqrt((miudp*epsdp-miu*eps)**2+(miu*epsdp+miudp*eps)**2))

deltae = np.arctan(epsdp/eps)
deltam = np.arctan(miudp/miu)


K = 4*pi*(np.sqrt(miu*eps))*np.sin((deltae+deltam)/2)/(c*np.cos(deltae)*np.cos(deltam))
M2 = (miu*np.cos(deltae)-eps*np.cos(deltam))**2+np.tan((deltam-deltae)/2)*np.tan((deltam-deltae)/2)*(miu*np.cos(deltae)+eps*np.cos(deltam))**2
M2f = 1/M2 
M = (4*miu*np.cos(deltae)*eps*np.cos(deltam))*M2f 
P = (np.sinh(K*freq*t)*np.sinh(K*freq*t))-M 
EMD = abs(P)



freq=freq.values
EMD=EMD.values
alpha=alpha.values
RLoss=RLoss.values


Final=np.transpose(np.vstack((freq,EMD,alpha,RLoss)))
df=pd.DataFrame(Final,columns=['freq','EMD','alpha','Rloss'])
df.to_csv('sutripto_out.csv',index=False)
