# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 18:09:56 2017

@author: NAZMUL
"""
from function import  step, lrange
import numpy as np
import mpmath as mp
import scipy
import scipy.stats as sp
import matplotlib.pyplot as plt
import subprocess
import cmath as cm
#import scipy.fftpack as sf

#x1 is a two dimensional list with one row as its value and
#other row as its domain point n of that value i.e. x1=[x,n]
def dtft(s,n,F):
    
    #x=x1[0]
    #j=cm.sqrt(-1)
    #n=x1[1]
    X=[]
    Z=[]
    
   # w=np.linspace(-np.pi,np.pi,N) # F
    for i in range(0,len(F)):
        F_tmp=F[i]
        X_tmp=0
        for k in range(0,len(F)):
            X_tmp+=(s[k]*np.exp(-2*np.pi*n[k]*F_tmp*1j))

       
        X.append(abs(X_tmp))
        Z.append(np.angle(X_tmp))
    

    return X, Z
#x=[1/2,1/2]

#n=[0,1]
#F = np.linspace(-np.pi,np.pi,100)
F = lrange(-1/2,1/2,1/0.01)
n = np.linspace (-11,11,len(F))
s = (0.6)**(abs(n))*(step(n+10)-step(n-11))
#x1=[s,n]
[X, Z] = dtft(s,n,F)

plt.subplot(311)
plt.plot(n,s)
plt.subplot(312)
plt.plot(F,X)
plt.subplot(313)
plt.plot(F,Z)
plt.show()