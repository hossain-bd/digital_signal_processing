#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 18:39:37 2017

@author: Nazmul
"""

from numpy import vdot, pi, exp, flatnonzero, arange, zeros, angle, abs, \
                    linspace, convolve, flipud, sinc, ones, pad, transpose
from scipy import signal
from matplotlib.pyplot import figure, legend, xlabel, ylabel, xlim, ylim, \
                    title, suptitle, subplots_adjust, setp, show, stem, grid, plot, axis, subplot
#from math import exp


def delta(n,a):
    e = zeros(len(n))
    d = flatnonzero(n == 0)
    e[d+a]=1
    return e
    
def step(t):
    r = 1.0*(t >= 0.0)
    return r
    
    
def rect(t, W=1.0):
    re = 1.0*(abs(t) <=W/2)
    return re
    
    
def SigAdd(s1,n1,s2,n2):  
    """ s1, n1, s2, n2 """
    
    n_min = min(n1[0], n2[0]) # min index for n
    n_max = max(n1[-1], n2[-1]) # max index for n
    n = arange(n_min, n_max+1) # total index for n  
    y = zeros(len(n)) # intialization 
    #y2 = y1
    i = n1[0] - n[0] # starting index for s1 in y1
    y[i:i+len(s1)] = s1 # copy s1 in y1
   
    j = n2[0] - n[0] # starting index for s2 in y2
    y[j:j+len(s2)] += s2 # copy s2 in y2
    #y= y1 + y2
    return y, n
 
def lrange(start, stop, rate=1, endpoint=True, dtype=None):
    """
    Return evenly spaced values over a specified interval with specified rate.
    Use linspace with arange-like parameters to generate the desired result

    Parameters
    ----------
    start : number
        Start of interval.  The interval includes this value.
    stop : number
        End of interval.
    rate : number, optional
        Sampling rate, i.e. inverse of the spacing between values. The default
        sampling rate is 1.
    endpoint : bool, optional
        If True, `stop` is the last sample. Otherwise, it is not included.
        Default is True.
    dtype : dtype, optional
        The type of the output array.  If `dtype` is not given, infer the data
        type from the other input arguments.

    Example
    -------
    >>> t=lrange(0, 1, 10)
        [ 0.   0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9  1. ]
    >>> t=lrange(0, 1, 10, endpoint=False)
        [ 0.   0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9]
    """
    if rate == 1:
        return arange(start, stop+1*endpoint, dtype=int)
    else:
        return linspace(start, stop, round((stop-start)*rate)+1*endpoint,
                           endpoint=endpoint, dtype=dtype)    
 
def conv(*args):
    """
    Convolution of continuos-time (CT) or discrete-time (DT) signals

    Parameters
    ----------

    ts : 1-D array, optional
        Support of input signal
    s : 1-D array,
        Input signal
    th : 1-D array, optional
        Support of second signal,
    h : 1-D array,
        Second signal

    Returns
    -------
    tg : 1-D array, optional
        Support of output signal
    g : 1-D array
        Output signal
    """
    nargs = len(args)
    if nargs == 4:
        ts = args[0]
        s = args[1]
        th = args[2]
        h = args[3]
        Ts = ts[1] - ts[0]
        Th = th[1] - th[0]
        if abs(Ts-1) < 1e-4 and abs(Th-1) < 1e-4:
            tg = lrange(ts[0]+th[0], ts[-1]+th[-1])
            g = convolve(s, h)
        else:
            if abs(Ts-Th) > 1e-4:
                print("dsppy.conv: error: CT signals with different rates:" +
                      " Sampling periods: T_s = %f != T_h=%f!" % (Ts, Th))
                return None
            tg = ts[0]+th[0] + arange(len(ts)+len(th)-1)*Ts
            g = convolve(s, h)*Ts
        return tg, g
    elif nargs == 2:
        s = args[0]
        h = args[1]
        tg = None
        g = convolve(s, h)
        return g
    else:
        print(nargs)
        print("dsppy.conv: wrong number of parameters: %d." %(nargs))
        print("Shold be either 2 or 4!")
        return None    
        
def sigshift(x, m, n0): 
      """implements y(n) = x(n-n0)"""
      """ [y,n] = sigshift(x,m,n0)"""
      ny = m+n0
      y = x
      return y,ny      
      
def sigfold(x,n):
    y = flipud(x)
    n = -flipud(n)
    return y,n
    
    
def corr(*args):
    """
    Crosscorrelation of continuous-time (CT) or discrete-time (DT) signals

    Parameters
    ----------

    ts : 1-D array, optional
        Support of input signal
    s : 1-D array,
        Input signal
    th : 1-D array, optional
        Support of second signal,
    h : 1-D array,
        Second signal

    Returns
    -------
    tg : 1-D array, optional
        Support of output correlation function
    g : 1-D array
        Output correlation function
    """
    nargs = len(args)
    if nargs == 4:
        ts = args[0]
        s = args[1]
        th = args[2]
        h = args[3]
        Ts = ts[1] - ts[0]
        Th = th[1] - th[0]
        if abs(Ts-1) < 1e-4 and abs(Th-1) < 1e-4:
            tg = lrange(ts[0]-th[-1], ts[-1]-th[0])
            g = signal.correlate(s, h)
        else:
            if abs(Ts-Th) > 1e-4:
                print("dsppy.corr: error: CT signals with different rates:" +
                      " Sampling periods: T_s = %f != T_h=%f!" % (Ts, Th))
                return None
            tg = ts[0]-th[-1] + arange(len(ts)+len(th)-1)*Ts
            g = signal.correlate(s, h)*Ts
    elif nargs == 2:
        s = args[0]
        h = args[1]
        tg = None
        g = signal.correlate(s, h)
    else:
        print(nargs)
        print("dsppy.corr: wrong number of parameters: %d." %(nargs))
        print("Shold be either 2 or 4!")
        return None
    return tg, g
    
    
def ideal_interpol(nT,sn,t):
    """ 
    performs ideal interpolation (with si/sinc functions) of a given 
    discrete- time signal s to obtain an interpolated signal s_i
    all arguments should be row vectors!
    s_i = ideally interpolated signal as a row vector
    nT = vector containing the sampling points of the discrete -time signal s
    s = discrete-time input signal in row vector form 
    t = time index vector with the sampling points for the interpolated signal
    """
    
    r=1/(nT[2]-nT[1]);
    N=len(nT);
    M= len(t);
    s_i= sn*sinc(r*(ones(N,1)*t-nT*ones(1,M)));
    return s_i

def dtft(s,n,F):   
    
    start_point = -1/2
    end_point= 1/2
    length = abs(start_point - end_point)
    F = arange(start_point,end_point,length/len(n))
    """
    As 's' is calculated according to 'n' number of points, the length of 'F' is
    also calculated same as length of 'n'. 
    To plot 's' against 'F' the length of both of them must be same. Because 
    in the number of points on x-axis and y-axis must be same. Otherwise the 
    code execution will raise a value error.
    """
    #d =zeros(len(F)) # Making all zero array of length F
    #g = d*1j    # Make the all zero array complex numbers
    
    #F=lrange(-.5,.5,1000) # for problem> 4.5
    #F = arange(-1/2,1/2,0.001) # for problem> 4.7
    #e = (complex(0,-1)*2*pi*n)
    #g[0:len(e)] = e
    
    #l = d
    #l[0:len(s)] = s
    #S = s*exp(e*F)
    S = s*exp (-1j*2*pi*transpose(n)*transpose(F))
    return S, F
    
def dtft2(s,n,F):   
    
    #start_point = -1/2
    #end_point= 1/2
    #length = abs(start_point - end_point)
    #F = arange(start_point,end_point,length/len(n))
    """
    As 's' is calculated according to 'n' number of points, the length of 'F' is
    also calculated same as length of 'n'. 
    To plot 's' against 'F' the length of both of them must be same. Because 
    in the number of points on x-axis and y-axis must be same. Otherwise the 
    code execution will raise a value error.
    """
    #d =zeros(len(F)) # Making all zero array of length F
    #g = d*1j    # Make the all zero array complex numbers
    
    #F=lrange(-.5,.5,1000) # for problem> 4.5
    #F = arange(-1/2,1/2,0.001) # for problem> 4.7
    #e = (complex(0,-1)*2*pi*n)
    #g[0:len(e)] = e
    
    #l = d
    #l[0:len(s)] = s
    #S = s*exp(e*F)
    temp = -1j*2*pi
    nF = n * F
    ex = exp (temp*nF)
    S = s*ex
    return S
    
def dtft3(s,n,F):
    
    if len(s)!=len(F) :
        len_add = (len(F) - len(s))%2
        x_pad = len(F) - len(s)
        
        if len_add>0 :
            x_pad = len_add + len(F) - len(s)
            pad_right = int(x_pad/2 - 1)
        else : 
            pad_right = int(x_pad/2)
            
        pad_left = int(x_pad/2)
        
        s = pad(s, (pad_left,pad_right), 'constant', constant_values=(0, 0))
        """
        The above If loop compares the lenth between F and s. If s is smaller 
        than F it pads zeroes on left and right sides of s. Then s and F will 
        be same lenth of array.
        """
    X=[]
    Z=[]
    #s=s
   # w=np.linspace(-np.pi,np.pi,N) # F
    for i in range(0,len(F)):
        F_tmp=F[i]
        X_tmp=0
        for k in range(0,len(F)):
            X_tmp+=(s[k]*(exp(-2j*pi*n[k]*F_tmp)))

        X.append(abs(X_tmp))
        Z.append(angle(X_tmp))
    
    abs_S = X[::-1] 
    angle_S = Z

    return abs_S, angle_S, s