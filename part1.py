# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 14:50:24 2017

@author: Nazmul
"""

"""
#First Tutorial 
# Definition of finite-length signals by two vectors:
# an index vector n and a signal value vector s """

from numpy import array, arange, cos, pi, exp, random, interp
from scipy import convolve
from function import delta, step, rect, SigAdd, conv, sigshift, sigfold, corr, ideal_interpol
from matplotlib.pyplot import figure, xlabel, ylabel, xlim, ylim, title, suptitle, subplots_adjust, setp, show, stem, grid, plot, axis, subplot
import pylab




s = array([1., 2., -1., 1.])
n = array([0., 1., 2., 3.])

""" Signal plot with a given index"""

print("Signal plot with a given index")
figure(1)
ylim(-2, 3)
xlim(-1, 4)
suptitle('an index vector n and a signal S ')
xlabel('n')
ylabel('s')
impulse = stem(n, s)
setp(impulse, 'markerfacecolor', 'b')
# setp(baseline, 'color', 'r', 'linewidth', 2)
grid(True)
show()

"""Signal plot with a given index plus padding"""


figure(2)
s = array([0., 0., 1., 2., -1., 1., 0., 0.])
n = array([-2., -1., 0., 1., 2., 3., 4., 5. ])
ylim(-2, 3)
xlim(-3, 6)
suptitle('An index vector n and a signal S using padding')
xlabel('n')
ylabel('s')
markerline, stemlines, baseline = stem(n, s)
setp(markerline, 'markerfacecolor', 'b')
# setp(baseline, 'color', 'r', 'linewidth', 2)
grid(True)
show()

"""delta function"""

#a=int(input('Enter a number for delta:'))
a=0;   
n=arange(-5,10)  
dd=delta(n,a)
figure(3)
suptitle('delta function')
ylim(-0.5, 1.5)
xlim(-6, 11)
xlabel('n')
ylabel('a')
stem(n,dd)
grid(True)

""" delta function behaves like "normal" i.e. built-in function"""
n==0;
se=n==0;

figure(4)
suptitle('SE function')
xlabel('n')
ylabel('se')
stem(n,se)
grid(True)


""" Step function"""
    
    
t = arange(-1,5,1)
x = step(t)
figure(5)
ylim(-1, 3)
xlim(-1, 5)
xlabel('t')
ylabel('r')
suptitle('Step function ')
stem(t,x)
grid(True)

""" Shifted step function by 1 to the right"""

figure(6)
ylim(-1, 3)
xlim(-1, 5)
xlabel('t')
ylabel('r')
x = step(t - 1.0)
stem(t,x)
suptitle('Step function step(t - 1.0) ')
grid(True)

""" Rect function"""

t = arange(-1,5,.01)
figure(7)
suptitle('an index vector t and a signal S: rect function ')
x = rect(t)
plot(t,x)
grid(True)

""" shifted by 1 to the right"""

x = rect(t - 1.0)
figure(8)
suptitle('an index vector t and a signal S: rect function rect(t - 1.0)')
plot(t,x)
grid(True)

""" sinusoidal"""

x=arange(-10,10.001,0.001)


figure(9)
suptitle('Sinusoidal ')
c=cos(2*pi*(x-2))
#plot(x,c)
pylab.plot(x,c,label='cosx')
pylab.legend(loc='upper right')


figure(10)
suptitle('cos(n) ')
stem(n,cos(n))  # stem is used only to draw step function

figure(11)
suptitle('cos(2*pi/8*n)')
stem(n,cos(2*pi/8*n))

figure(12)
suptitle('cos(2*pi*5/8*n)')
stem(n,cos(2*pi*5/8*n))

figure(13)
suptitle('cos(2*pi*5/8*n) and cos(2*pi*5/8*x1)')
stem(n,cos(2*pi*5/8*n))
markerline, stemlines, baseline = stem(n,cos(2*pi*5/8*n))
setp(stemlines, 'color', 'r')
setp(markerline, 'markerfacecolor', 'r')
x1 = arange(-4,10.001,0.001)
plot(x1,cos(2*pi*5/8*x1))
#grid(True)

""" accessing data: complete vector"""
x
""" python indexing starts at 0"""
x[0]

""" trying c or Java like indexing starting at 1 results in an error"""
x[0]
x[1]

""" accessing the last element"""
x[-1]
""" generating a logical index vector """

x = arange(-4,10.001,0.001)
a=x[-1]
#x>=-2
x2=(x>=-2)*1 #convert to binary
x4=(x<=4)*1
x24 = x2 & x4
figure(14)
suptitle('x2 & x4')
plot(x24)

""" # Operations on signals
# generate two sequences of different length"""

n1= arange(-4,7)
s1=cos(pi/6*n1)
n2= arange(-2,11)
s2=cos(pi/8*n2)
len(s1)
len(s2)

""" trying the + operator to add them results in an error"""

#s1+s2

""" #since the vectors are of different size
#generate new index vector containing all necessary indices"""

n = arange((min(min(n1),min(n2))),(max(max(n1),max(n2)))+1)

""" create new vectors of equal size"""

#y1 = np.zeros(shape=(1,len(n)))

"""now find the index vectors n1 and n2 within n"""


na=((n>=min(n1)) & (n<=max(n1)))*1
nb=((n>=min(n2)) & (n<=max(n2)))*1


r1 = [i for i, h in enumerate(na) if h == 1] # i for i used for iteration and enumerate for index
print (r1)

r2 = [i for i, h in enumerate(nb) if h == 1]
print(r2)

"""lets now look at a function to add signals"""

figure(15)
suptitle('signal Add')
[y,n]=SigAdd(s1,n1,s2,n2)
stem(n,y)

""" exponential signals"""

n = arange(-3,10)
s=exp(-.5*n)
figure(16)
suptitle('exponential signals')
stem(n,s)

"""complex exponential signals"""

s = exp(-.5*1j*n)
figure(17)
suptitle('complex exponential signals')
plot(n,abs(s))

s = exp((-0.7-.5*1j)*n)
figure(18)
suptitle('complex exponential signals')
plot(n,abs(s))

""" random signals
 uniformly distributed: """
 
uni_s = random.rand (100)
sum(uni_s)
figure(19)
suptitle('uniformly distributed:')
plot(uni_s)


""" normally distributed:"""
norm_s = random.randn (100)
sum(norm_s)
figure(20)
suptitle('normally distributed:')
plot(norm_s)

""" Second tutorial"""
"""one dimentional input array"""
s = array ([3, 11, 7, 0, -1, 4, 2]) 
h = array ([2, 3, 0, -5, 2, 1])
g = convolve(s, h)


""" The conv function assumes that the two sequences start at n = 0
However, the conv function does not provide or accept any timing 
information if the sequences have arbitrary support. What is needed is a
start point and an end point of g[n] """


ns=arange (-3,4)
nh=arange(-2,4)
[tg,g]=conv(ns,s,nh,h)
figure(21)
suptitle('convolution function:')
stem(tg,g)

"""crosscorrelation """
""" sequence 1"""
s1 = array ([-1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1])
ns = arange(-5,6)
"""generate shifted sequence"""
[y,ny]=sigshift(s1,ns,2)
figure(22)
suptitle('shifted sequence:')
stem(ny,y)

""" generate noise"""

noise = 2*random.normal(0,1,len(y))
noise_index = ny
"""add moise to shifted sequence"""
[y,ny] = SigAdd(y,ny,noise,noise_index)
figure(23)
suptitle('Add noise to shifted sequence:')
stem(ny,y)

"""s[-n]"""
s=s1
[s,ns] = sigfold(s,ns)
figure(24)
suptitle('Flip the signal:')
stem(ns,s)
#s=array([3,11,7,0,-1,4,2])
"""Flip the array in the up to down direction"""
#flip_signal=flipud(s)
"""flip : Reverse the order of elements in an array along the given axis."""


"""compute cross-correlation via convolution"""

[nC, C1] = conv(ny,y,ns,s)
figure(25)
#subplot(1,1,1) 
subplot(211)
stem(C1, nC)
#axis([-5,10,-15,20])
xlabel('lag variable 1')
ylabel('C_{sy}')
title('Crosscorrelation: sequence 1')
grid(True)

""" lets use a better sequence """
sb= [ 1, 1, 1,-1,-1,-1, 1,-1,-1,1,-1]
s=sb

""" generate shifted sequence"""
[y,ny] = sigshift(s,ns,2)

""" generate noise"""
noise =2* random.normal(0,1,len(y))
noise_index = ny
""" add moise to shifted sequence"""
[y,ny] = SigAdd(y,ny,noise,noise_index)
# s[-n]

[s,ns] = sigfold(s,ns)
""" compute cross-correlation via convolution"""

[nC, C1] = conv(ny,y,ns,s)
#figure(26)
subplot(212)
stem(C1, nC)
title('Crosscorrelation: sequence 2')
#axis([-5,10,-15,20])
xlabel('lag variable 1')
ylabel('C_{sy}')
subplots_adjust(top=1, bottom=0.1, left=0.1, right=1, hspace=0.50, wspace=0.35)
grid(True)

""" why is sequence 2 better than sequence 1?"""
""" let's compute their ACFs:"""
nc=arange(-10,11)
[nc11,C11]=corr(s1,s1)
[nc22,C22]=corr(sb,sb)
figure(26)
subplot(211)
stem(nc,C11)
title('ACF')
subplot(212) 
stem(nc,C22)


""" Homework:
% Problem 2.7.
% Solution: A Matlab script (.m file) that generates the requested plots.
% Write an m-file corr_m.m to implement the cross-correlation function corr_m
% of two signals with arbitrary support that behaves like the conv_m function.
% Third tutorial: interpolation """

t1=-1
t2=-t1
"""sampling period"""
T=0.1
"""step size for the analog signal (which is also a discrete-time signal in
Matlab/Ocatve, but with a much smaller sampling period)"""
dt=.001
"""create the support vector (time vector) for the analog signal"""
t=arange(t1,t2,dt)
"""vector of the sampling points"""
nT=arange(t1,t2,T)
""" create a very peaky two-sided exponential impulse"""
c=10
""" analog signal"""
s=exp(-c*abs(t))
"""discrete-time signal"""
sn=exp(-c*abs(nT))
figure(27)
plot(t,s)
stem(nT,sn)

""" ideal inpterpolation with si functions"""
s_r=ideal_interpol(nT,sn,t)
""" plot reconstructed signal"""
figure(28)
plot(nT, s_r)
title('ideal inpterpolation with si function')
""" this doesn't look too good why? """
""" inpterpolation with spline functions"""