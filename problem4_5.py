# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 14:50:24 2017

@author: Nazmul
"""

"""
#First Tutorial 
# Definition of finite-length signals by two vectors:
# an index vector n and a signal value vector s """

from matplotlib.lines import Line2D
from scipy.fftpack import fft

from numpy import array, arange, zeros, append, cos, sin, complex, pi, exp, random, interp, angle, linspace

from function import lrange, dtft, step
from matplotlib.pyplot import legend, figure, xlabel, ylabel, xlim, ylim, title, suptitle, subplots_adjust, setp, show, stem, grid, plot, axis, subplot

""" Problem 3.2 """
fig_num=1
F = lrange(-1/2,1/2,1000)

"""
a) s[n]=(0.6).^abs(n)[ step[ n+10]-step[n-11] ]. """

#F=lrange(-.5,.5,1000)
n = lrange(-10,10,50)
s = (0.6)**(abs(n))*(step(n+10)-step(n-11))

print('\n\t\tChapter 4: Problem 4.5 a')

S = fft(s,len(n))
#S=dtft(s,n,F)
figure (fig_num)
fig_num = fig_num+1
subplot(221)
plot(n,s)
suptitle('Problem 4.5 Part A')
xlabel(r'$n\longrightarrow$')
ylabel(r'$s\longrightarrow$')

graph_label = Line2D([], [], color='blue', marker='o',
                          markersize=10, label='S[n]')
legend(handles=[graph_label])
grid (True)
#show()

subplot(222)
plot(F,abs(S))
xlabel(r'$F\longrightarrow$')
ylabel(r'$|s|\longrightarrow$')
graph_label = Line2D([], [], color='blue', marker='o',
                          markersize=10, label='Abs(S)')
legend(handles=[graph_label])
subplots_adjust(top=.93, bottom=0.1, left=0.1, right=.95, hspace=0.30, wspace=.25)
grid (True)

subplot(212)
plot(F,angle(S,deg=False)) 
xlabel(r'$F\longrightarrow$')
ylabel(r'$\phi_a(F)\longrightarrow$')   # https://matplotlib.org/users/mathtext.html
graph_label = Line2D([], [], color='blue', marker='o',
                          markersize=10, label='Angle(S)')
legend(handles=[graph_label])
grid (True)


###############################################################

###############################################################
"""
#b) s[n]=n*(0.9)^n [ step[n ]-step[ n-21]]. 
"""


n = lrange(0,20,50)
s = n*(0.9)**(n)*(step(n)-step(n-21))

print('\n\t\tChapter 4: Problem 4.5 a')

S = fft(s,len(n))

figure (fig_num)
fig_num = fig_num+1
subplot(221)
plot(n,s)
suptitle('Problem 4.5 Part B')
xlabel(r'$n\longrightarrow$')
ylabel(r'$s\longrightarrow$')

graph_label = Line2D([], [], color='blue', marker='o',
                          markersize=10, label='S[n]')
legend(handles=[graph_label])
grid (True)

subplot(222)
plot(F,abs(S)) 
xlabel(r'$F\longrightarrow$')
ylabel(r'$|s|\longrightarrow$')
graph_label = Line2D([], [], color='blue', marker='o',
                          markersize=10, label='Abs(S)')
legend(handles=[graph_label])
subplots_adjust(top=.93, bottom=0.1, left=0.1, right=.95, hspace=0.30, wspace=.25)
grid (True)

subplot(212)
plot(F,angle(S,deg=False)) 
xlabel(r'$F\longrightarrow$')
ylabel(r'$\phi_a(F)\longrightarrow$')   # https://matplotlib.org/users/mathtext.html
graph_label = Line2D([], [], color='blue', marker='o',
                          markersize=10, label='Angle(S)')
legend(handles=[graph_label])
grid (True)


###############################################################


""" 
#c) s[n]=(cos(0.5 Πn)+ j sin(0.5 Πn))[ step[ n]-step[ n-51] ]
"""


n = lrange(0,50,20)
s = (cos(0.5*pi*n)+complex(0+1)*(sin(0.5*pi*n)))*(step(n)-step(n-51))

""" 
 COMMENT:
% Both the sine and the cosine signal have the frequency F=0.25. Because
% the sine signal is multiplied by j, the peaks at f=-0.25 cancel out
% while the superposition at F=0.25 doubles the amplitude.
""" 
S = fft(s,len(n))

figure (fig_num)
fig_num = fig_num+1
subplot(221)
plot(n,s)
suptitle('Problem 4.5 Part C')
xlabel(r'$n\longrightarrow$')
ylabel(r'$s\longrightarrow$')

graph_label = Line2D([], [], color='blue', marker='o',
                          markersize=10, label='S[n]')
legend(handles=[graph_label])
grid (True)

subplot(222)
plot(F,abs(S)) 
xlabel(r'$F\longrightarrow$')
ylabel(r'$|s|\longrightarrow$')
graph_label = Line2D([], [], color='blue', marker='o',
                          markersize=10, label='Abs(S)')
legend(handles=[graph_label])
subplots_adjust(top=.93, bottom=0.1, left=0.1, right=.95, hspace=0.30, wspace=.25)
grid (True)

subplot(212)
plot(F,angle(S,deg=False)) 
xlabel(r'$F\longrightarrow$')
ylabel(r'$\phi_a(F)\longrightarrow$')   # https://matplotlib.org/users/mathtext.html
graph_label = Line2D([], [], color='blue', marker='o',
                          markersize=10, label='Angle(S)')
legend(handles=[graph_label])
grid (True)


##########################################################


""" 
#d) s[n] = {4,3,2,1,1,2,3,4} 
"""

n = lrange(0,8,125)
a = array([4, 3, 2, 1, 1, 2, 3, 4])
l=lrange(0,7)
z=zeros(993)
#a=[4,3,2,1,1,2,3,4]
s=append(a,z)
S = fft(s,len(n))

figure (fig_num)
fig_num = fig_num+1
subplot(221)
plot(l,a)
#plot(n,s)
suptitle('Problem 4.5 Part D')
xlabel(r'$n\longrightarrow$')
ylabel(r'$s\longrightarrow$')

graph_label = Line2D([], [], color='blue', marker='o',
                          markersize=10, label='S[n]')
legend(handles=[graph_label])
grid (True)

subplot(222)
plot(F,abs(S)) 
xlabel(r'$F\longrightarrow$')
ylabel(r'$|s|\longrightarrow$')
graph_label = Line2D([], [], color='blue', marker='o',
                          markersize=10, label='Abs(S)')
legend(handles=[graph_label])
subplots_adjust(top=.93, bottom=0.1, left=0.1, right=.95, hspace=0.30, wspace=.25)
grid (True)

subplot(212)
plot(F,angle(S,deg=False)) 
xlabel(r'$F\longrightarrow$')
ylabel(r'$\phi_a(F)\longrightarrow$')   # https://matplotlib.org/users/mathtext.html
graph_label = Line2D([], [], color='blue', marker='o',
                          markersize=10, label='Angle(S)')
legend(handles=[graph_label])
grid (True)


###############################################################


"""
 #e) s[n] = {4,3,2,1,-1,-2,-3,-4} 
"""

n = lrange(0,10,100)
a = array([0, 4, 3, 2, 1, -1, -2, -3, -4, 0])
l=lrange(0,9)
z=zeros(991)
#a=[4,3,2,1,1,2,3,4]
s=append(a,z)
S= fft(s,len(n))

figure (fig_num)
fig_num = fig_num+1
subplot(221)
plot(l,a)
#plot(n,s)
suptitle('Problem 4.5 Part E')
xlabel(r'$n\longrightarrow$')
ylabel(r'$s\longrightarrow$')

graph_label = Line2D([], [], color='blue', marker='o',
                          markersize=10, label='S[n]')
legend(handles=[graph_label])
grid (True)

subplot(222)
plot(F,abs(S)) 
xlabel(r'$F\longrightarrow$')
ylabel(r'$|s|\longrightarrow$')
graph_label = Line2D([], [], color='blue', marker='o',
                          markersize=10, label='Abs(S)')
legend(handles=[graph_label])
subplots_adjust(top=.93, bottom=0.1, left=0.1, right=.95, hspace=0.30, wspace=.25)
grid (True)

subplot(212)
plot(F,angle(S,deg=False)) 
xlabel(r'$F\longrightarrow$')
ylabel(r'$\phi_a(F)\longrightarrow$')   # https://matplotlib.org/users/mathtext.html
graph_label = Line2D([], [], color='blue', marker='o',
                          markersize=10, label='Angle(S)')
legend(handles=[graph_label])
grid (True)
show()