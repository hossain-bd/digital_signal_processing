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
from cmath import *

from numpy import array, cos, sin, pi, angle, imag, linspace, pad
#from scipy import convolve
from function import lrange, dtft, dtft3, step
from matplotlib.pyplot import legend, figure, xlabel, ylabel, suptitle, \
                            subplots_adjust, show, stem, grid, plot, subplot, \
                            xlim, ylim


""" Problem 3.2 """
fig_num=1
F = lrange(-1/2,1/2,1/0.01)

"""
a) s[n]=(0.6).^abs(n)[ step[ n+10]-step[n-11] ]. """

n = linspace (-11,11,len(F))
s = (0.6)**(abs(n))*(step(n+10)-step(n-11))

print('\n\t\tChapter 4: Problem 4.5')

[abs_S, angle_S, s] = dtft3(s,n,F)

figure (fig_num)
fig_num = fig_num+1
subplot(221)
plot(n,s)
suptitle('Problem 4.5 Part A')
xlabel(r'$n\longrightarrow$')
ylabel(r'$s\longrightarrow$')

graph_label = Line2D([], [], color='blue', marker='o',
                          markersize=1, label='S[n]')
legend(handles=[graph_label])
grid (True)

subplot(222)
plot(F,abs_S) 
xlabel(r'$F\longrightarrow$')
ylabel(r'$|s|\longrightarrow$')
graph_label = Line2D([], [], color='blue', marker='o',
                          markersize=10, label='Abs(S)')
legend(handles=[graph_label])
subplots_adjust(top=.93, bottom=0.1, left=0.1, right=.95, hspace=0.30, wspace=.25)
grid (True)

subplot(212)
ylim(-1,1)
plot(F,angle_S) 
xlabel(r'$F\longrightarrow$')
ylabel(r'$\phi_a(F)\longrightarrow$')   # https://matplotlib.org/users/mathtext.html
graph_label = Line2D([], [], color='blue', marker='o',
                          markersize=10, label='Angle(S)')
legend(handles=[graph_label])
grid (True)
#show()

###############################################################
"""
b) s[n]=n*(0.9)^n [ step[n ]-step[ n-21]]. 
"""


n = linspace(-1,21,len(F))
s = n*(0.9)**(n)*(step(n)-step(n-21))


[abs_S, angle_S,s] = dtft3(s,n,F)

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
plot(F,abs_S) 
xlabel(r'$F\longrightarrow$')
ylabel(r'$|s|\longrightarrow$')
graph_label = Line2D([], [], color='blue', marker='o',
                          markersize=10, label='Abs(S)')
legend(handles=[graph_label])
subplots_adjust(top=.93, bottom=0.1, left=0.1, right=.95, hspace=0.30, wspace=.25)
grid (True)

subplot(212)
plot(F,angle_S) 
xlabel(r'$F\longrightarrow$')
ylabel(r'$\phi_a(F)\longrightarrow$')   # https://matplotlib.org/users/mathtext.html
graph_label = Line2D([], [], color='blue', marker='o',
                          markersize=10, label='Angle(S)')
legend(handles=[graph_label])
grid (True)


###############################################################
""" 
c) s[n]=(cos(0.5 Πn)+ j sin(0.5 Πn))[ step[ n]-step[ n-51] ]
"""


n = linspace(-1,51,len(F))
s = (cos(0.5*pi*n)+1j*(sin(0.5*pi*n)))*(step(n)-step(n-51))

""" COMMENT:
% Both the sine and the cosine signal have the frequency F=0.25. Because
% the sine signal is multiplied by j, the peaks at f=-0.25 cancel out
% while the superposition at F=0.25 doubles the amplitude."""

[abs_S, angle_S,s] = dtft3(s,n,F)

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
plot(F,abs_S) 
xlabel(r'$F\longrightarrow$')
ylabel(r'$|s|\longrightarrow$')
graph_label = Line2D([], [], color='blue', marker='o',
                          markersize=10, label='Abs(S)')
legend(handles=[graph_label])
subplots_adjust(top=.93, bottom=0.1, left=0.1, right=.95, hspace=0.30, wspace=.25)
grid (True)

subplot(212)
plot(F,angle_S) 
xlabel(r'$F\longrightarrow$')
ylabel(r'$\phi_a(F)\longrightarrow$')   # https://matplotlib.org/users/mathtext.html
graph_label = Line2D([], [], color='blue', marker='o',
                          markersize=10, label='Angle(S)')
legend(handles=[graph_label])
grid (True)

##########################################################

""" 
d) s[n] = {4,3,2,1,1,2,3,4} 
"""

n = linspace(-1,8,len(F))
s = array([4, 3, 2, 1, 1, 2, 3, 4])

[abs_S, angle_S, s] = dtft3(s,n,F)

figure (fig_num)
fig_num = fig_num+1
subplot(221)
plot(n,s)
#plot(n,s)
suptitle('Problem 4.5 Part D')
xlabel(r'$n\longrightarrow$')
ylabel(r'$s\longrightarrow$')

graph_label = Line2D([], [], color='blue', marker='o',
                          markersize=10, label='S[n]')
legend(handles=[graph_label])
grid (True)

subplot(222)
plot(F,abs_S) 
xlabel(r'$F\longrightarrow$')
ylabel(r'$|s|\longrightarrow$')
graph_label = Line2D([], [], color='blue', marker='o',
                          markersize=10, label='Abs(S)')
legend(handles=[graph_label])
subplots_adjust(top=.93, bottom=0.1, left=0.1, right=.95, hspace=0.30, wspace=.25)
grid (True)

subplot(212)
plot(F,angle_S) 
xlabel(r'$F\longrightarrow$')
ylabel(r'$\phi_a(F)\longrightarrow$')   # https://matplotlib.org/users/mathtext.html
graph_label = Line2D([], [], color='blue', marker='o',
                          markersize=10, label='Angle(S)')
legend(handles=[graph_label])
grid (True)

###############################################################

"""e) s[n] = {4,3,2,1,-1,-2,-3,-4} """

n = linspace(-1,8,len(F))
s = array([4, 3, 2, 1, -1, -2, -3, -4])
#if len(s)!=len(F) :
#    len_add = (len(F) - len(s))%2
#    if len_add>0 :
#        x_pad = len_add + len(F) - len(s)
#    pad_left = int(x_pad/2)
#    pad_right = int(x_pad/2 - 1)
#
#s = pad(s, (pad_left,pad_right), 'constant', constant_values=(0, 0))
[abs_S, angle_S, s] = dtft3(s,n,F)

figure (fig_num)
fig_num = fig_num+1
subplot(221)
plot(n,s)
#plot(n,s)
suptitle('Problem 4.5 Part E')
xlabel(r'$n\longrightarrow$')
ylabel(r'$s\longrightarrow$')

graph_label = Line2D([], [], color='blue', marker='o',
                          markersize=10, label='S[n]')
legend(handles=[graph_label])
grid (True)

subplot(222)
plot(F,abs_S) 
xlabel(r'$F\longrightarrow$')
ylabel(r'$|s|\longrightarrow$')
graph_label = Line2D([], [], color='blue', marker='o',
                          markersize=10, label='Abs(S)')
legend(handles=[graph_label])
subplots_adjust(top=.93, bottom=0.1, left=0.1, right=.95, hspace=0.30, wspace=.25)
grid (True)

subplot(212)
plot(F,angle_S) 
xlabel(r'$F\longrightarrow$')
ylabel(r'$\phi_a(F)\longrightarrow$')   # https://matplotlib.org/users/mathtext.html
graph_label = Line2D([], [], color='blue', marker='o',
                          markersize=10, label='Angle(S)')
legend(handles=[graph_label])
grid (True)
show()
