# -*- coding: utf-8 -*-
"""
Created on Wed May 24 18:25:59 2023

@author: Pablo
"""

from pylab import *


L = pi
c = 1
A = 1

def dispersion(L,T,Nx,Nt,gamma):
    ''' 

    Parameters
    ----------
    L : float
        punto final el intervalo
    T : float
        tiempo final para pintar
    Nx : int
        División espacial (Nx+1 puntos)
    Nt : int
        División temporal (Nt+1 puntos)

    Returns
    -------
    None.

    '''
    npintar = 5 
    
    dx = L/float(Nx)
    x = arange(0,L+dx,dx)
    
    dt = T/float(Nt)
    
    for n in range(0,Nt+1):
        t = n*dt
        u3 = A*cos(3*(x-sqrt(c*c + gamma*gamma/(3*3))*t))
        u5 = A*cos(5*(x-sqrt(c*c + gamma*gamma/(5*5))*t))
        if mod(n,npintar) == 0:
            plot(x,u3,'b',x,u5,'r') 
            legend(('k = 3','k = 5'))
            show()
            pause(0.1)
            cla()
            axis([0,L,-1.5,1.5])


#gamma = 0  
gamma = 15      
dispersion(pi,5,1000,1000,gamma)

