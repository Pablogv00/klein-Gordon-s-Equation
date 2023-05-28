# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 12:08:01 2023

@author: Pablo
"""

import time
from numpy import *
from scipy.sparse.linalg import splu
from scipy.sparse import lil_matrix,identity
from scipy.linalg import lu_factor,lu_solve,cho_factor,cho_solve
from matplotlib.pyplot import *

k = 5
L = pi
T = 0.1
c = 1
gamma = 0



def KG_exacta(t,x):
    return (1/sqrt(k*k*c*c + gamma*gamma))*sin(sqrt(k*k*c*c + gamma*gamma)*t)*sin(k*x)

def u_0(x): 
    return 0.0*x

def u_1(x):
    return sin(k*x)



def KG_implicito(a,b,T,N,ua,ub,npintar):
    N=int(N) 
    T=float(T)
    a=float(a)
    b=float(b)
    gamma2=gamma*gamma
    c2=c*c
    dx=(b-a)/float(N) 
    dx2=dx*dx
    dt=T/N
    M=int(T/dt)+1
    dt=T/M
    dt2=dt*dt
    ua=float(ua)
    ub=float(ub)
    t = linspace(0,T,M+1)
    x = linspace(a,b,N+1)
    xp=x[1:N+1]
    D = lil_matrix((N+1,N+1), dtype='float64')
    Id=identity(N+1,dtype='float64',format='csc')
   
    D.setdiag(2.0*ones(N+1),0)
    D.setdiag(-1.0*ones(N),1)
    D.setdiag(-1.0*ones(N),-1)
    #cambiamos del formato lil al formato csc
    D=D.tocsc()
    
    
    #Calulamos primero U^0
    uold0=zeros(N+1)
    uold0[1:N+1]=u_0(xp)
    uold0[0]=ua
    uold0[N]=ub
    
    u1 = u_1(x) #vector con la malla evaluada en u_1
    
    #Calulamos ahora U^1
    A1 = (2+gamma2*dt2)*Id + (c2*dt2/dx2)*D
    A1[0,0] = 1
    A1[0,1] = 0
    A1[N,N-1] = 0
    A1[N,N] = 1
    LU1 = splu(A1)
    uold1 = LU1.solve(2*uold0 + 2*dt*u1)
    
    #Calulamos U^n+1 para n >= 1:
    A = (1+gamma2*dt2)*Id + (c2*dt2/dx2)*D
    A[0,0] = 1
    A[0,1] = 0
    A[N,N-1] = 0
    A[N,N] = 1
    LU = splu(A)
    
    error = 0
    fig = figure("K-G Implicito")
    for n in range(M+1):
        u = LU.solve(2*uold1 - uold0)
        error_local= max(abs(u-KG_exacta((n+2)*dt,x)))
        error=max(error,error_local)
        if mod(n,npintar)==0:
            plot(x,u,x,KG_exacta((n+2)*dt,x))
            show()
            pause(0.1)
            cla()
            axis([a,b,-0.5,0.5])
        
       
        uold0 = uold1
        uold1 = u
    
    return error


# for N in [50,100,200,400,800,1600,3200]:
#     e1 = KG_implicito(0,L,T,N,0,0,20)
#     e2 = KG_implicito(0,L,T,2*N,0,0,20)
#     p = log(e1/e2)/log(2)
#     print('Para N = ' + str(N) + ', el error es e = ' + str(e1) + ' y el orden es p = ' + str(p))


def u_0_ar(x): 
    return 1


def u_1_ar(x):
    return 0.0*x

# emplearemos esta funcion para calcular soluciones aproximadas de problemas 
# cuyas soluciones exactas desconocemos.


def KG_implicito_ar(a,b,T,N,ua,ub,npintar):
    N=int(N) 
    T=float(T)
    a=float(a)
    b=float(b)
    gamma2=gamma*gamma
    c2=c*c
    dx=(b-a)/float(N) 
    dx2=dx*dx
    dt=T/N
    M=int(T/dt)+1
    dt=T/M
    dt2=dt*dt
    ua=float(ua)
    ub=float(ub)
    t = linspace(0,T,M+1)
    x = linspace(a,b,N+1)
    xp=x[1:N+1]
    D = lil_matrix((N+1,N+1), dtype='float64')
    Id=identity(N+1,dtype='float64',format='csc')
   
    D.setdiag(2.0*ones(N+1),0)
    D.setdiag(-1.0*ones(N),1)
    D.setdiag(-1.0*ones(N),-1)
    #cambiamos del formato lil al formato csc
    D=D.tocsc()
    
    
    #Calulamos primero U^0
    uold0=zeros(N+1)
    uold0[1:N+1]=u_0_ar(xp)
    uold0[0]=ua
    uold0[N]=ub
    
    u1 = u_1_ar(x) #vector con la malla evaluada en u_1
    
    #Calulamos ahora U^1
    A1 = (2+gamma2*dt2)*Id + (c2*dt2/dx2)*D
    A1[0,0] = 1
    A1[0,1] = 0
    A1[N,N-1] = 0
    A1[N,N] = 1
    LU1 = splu(A1)
    uold1 = LU1.solve(2*uold0 + 2*dt*u1)
    
    #Calulamos U^n+1 para n >= 1:
    A = (1+gamma2*dt2)*Id + (c2*dt2/dx2)*D
    A[0,0] = 1
    A[0,1] = 0
    A[N,N-1] = 0
    A[N,N] = 1
    LU = splu(A)
    
    error = 0
    fig = figure("K-G Implicito")
    for n in range(M+1):
        u = LU.solve(2*uold1 - uold0)
        if mod(n,npintar)==0:
            plot(x,u)
            show()
            pause(0.1)
            cla()
            axis([a,b,-1.5,1.5])
        
       
        uold0 = uold1
        uold1 = u
    
T = 10

KG_implicito_ar(0,L,T,1000,0,0,1)
