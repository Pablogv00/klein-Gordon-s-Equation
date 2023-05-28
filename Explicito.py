#Metodo Explicito

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
gamma = 3


def KG_exacta(t,x):
    return (1/sqrt(k*k*c*c + gamma*gamma))*sin(sqrt(k*k*c*c + gamma*gamma)*t)*sin(k*x)

def u_0(x): 
    return 0.0*x

def u_1(x):
    return sin(k*x)

# emplearemos esta funcion para comparar con la exacta:
    
def KG_explicito(a,b,T,N,ua,ub,npintar):
    N=int(N) 
    T=float(T)
    a=float(a)
    b=float(b)
    gamma2=gamma*gamma
    c2=c*c
    dx=(b-a)/float(N) 
    dx2=dx*dx 
    dt = 0.9*(1/(sqrt(c2/dx2 + (1/4)*gamma2))) #Condicion CFL
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
    A1 = (1-gamma2*dt2/2)*Id - (c2*dt2/(2*dx2))*D
    A1[0,0] = 1
    A1[0,1] = 0
    A1[N,N-1] = 0
    A1[N,N] = 1
    uold1 = A1*uold0 + dt*u1
    
    #Calulamos U^n+1 para n >= 1:
    A = (2-gamma2*dt2)*Id - (c2*dt2/dx2)*D
    A[0,0] = 1
    A[0,1] = 0
    A[N,N-1] = 0
    A[N,N] = 1
    
    error = 0
    fig = figure("K-G Explicito")
    for n in range(M+1):
        u = A*uold1 - uold0
        error_local= max(abs(u-KG_exacta((n+2)*dt,x)))
        error=max(error,error_local)
        if mod(n,npintar)==0:
            plot(x,u,x,KG_exacta((n+2)*dt,x))
            show()
            pause(0.1)
            cla()
            axis([0,L,-0.5,0.5])
            
        uold0 = uold1
        uold1 = u
    
    
    return error   


T = 50

def u_0_ar(x):
    return x*(x<=L/2) + (L-x)*(x>L/2)

def u_1_ar(x):
    return 0.0*x

# emplearemos esta funcion para calcular soluciones aproximadas de problemas 
# cuyas soluciones exactas desconocemos.

def KG_explicito_ar(a,b,T,N,ua,ub,npintar):
    t1=time.time()
    N=int(N) 
    T=float(T)
    a=float(a)
    b=float(b)
    gamma2=gamma*gamma
    c2=c*c
    dx=(b-a)/float(N) 
    dx2=dx*dx 
    dt = 0.9*(1/(sqrt(c2/dx2 + (1/4)*gamma2))) #Condicion CFL
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
    uold0=u_0_ar(x)
    #uold0=zeros(N+1)
    #uold0[1:N+1]=u_0_ar(xp)
    #uold0[0]=ua
    #uold0[N]=ub
    
    u1 = u_1_ar(x) #vector con la malla evaluada en u_1
    
    #Calulamos ahora U^1
    A1 = (1-gamma2*dt2/2)*Id - (c2*dt2/(2*dx2))*D
    A1[0,0] = 1
    A1[0,1] = 0
    A1[N,N-1] = 0
    A1[N,N] = 1
    uold1 = A1*uold0 + dt*u1
    
    #Calulamos U^n+1 para n >= 1:
    A = (2-gamma2*dt2)*Id - (c2*dt2/dx2)*D
    A[0,0] = 1
    A[0,1] = 0
    A[N,N-1] = 0
    A[N,N] = 1
    
    error = 0
    fig = figure("K-G Explicito")
    for n in range(M+1):
        t = n*dt
        u = A*uold1 - uold0
        if mod(n,npintar)==0:
            title("Tiempo: " + str(round(t,1)))
            plot(x,u)
            show()
            pause(0.01)
            cla()
            axis([a,b,-2,2])
            
        uold0 = uold1
        uold1 = u
    
    
    tf=time.time()
    print ("Tiempo de ejecucion:",format(tf-t1))
    #print("Error cometido:",format(error))
    return error
