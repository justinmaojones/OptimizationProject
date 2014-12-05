
import numpy as np
import numpy.linalg as la



def GaussMV(x,m,Sigma):
    n = x.shape[1]
    C = 1/((2*np.pi)**(n/2.0)*la.det(Sigma)**(0.5))
    v = x-m
    A = la.inv(Sigma)
    Av = np.dot(A,v.T)
    vAv = np.sum(v*Av.T,axis=1)
    exponent = np.exp(-0.5*vAv)
    return C*exponent