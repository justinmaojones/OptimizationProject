# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 14:27:15 2014

@author: justinmaojones
"""

#import QN_Accel_GaussNoCorrs as QN

import EM as em
import ProblemGenerator as PG
import numpy as np
import numpy.linalg as la
import MixtureGaussians as mg

M=5
N=1
K=2
x,p_sol,m_sol = PG.generateProblem(M,N,K,corr=False)
p,m,S = PG.InitialIterate(N,K,corr=False)



p = [0.2,0.8]
m = [-1,-2]
S = [2,3]

slow = np.zeros((M,K))

for i in range(M):
    for j in range(K):
        v = x[i][0]-m[j]
        sig2 = S[j]
        slow[i,j] = p[j]*np.exp(-0.5*(v**2)/(sig2))/(2*np.pi*sig2)**(0.5)
    slow[i,:] = slow[i,:]/sum(slow[i:,])

m = [np.array([i]) for i in m]
S = [np.array([i]) for i in S]
S_mat = [np.array([s]) for s in S]

n = 1
C = 1/((2*np.pi)**(n/2.0)*la.det(S_mat[1])**(0.5))
v = x-m[1]
A = la.inv(S_mat[1])
Av = np.dot(A,v.T)
vAv = np.sum(v*Av.T,axis=1)
exponent = np.exp(-0.5*vAv)
gauss1 = C*exponent
sig2 = float(S[1])
gauss2 = np.exp(-0.5*(v**2)/(sig2))/(2*np.pi*sig2)**(0.5)

######## E STEP
gauss = []

M = len(x)
K = len(p)    

W = []
pgauss_sum = np.zeros(M)
for j in range(K):
    gauss.append(p[j]*mg.GaussMV(x,m[j],S_mat[j]))
    W.append(gauss[j])
    pgauss_sum += W[j]
W = W/np.array([pgauss_sum])
###########################

W = em.Estep(x,[p,m,S_mat])

def dp(p,w_sum,K):
    dplist = []    
    for j in range(K):
        dplist.append(-w_sum/p[j])
    return dplist
    

j = 0
w = W[j]
e = np.ones(M)
v = x - m[j]
H = np.zeros((2,2))
H[0,0] = np.dot(w,e)
H[0,1] = np.dot(w,v)
H[1,0] = np.dot(w,v)
H[1,1] = np.dot(0.5*w,v**2)
H = -H














