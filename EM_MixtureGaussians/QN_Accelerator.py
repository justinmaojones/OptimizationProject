
import numpy as np
import numpy.linalg as la
from ProblemGenerator import *
from MixtureGaussians import *
from EM import *



np.set_printoptions(
            precision=2,
            edgeitems=4,
            linewidth=1000,
            formatter={'float': lambda x: ('%0.5g' % x).rjust(10)}
            )



M=100
N=3
K=2
x,p_sol,m_sol = generateProblem(M,N,K)
p,m,S = InitialIterate(3,3)

ddp = []
ddm = []
ddS = []

matsize = K+N*K+N*N*K
DDQ = np.zeros((matsize,matsize))

ddp = np.zeros((K,K))
ddm = np.zeros((N*K,N*K))
dmds = np.zeros((N*K,N*N*K))
dmdsT = np.zeros((N*K,N*N*K)).T
dds = np.zeros((K*N**2,K*N**2))

W = Estep(x,[p,m,S])
for j in range(K):
    w = W[j]
    w_sum = np.sum(w)
    
    ddp[j,j] = -w_sum/p[j]**2
    
    ddm[j*N:(j+1)*N,j*N:(j+1)*N] = -w_sum*la.inv(S[j])
    
    
    v = -0.5*np.sum(np.array([w]).T*(x-m[j]),axis=0)
    NI = np.array([np.identity(N)*v[k] for k in range(N)]).reshape(N**2,N).T
    eeT = np.zeros((N,N**2))    
    for k in range(N):
        eeT[k,k*N:(k+1)*N]=v
    dmds[j*N:(j+1)*N,j*N**2:(j+1)*N**2] = NI+eeT
    
    
    for i in range(N):
        for k in range(N):
            dds[j*K+i*N:j*K+(i+1)*N,j*K+k*N:j*K+(k+1)*N] = -0.5*w_sum*np.outer(S[j][i,:],S[j][k,:])




DDQ[0:K,0:K] = ddp
DDQ[K:K+N*K,K:K+N*K] = ddm
DDQ[K:K+N*K,K+N*K:] = dmds
DDQ[K+N*K:,K:K+N*K] = dmds.T
DDQ[K+N*K:,K+N*K:] = dds

print "N =",N
print "K =",K

print S[0]
print dds[:N*N,:N*N]





  