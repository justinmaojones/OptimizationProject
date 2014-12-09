
import numpy as np
import numpy.linalg as la
from ProblemGenerator import *
from MixtureGaussians import *


'''np.set_printoptions(
            precision=5,
            edgeitems=10,
            linewidth=1000,
            formatter={'float': lambda x: ('%0.5g' % x).rjust(5)}
            )
'''




def Estep(x,theta):
    p,mu,S = theta
    gauss = []
    
    M = len(x)
    K = len(p)    
    
    W = []
    pgauss_sum = np.zeros(M)
    for j in range(K):
        gauss.append(GaussMV(x,mu[j],S[j]))
        W.append(p[j]*gauss[j])
        pgauss_sum += W[j]
    W = W/np.array([pgauss_sum])
    return W

def ThetaRavel(p,m,S):
    N = len(m[0])
    K = len(p)
    
    theta = np.array(p)
    theta = np.hstack((theta,np.array(m).ravel()))
    
    theta_S = np.array(S)
    for k in range(K):
        theta = np.hstack((theta,theta_S[k].T.ravel()))
    
    return theta
    
def ThetaUnravel(theta,N,K):
    p = theta[0:K].tolist()
    
    m = theta[K:K+N*K]
    m = [m[k*K:(k+1)*K] for k in range(K)]
    
    S = theta[K+N*K:]
    S = [S[k*N**2:(k+1)*N**2].reshape((N,N)).T for k in range(K)]
    
    return p,m,S

def dQ(x,theta,theta_k,printstuff=False):
    p,mu,S = theta
    pk,muk,Sk = theta_k
    
    M = len(x)
    N = len(x[0])
    K = len(p)        
    
    dp = []
    dmu = []
    dS = []

    W = Estep(x,theta_k)
    
    W_sum = np.sum(W)
    for j in range(K):
        w = W[j]
        w_sum = np.sum(w)
        A = la.inv(S[j])
        v = x-mu[j]
        lam = W_sum
        
        dmu.append(np.dot(A,np.sum(np.array([w]).T*v,axis=0)))
        dp.append(w_sum/p[j]-lam)
        dS.append(0.5*w_sum*S[j]-0.5*np.dot(v.T,np.array([w]).T*v))
        
        
    #A[j] = np.dot(v.T,np.array([w]).T*v)/w_sum
    DQ = ThetaRavel(dp,dmu,dS)
    
    if printstuff:
        print "||dp|| =",la.norm(np.array(dp)),
        print "||dm|| =",la.norm(np.array(dmu).ravel()),
        print "||dS|| =",la.norm(np.array(dS).ravel()),
    
    return DQ

def dQ_norm(x,theta,theta_k,printstuff=False):
    DQ = dQ(x,theta,theta_k,printstuff=printstuff)    
    return la.norm(DQ)
    


def EM(x,theta,solution,maxit=1000,tol = 10**(-10)):
    p,m,A = theta
    p_sol,m_sol=solution
    
    M = len(x)
    N = len(x[0])
    K = len(p)
    
    
    t = 0
    Merit = dQ_norm(x,theta=[p,m,A],theta_k=[p,m,A])
    print str(t).ljust(3),
    print "||dQ||="+('[%0.6E' % Merit)+']  ',
    print "m=",m[0],m[1]
    
    
    while Merit > tol and t < maxit:
        t=t+1
        
        #  E-STEP
        W = Estep(x,[p,m,A])
        
        #  M-STEP
        for j in range(K):
            w = W[j]
            w_sum = np.sum(w)
            
            m[j] = np.sum(np.array([w]).T*x,axis=0)/w_sum
            p[j] = w_sum/M
            
            v = x-m[j]
            
            A[j] = np.dot(v.T,np.array([w]).T*v)/w_sum
        
        Merit = dQ_norm(x,theta=[p,m,A],theta_k=[p,m,A]) 
        print str(t).ljust(3),
        print "||dQ||="+('[%0.6E' % Merit)+']  ',
        print "m=",        
        for mu in m:
            print mu,
        print "\n",
        
    print "Final Iteration:"
    print "m ="
    print m
    print "p ="
    print np.array(p)
    print "\n",
    print "Solution at:"
    print "m ="
    print m_sol
    print "p ="
    print p_sol





def QN_Accelerator():
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
    
    W = Estep(x,[p,m,S],M)
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
                dds[j*K+i*N:j*K+(i+1)*N,j*K+k*N:j*K+(k+1)*N] = np.outer(S[j][i,:],S[j][k,:])
    
    
    
    
    DDQ[0:K,0:K] = ddp
    DDQ[K:K+N*K,K:K+N*K] = ddm
    DDQ[K:K+N*K,K+N*K:] = dmds
    DDQ[K+N*K:,K:K+N*K] = dmds.T
    DDQ[K+N*K:,K+N*K:] = dds






  