
import numpy as np
import numpy.linalg as la
from ProblemGenerator import *
from MixtureGaussians import *
from EM import *



np.set_printoptions(
            precision=2,
            edgeitems=4,
            linewidth=1000,
            formatter={'float': lambda x: ('%0.2g' % x).rjust(7)}
            )


    
def D_ThetaUnravel(theta,N,K):
    p = theta[0:K].tolist()
    
    m = theta[K:K+N*K]
    m = [m[k*N:(k+1)*N] for k in range(K)]
    
    S = theta[K+N*K:]
    S = [S[k*N:(k+1)*N] for k in range(K)]
    
    return p,m,S






def dQ_ddQ(x,theta):
    p,m,S = theta
    
    dp = []
    dm = []
    dS = []
    
    ddp = []
    ddm = []
    dmds = []
    ddS = []
    
    S_mat = [np.diag(s) for s in S]    
    
    W = Estep(x,[p,m,S_mat])
    W_sum = np.sum(W)
    for k in range(K):
        w = W[k]
        w_sum = np.sum(w)    
        v = x-m[k]
        sigsinv = 1/S[k]
        Sinv = np.diag(sigsinv)
        
        dp.append(
                w_sum/p[k] - W_sum
                )
        dm.append(
                np.dot(Sinv,np.dot(v.T,w))
                )
        dS.append(
                w_sum*(sigsinv**(0.5))
                +np.dot((Sinv**(1.5)),np.dot((v**2).T,w))
                )
        ddp.append(
            -w_sum/p[k]**2
            )
        ddm.append(
                np.diag(
                    -w_sum*Sinv            
                    )
                )
        dmds.append(
                np.diag(
                    -2*np.dot(Sinv**(1.5),np.diag(np.dot(v.T,w)))
                    )
                )
        ddS.append(
                np.diag(
                    w_sum*Sinv
                    -3*np.dot(Sinv**2,np.diag(np.dot((v**2).T,w)))    
                    )
                )
    
    dQ = [dp,dm,dS]
        
    ddQ = -np.diag(ThetaRavel(ddp,ddm,ddS))
    #dmds_mat = np.diag(np.array(dmds).ravel())
    #ddQ[K:K+N*K,K+N*K:]=dmds_mat
    #ddQ[K+N*K:,K:K+N*K]=dmds_mat.T            
    return dQ,ddQ 


def Mx(dtheta):
    return la.norm(dQ)

def isnegdef(A):
    return all(la.eigvals(A)<0)

##################################################


M=100
N=2
K=2
x,p_sol,m_sol = generateProblem(M,N,K,corr=False)
p,m,S = InitialIterate(N,K,corr=False)

mean_init = np.mean(x,axis=0)
std_init = np.std(x,axis=0)

#x = (x-mean_init)/std_init


dQ_list,ddQ = dQ_ddQ(x,[p,m,S])
dQ = ThetaRavel(*dQ_list)
dQ = -dQ
M = Mx(dQ)
B = -ddQ#np.identity(K+2*N*K)
theta = ThetaRavel(p,m,S)


maxit = 100
ftol = 10**(-5)

t = 0
while M > ftol and t < maxit:
    print t
    t = t+1
    
    if not isnegdef(ddQ):
        print "ddQ not posdef"
    
    #Hprev = H.copy()
    #H = ddQ - B
    alpha = 1.0
    maxJ = 100
    j = 0
    while not isnegdef(B) and j < maxJ:
        j=j+1
        alpha = alpha/2.0
        #H = ddQ - alpha*B
    if j == maxJ:
        print "ERROR:  MaxJ hit"
        #H = Hprev
        
        
    #Hinv = la.inv(H)
    theta_prev = theta.copy()
    dQ_prev = dQ.copy()
    
    # Update parameters, then update dQ and merit function
    #theta = theta - np.dot(Hinv,dQ)
    theta = theta - np.dot(la.inv(B),dQ)
    p,m,S = D_ThetaUnravel(theta,N,K)
    
    dQ_list,ddQ = dQ_ddQ(x,[p,m,S])
    dQ = ThetaRavel(*dQ_list)
    dQ = -dQ
    M = Mx(dQ)
    
    # Update B
    s = theta_prev - theta
    g = dQ_prev - dQ    
    c = 1/np.dot(g-np.dot(B,s),s)
    v = g - np.dot(B,s)
    #Bprev = B.copy()
    #B = B + c*np.outer(v,v)
    
    y = dQ - dQ_prev
    s = theta - theta_prev
    if np.dot(y,s) > 0:
        # BFGS UPDATE
        yTs = np.dot(y,s)
        yyT = np.outer(y,y)
        sTBs = np.dot(s,np.dot(B,s))
        BssTB = np.dot(B,np.dot(s,np.dot(s,B)))
        B = B - BssTB/sTBs + yyT/yTs
        #B = B - B*s*s.T*B/float(s.T*B*s)+y*y.T/float(y.T*s)
        skipped = False
    else:
        skipped = True
        print "skipped"
    
    #gamma = 0.5
    #if abs(1.0/c) < gamma*la.norm(g-np.dot(B,s))*la.norm(s):
    #    print "Use previous B"        
    #    B = Bprev
    
    print M
    
    
 


