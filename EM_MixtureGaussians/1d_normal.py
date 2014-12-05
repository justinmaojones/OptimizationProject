
import numpy as np
import numpy.linalg as la
import ProblemGenerator as pg
import EM as em

def Gauss(x,mu,sig):
    C = 1/(2*pi*sig**2)**(0.5)
    v = x - mu
    exponent = np.exp(-0.5*(v**2)/sig**2)
    return C*exponent
    
#print p_sol
#print m_sol
#print np.sum(W,axis=1)


#mu = [np.array(i) for i in mu]
#sig = [np.array(i) for i in sig]
#W2 = em.Estep(x,[p,mu,sig])

def Estep(x,theta):
    p,mu,sig = theta
    p = [p,1-p]
    gauss = []

    M = len(x)
    K = len(p)    
    
    W = []
    pgauss_sum = np.zeros((M,1))
    for j in range(K):
        gauss.append(Gauss(x,mu[j],sig[j]))
        W.append(p[j]*gauss[j])
    W = np.hstack(W)
    rowsums = np.sum(W,axis=1).reshape((-1,1))
    W = W/rowsums
    return W


def dqddq(x,theta,w_theta=[]):
    p,mu,sig = theta    
    
    if len(w_theta) > 0:
        W = Estep(x,w_theta)
    else:
        W = Estep(x,[p,mu,sig])

    dp = []
    dmu = []
    dsig = []
    
    ddp = []
    ddmu = []
    ddsig = []
    
    #x = np.matrix(x)
    #W = np.matrix(W)
    W_sum = np.sum(W)
    
    w = W[:,0]
    w_sum = sum(w)
    dp = [w_sum/p - (1-w_sum)/(1-p)]
    ddp = [-w_sum/p**2 - (1-w_sum)/(1-p)**2]
    
    for j in range(K):
        w = W[:,j]
        w_sum = sum(w)
        v = x-mu[j]    
        #lam = W_sum
        
        #dp.append(w_sum/p[j]-lam)
        dmu.append(np.sum(w.T*v)*np.exp(-2*sig[j]))
        dsig.append(-w_sum*(1-np.exp(-2*sig[j])*np.sum(w.T*(v**2))))
        
        #ddp.append(-w_sum/p[j]**2)
        ddmu.append(-w_sum*np.exp(-2*sig[j]))
        #ddsig.append(-2*np.exp(-2*sig[j])*np.sum(w.T*(v**2)))
        ddsig.append(-2*w_sum)
        
    dq = np.array(dp+dmu+dsig)
    ddq = np.diag(ddp+ddmu+ddsig)
    
    return dq,ddq,W
    
def J(p):
    Jx = np.zeros((1,5))
    Jx[0,0] = -1/p
    return Jx
    
def F(dQ,p,lam,mu):
    Jx = J(p)
    F1 = dQ-Jx
    F2 = -lam*log(p)-mu
    return np.hstack((F1,[[F2]]))
    
    
def K_matrix(Hx,p,lam):
    Jx = J(p)
    K11 = Hx
    K12 = Jx.T
    K21 = lam*Jx
    K22 = -lam*log(p)
    K1 = np.hstack((K11,K12))
    K2 = np.hstack((K21,[[K22]]))
    return np.vstack((K1,K2))

def checknegdef(A):
    ev = la.eigvals(A)
    if all(ev < 0):
        return True
    else:
        return False
        
def ThetaUnravel(theta):
    p = theta[0]
    mu = theta[1:3]
    sig = theta[3:5]
    return p,mu,sig

#x,p_sol,m_sol = pg.generateProblem(M=20,N=1,K=2,corr=False)

mu_barr = 0.9
lam = .1
p = 0.5
mu = [np.mean(x)-1,np.mean(x)+1]
sig = [np.std(x),np.std(x)]
theta = np.array([p]+mu+sig)
dQ,ddQ,W = dqddq(x,[p,mu,sig])
B = np.identity(len(dQ))
Hx = ddQ - B
Fx = F(dQ,p,lam,mu_barr)

maxit = 20
ftol = 10**(-5)
approach = 2

if approach == 1:
    Hx = ddQ - B
else:
    Hx = -B

    

t = 0
print t,theta,
while la.norm(Fx) > ftol and t < maxit:
    t = t+1
    
    alpha = 1.0
    maxJ = 100
    j = 0
    
    if approach == 1:
        while checknegdef(ddQ-alpha*B) == False and j < maxJ:
            alpha = 0.5*alpha    
            j = j+1
        if j == maxJ:
            print "MAXJ",checknegdef(ddQ),
        Hx = ddQ-alpha*B
    
    
    theta_prev = theta.copy()
    dQ_prev = dQ.copy()
    
    #alpha
    #theta = theta - 0.1*np.dot(la.inv(ddQ-alpha*B),dQ)
    Fx = F(dQ,p,lam,mu_barr)    
    Kx = K_matrix(Hx,p,lam)
    
    pdelta = -np.dot(la.inv(Kx),Fx.T)
    p_theta = pdelta.ravel()[:-1]
    delta = -pdelta.ravel()[-1]
    
    alpha = 1.0
    while -np.log(p+alpha*p_theta[0]) < 0:
        alpha = alpha/2
    
    print alpha,
    theta = theta + alpha*p_theta
    lam = lam + alpha*delta
    
    p,mu,sig = ThetaUnravel(theta)
    dQ,ddQ,W = dqddq(x,[p,mu,sig])
    dQ_prevtheta,_,_ = dqddq(x,[p,mu,sig],ThetaUnravel(theta_prev))
    
    
    
    if approach == 1:
        # SR1
        s = theta_prev - theta
        g = dQ_prevtheta - dQ_prev
        c = 1/np.dot(g-np.dot(B,s),s)
        v = g - np.dot(B,s)
        Bprev = B.copy()
        if np.dot(g-np.dot(B,s),s) > 1e-3 and abs(np.dot(g-np.dot(B,s),s)) >= 0.2*la.norm(g-np.dot(B,s))*la.norm(s):
            B = B + c*np.outer(v,v)
        else:
            print "Skip Update",
        Hx = ddQ - B
    else:
        #BFGS
        g = dQ_prev-dQ_prevtheta
        y = dQ - dQ_prev - g
        s = theta-theta_prev
    
        y = np.matrix(y).T
        s = np.matrix(s).T
        if y.T*s > 0:
            # BFGS UPDATE
            B = np.matrix(B)
            B = B - B*s*s.T*B/float(s.T*B*s)+y*y.T/float(y.T*s)
            skipped = False
            B = np.array(B)
        else:
            skipped = True
            print "skipped",
        Hx = -B
        
    
    
    print '\n',
    print t,theta,any(W>1),
    
    #gamma = 0.5
    #if abs(1.0/c) < gamma*la.norm(g-np.dot(B,s))*la.norm(s):
    #    print "Use previous B"        
    #    B = Bprev
    
    #print M
    
print '\n'
print p_sol
print m_sol
print la.norm(dQ)


    




