import numpy as np
import numpy.linalg as la

def x_init(p,m,S,M,N,K):
    P = np.random.multinomial(1,p,M)
    Psumlist = np.sum(P,axis=0).tolist()    
    p_sol = np.array(Psumlist)*1.0/sum(Psumlist)

    X = []
    for k in range(K):
        X.append(np.random.multivariate_normal(m[k],S[k],M))

    
    x = np.zeros((M,N))
    for i in range(M):
        p = P[i]
        j = 0
        for k in range(K):
            if p[k]>0:
                j=k
        x[i] = X[j][i]
    
    m_sol = []
    for k in range(K):
        m_sol.append(np.mean(x[P[:,k]==1],axis=0))
    
    
    return x,p_sol,m_sol

def mu_init(N,K):
    m = []
    for k in range(K):
        m.append(np.random.randint(-10,10,N))
    return m

def S_init(N,K,corr=True):
    S = []
    if corr:
        for k in range(K):
            eigvals = np.random.randint(1,20,N)
            randvec = np.random.randint(1,20,N)
            q,r = la.qr(np.reshape(randvec,(N,1)),mode='complete')
            d = np.diag(eigvals)
            S.append(np.dot(q,np.dot(d,q.T)))
            if all(la.eigvals(S[k])>0)==False or all(np.isreal(la.eigvals(S[k])))==False:
                print "WARNING: S not posdef"
    else:
        for k in range(K):
            S.append(np.diag(np.random.uniform(0.1,5,N)**2))
    return S
    
def p_init(K):
    P = np.random.randint(0,1000,K)
    P = P*1.0/sum(P)
    return P.tolist()

def generateProblem(M,N,K,corr=True):
    p = p_init(K)
    mu = mu_init(N,K)
    S = S_init(N,K,corr)
    x,p_sol,m_sol = x_init(p,mu,S,M,N,K)
    return x,p_sol,m_sol
    

def InitialIterate(N,K,corr=True):
    A=[]
    m=[]
    p=[]
    
    for j in range(K):
        if corr:
            A.append(np.identity(N))
        else:
            A.append(np.ones(N))
        m.append(np.ones(N))
        p.append(1.0/K)
    
    return p,m,A
    
def Problem2D():
    A0 = np.array([[1,-0.5],[-0.5,1]])
    A1 = np.array([[3,0.8],[0.8,1]])
    
    m0 = np.array([5,2])
    m1 = np.array([-4,3.5])
    
    p0 = 0.2
    p1 = 0.8
    
    M=10000
    N=2
    K=2
    
    x0 = np.random.multivariate_normal(m0,A0,M)
    x1 = np.random.multivariate_normal(m1,A1,M)
    X = [x0,x1]

    P = np.random.multinomial(1,[p0,p1],M)
    x = np.zeros((M,N))
    for i in range(M):
        p = P[i]
        j = 0
        for k in range(K):
            if p[k]>0:
                j=k
        x[i] = X[j][i]
       
    p_sol = np.sum(P,axis=0)*1.0/M
    m_sol = [np.mean(x0,axis=0),np.mean(x1,axis=0)]
    
    return x,p_sol,m_sol,M,N,K

def Problem3D():
    A0 = np.identity(3)
    A1 = np.identity(3)
    A2 = np.identity(3)
    
    m0 = np.array([5,2,-3])
    m1 = np.array([-4,3.5,0])
    m2 = np.array([-1,-10,4])
    
    p0 = 0.2
    p1 = 0.5
    p2 = 0.3
    
    M=10000
    N=3
    K=3
    
    x0 = np.random.multivariate_normal(m0,A0,M)
    x1 = np.random.multivariate_normal(m1,A1,M)
    x2 = np.random.multivariate_normal(m2,A2,M)
    X = [x0,x1,x2]
    
    P = np.random.multinomial(1,[p0,p1,p2],M)
    x = np.zeros((M,N))
    for i in range(M):
        p = P[i]
        j = 0
        for k in range(K):
            if p[k]>0:
                j=k
        x[i] = X[j][i]
    
    p_sol = np.sum(P,axis=0)*1.0/M
    m_sol = [np.mean(x0,axis=0),np.mean(x1,axis=0),np.mean(x2,axis=0)]
    return x,p_sol,m_sol,M,N,K