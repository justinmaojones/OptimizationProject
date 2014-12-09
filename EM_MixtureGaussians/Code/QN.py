import numpy as np
import numpy.linalg as la
from EM import *
from MixtureGaussians import *
from utils import *




        

def fx(x):
    return 0

def gx(x,obsX,N,K):
    xarray = np.array(np.matrix(x)).reshape(-1,).copy()
    theta = ThetaUnravel(xarray,N,K)
    g = np.matrix(dQ(obsX,theta,theta)).T
    return g
    
def Mx(g):
    return la.norm(g)


def BFGS(x0,obsX,N,K,Method=2,maxit=10,ftol=1.0E-8,title=""):       
    
    
    x = np.matrix(x0).T*1.0

    f = fx(x)
    g = gx(x,obsX,N,K)
    M = Mx(g)
    B0 = np.matrix(np.identity(len(x0)))
    B = B0.copy()
    
    etas = 0.001
    gammac = 0.5
    
    k =0
    Log = []
    Log.append([k,x.copy(),f,M,1,B,False,False])
    
    
    while k < maxit and M >= ftol:
        k = k+1
        p = -la.inv(B)*g
        alphareset = False
        
        if Method == 1:
            alpha = float(-g.T*p/(p.T*B*p))
        else:
            
            # BACKTRACKING LINE SEARCH
            alpha = 1.0
            j = 0
            maxJ = 60 # MAX NUM BACKTRACKING STEPS
            #print "START ARMIJO"
            #print p
            #print g.T*p
            # ARMIJO SUFFICIENT CONDITION
            
            while f-fx(x+alpha*p) < float(-etas*alpha*g.T*p) and j < maxJ:
                #print f-fx(x+alpha*p),"<",float(-etas*alpha*g.T*p)
                alpha = gammac*alpha
                j = j+1
            if j == maxJ:
                alpha= 1.0
                alphareset = True
        # UPDATE x, y, s
        gprev = g.copy()
        xprev = x.copy()
        x = x + alpha*p
        f = fx(x)
        g = gx(x,obsX,N,K)
        
        M = Mx(g)
        
        y = g-gprev
        s = x-xprev
        
        if y.T*s > 0:
            # BFGS UPDATE
            B = B - B*s*s.T*B/float(s.T*B*s)+y*y.T/float(y.T*s)
            skipped = False
        else:
            skipped = True
        
        Log.append([k,x.copy(),f,M,alpha,B,skipped,alphareset])
    
    print "\n"
    print title
    print "\n",
    if M < ftol:
        converged = "CONVERGED"
        print "x* =", formatfloats(float(x[0])).rjust(12),
        print "       ",
        print "f(x*) = ",formatfloats(float(f))
        for i in range(1,len(x0)):
            print formatfloats(float(x[i])).rjust(18)
        print "\n",
    else:
        converged = "did NOT converge"
    
    print "Method",Method,converged    
    print "etas =",etas
    print "gammac =",gammac   
    print "maxit =",maxit
    print "ftol =",ftol
    
    
    #printlog(Log,Method)
    return x, f



M=1000
N=2
K=2
x,p_sol,m_sol = generateProblem(M,N,K)
p,m,S = InitialIterate(N=N,K=K)
x0 = ThetaRavel(p,m,S)

theta, fa = BFGS(x0,obsX=x,N=N,K=K,Method=1,maxit=500,ftol=1E-8,title="Exercise 5.5 a)")