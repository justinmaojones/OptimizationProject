import numpy as np
import numpy.linalg as la
import math

c = [552,703,454,180,84,23,4]


def thetavector(gamma,t1,t2):
    return np.matrix([[gamma],[t1],[t2]])

def getparams(theta):
    gamma = theta[0,0]    
    t1 = theta[1,0]
    t2 = theta[2,0]
    return gamma,t1,t2

def f_(j,theta):
    # f(x|theta)    
    gamma,t1,t2 = getparams(theta)
    p = gamma
    f1 = math.exp(-t1)*t1**j/math.factorial(j)
    f2 = math.exp(-t2)*t2**j/math.factorial(j)
    return p*f1+(1-p)*f2
    
def f_tilda_(j,theta):
    gamma,t1,t2 = getparams(theta)
    f1 = math.exp(-t1)*t1**j/math.factorial(j)
    return gamma*f1


def w_(j,theta):
    # gamma*exp(-t1)t1^j/j!/f(j|theta)
    gamma,t1,t2 = getparams(theta)
    f_tilda = f_tilda_(j,theta)
    f = f_(j,theta)
    return f_tilda/f
    
def M_(c,theta):
    # M(theta) = argmax{theta}{Q(theta|theta_k)}
    gamma,t1,t2 = getparams(theta)

    gam = 0
    tt1 = 0
    tt2 = 0
    c_sum = sum(c)
    
    for j in range(len(c)):
        w = w_(j,theta)
        gam += c[j]*w
        tt1 += j*c[j]*w
        tt2 += j*c[j]*(1-w)
    
    denom1 = sum(gam)
    denom2 = c_sum - denom1
    
    gam = gam/c_sum
    tt1 = tt1/denom1
    tt2 = tt2/denom2
    
    return thetavector(gam,tt1,tt2)
    

    

    
    
def dQ_(c,theta):
    # gradient of Q(theta,theta)
    gamma,t1,t2 = getparams(theta)
    
    dga = 0
    dt1 = 0
    dt2 = 0
    
    for j in range(len(c)):
        w = w_(j,theta)
        
        dga += c[j]*(w-gamma)
        dt1 += c[j]*w*(j-t1)
        dt2 += c[j]*(1-w)*(j-t2)
    
    dga = dga/(gamma*(1-gamma))
    dt1 = dt1/t1
    dt2 = dt2/t2
    
    return thetavector(dga,dt1,dt2)

def fz_(j,k,theta):
    # f(j|z=k,theta)    
    gamma,t1,t2 = getparams(theta)    
    if k == 1:
        p = gamma
        t = t1
    else:
        p = 1-gamma
        t = t2
    return math.exp(-t)*t**j/math.factorial(j)
    
    
def dl_(c,theta):
    # gradient of log(L)
    gamma,t1,t2 = getparams(theta)
    
    dga = 0
    dt1 = 0
    dt2 = 0
    
    for j in range(len(c)):
        fz1 = fz_(j,1,theta)
        fz2 = fz_(j,2,theta)
        fj = f_(j,theta)
        
        dga += c[j]*(fz1-fz2)/fj
        dt1 += c[j]*(-1+j/t1)*fz1/fj
        dt2 += c[j]*(-1+j/t2)*fz2/fj
    return thetavector(dga,dt1,dt2)

def conv(c,theta,g):
    # Convergence criteria
    ll_abs = np.abs(ll_(c,theta))
    lmax = max(ll_abs,1)
    g_abs = np.abs(g)
    theta_abs = np.abs(theta)
    theta_max = np.maximum(theta_abs,1)
    rg = np.max(np.multiply(g_abs,theta_max)/lmax)
    return rg    
    
def EM_(c,theta,maxit=5,printing=False):
    theta_prev = theta.copy()
    theta_next = M_(c,theta_prev)
    
    rg = conv(c,theta_next,g_(c,theta_next))
    t = 0
    while t < maxit and rg >= 1e-6:
        t = t+1
        theta_prev = theta_next
        theta_next = M_(c,theta_prev)
        if printing == True:
            print t,theta_prev.T
        rg = conv(c,theta_next,g_(c,theta_next))
    return theta_next,t
#def ddQ_(c,theta):
    

def gt_(c,theta):
    # M(theta) - theta
    return M_(c,theta) - theta

def g_(c,theta):
    # grad of log(L) = grad of Q
    return dQ_(c,theta)

def dS_(dg,dtheta,dtstar):
    # BFGS update for S
    C1 = float((1.0+(dg.T*dtstar)/(dg.T*dtheta))/(dg.T*dtheta))
    C2 = float(1.0/(dg.T*dtheta))
    return C1*dtheta*dtheta.T - C2*(dtstar*dtheta.T + (dtstar*dtheta.T).T)

def testdS_(dg,dgt,dtheta,S):
    # tests dS_
    C1_test = (1+float(dg.T*(-dgt+S*dg))/float(dg.T*dtheta))/float(dg.T*dtheta)
    C2_test = float(dg.T*dtheta)
    testdS = C1_test*dtheta*dtheta.T - (dtheta*(-dgt+S*dg).T+(-dgt+S*dg)*dtheta.T)/C2_test
    return testdS
    
def ll_(c,theta):
    # log likelihood function
    ll = 0
    for j in range(len(c)):
        ll += c[j]*math.log(f_(j,theta))
    return ll

def dA_(A,df,dtheta):
    # Broyden Update for QN1
    return (dtheta - A*df)*dtheta.T*A/float(dtheta.T*A*df)
    

    
def linesearch(c,theta,d,alpha=1.0):
    success = True
    maxJ = 10
    j = 0
    g = g_(c,theta)
    
    def armijo(c,theta,d,alpha):
        eta_c = 1e-4
        return ll_(c,theta + alpha*d) - ll_(c,theta) >= eta_c*alpha*float(g.T*d)
    def wolfe(c,theta,d,alpha):
        eta_w = 0.99
        g1 = float(g_(c,theta+alpha*d).T*d)
        g2 = float(g_(c,theta).T*d)
        return g1 <= eta_w*g2
       
    while (armijo(c,theta,d,alpha) and wolfe(c,theta,d,alpha))==False and j < maxJ:
        alpha = 0.5*alpha
        j = j+1
    if j >= maxJ:
        success = False
    return success,alpha,j

def paramconstraint(theta,d,alpha=1.0):
    t = theta + alpha*d
    
    while t[0] > 1 or any(t < 0):
        alpha = alpha / 2.0
        t = theta + alpha*d
    return alpha
    
def linesearch_secant(c,theta,d,alpha=1.0):
    success = False
    a0 = 0
    a1 = alpha
    n = 0
    
    def F_(c,theta,d,alpha):
        g = g_(c,theta+alpha*d)
        return float(d.T*g)
    def sign(x):
        if x == 0:
            return 0
        elif x < 0:
            return -1
        else:
            return 1
        
    F00 = F_(c,theta,d,a0)
    F0 = F00
    F1 = F_(c,theta,d,a1)
    n = n+1
    while n < 10:
        if n != 1 and abs(F1) < 0.1*F00:
            success = True
            break
        else:
            if sign(a1-a0)*(F0-F1)/(abs(F0)+abs(F1)) < 1e-5:
                break
            else:
                a = (a1*F0 - a0*F1)/(F0-F1)
                a0 = a1
                F0 = F1
                a1 = a  
        F1 = F_(c,theta,d,a1)
        n = n+1
    return success, a1, n
        
theta0 = (0.3,1.0,1.5)
theta = thetavector(*theta0)
theta,_ = EM_(c,theta,maxit=5)
S = np.matrix(np.zeros((3,3)))
g = g_(c,theta)
gt = gt_(c,theta)
rg = conv(c,theta,g)

linesearchiterations = 0

maxit = 100
t = 0
print t,theta.T,g.T*d
while t < maxit and rg >= 1e-6:
    t=t+1
    
    # step a)
    d = gt - S*g
    
    # step b)
    alpha = paramconstraint(theta,d)
    foundalpha, alpha, iters = linesearch(c,theta,d,alpha)
    #foundalpha, alpha, iters = linesearch_secant(c,theta,d,alpha)
    linesearchiterations += iters
    if foundalpha == False:
        S = np.matrix(np.zeros((3,3)))
        g = g_(c,theta)
        gt = gt_(c,theta)
        rg = conv(c,theta,g)
    else:
        dtheta = alpha*d
        
        # step c)
        dg = g_(c,theta+dtheta) - g
        dgt = gt_(c,theta+dtheta) - gt
        
        # step d)
        dtstar = -dgt + S*dg
        dS = dS_(dg,dtheta,dtstar)
        
        # step e)
        theta = theta + dtheta
        g = g + dg
        gt = gt + dgt
        S = S + dS
        
        rg = conv(c,theta,g)    
    print t,theta.T,alpha,foundalpha
print "total linesearch iterations:",linesearchiterations


theta0 = (0.3,1.0,1.5)
theta = thetavector(*theta0)
#theta_EM,k = EM_(c,theta,maxit=100000)
print "EM =",k,theta_EM.T    
'''  
print "################################"
  

    
theta = thetavector(0.1,1.0,1.5)
A = np.matrix(-np.identity(3))
gt = gt_(c,theta)
rg = 1
dtheta = theta

maxit = 1000
t = 0
    
while t < maxit and la.norm(dtheta) >= 1e-6:
    t=t+1    
    
    #print "step 1"
    dtheta = -A*gt    
    alpha = paramconstraint(theta,dtheta)
    dtheta = alpha*dtheta
    dgt = gt_(c,theta+dtheta) - gt
    
    #print "step 2"
    dA = dA_(A,dgt,dtheta)
    A = A + dA
    
    #print "step 3"
    theta = theta + dtheta
    gt = gt + dgt
    
    #print "step rg"
    rg = conv(c,theta,gt)
    
    #print "finished"
    print str(t).ljust(5),theta.T,rg
    print " "*5,dtheta.T,alpha
    
    

maxit = 1000
t = 0
dtheta = 1

while t < maxit and dtheta >= 1e-6:
    t = t+1
    
    # Q-step
    
'''


    