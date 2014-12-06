import numpy as np
import numpy.linalg as la
import math
import scipy.misc

def theta_matrix(gammas,thetas):
    return np.matrix(np.hstack((np.array(gammas),np.array(thetas)))).T

def theta_list(theta):
    m,n = theta.shape
    arr = np.array(theta).ravel().tolist()
    gammas = arr[:m/2]
    thetas = arr[m/2:]
    return gammas,thetas
    
def theta_array(theta):
    m,n = theta.shape
    arr = np.array(theta).ravel()
    gammas = arr[:m/2]
    thetas = arr[m/2:]
    return gammas,thetas
    
def f_(j,theta):
    # f(x|theta)    
    gammas,thetas = theta_array(theta)
    return np.sum(gammas*np.exp(-thetas)*thetas**j/math.factorial(j))

def f_tilda_(k,j,theta):
    gammas,thetas = theta_list(theta)
    t = thetas[k]
    f = math.exp(-t)*t**j/math.factorial(j)
    return gammas[k]*f
    
def f_tilda_array(j,theta):
    gammas,thetas = theta_array(theta)
    return gammas*np.exp(-thetas)*thetas**j/math.factorial(j)

def f_tilda_array2d(J,theta):
    gammas,thetas = theta_array(theta)
    gammas = gammas.reshape(-1,1)
    thetas = thetas.reshape(-1,1)
    return gammas*np.exp(-thetas)*thetas**J/scipy.misc.factorial(J)
    
def w_(J,theta):
    # gamma*exp(-t1)t1^j/j!/f(j|theta)
    f_tilda = f_tilda_array2d(J,theta)
    return f_tilda/np.sum(f_tilda,axis=0)
    
def M_(c,theta):
    # M(theta) = argmax{theta}{Q(theta|theta_k)}
    J = np.arange(len(c))
    w = w_(J,theta)
    wc = w*np.array(c)
    wc_sum = np.sum(wc,axis=1)
    wcJ_sum = np.sum(wc*J,axis=1)
    c_sum = sum(c)
    
    gammas = wc_sum/c_sum
    thetas = wcJ_sum/wc_sum
    return theta_matrix(gammas,thetas)
       
def dQ_(c,theta):
    # gradient of Q(theta,theta)
    gammas,thetas = theta_array(theta)
    
    J = np.arange(len(c))
    w = w_(J,theta)
    wc = w*np.array(c)
    wc_sum = np.sum(wc,axis=1)
    wcJ_sum = np.sum(wc*J,axis=1)
    c_sum = sum(c)

    dgammas = wc_sum/gammas - c_sum
    dthetas = wcJ_sum/thetas - wc_sum
    return theta_matrix(dgammas,dthetas)

def ll_(c,theta):
    # log likelihood function
    ll = 0
    for j in range(len(c)):
        ll += c[j]*math.log(f_(j,theta))
    return ll

def conv(c,theta,g):
    # Convergence criteria
    ll_abs = np.abs(ll_(c,theta))
    lmax = max(ll_abs,1)
    g_abs = np.abs(g)
    theta_abs = np.abs(theta)
    theta_max = np.maximum(theta_abs,1)
    rg = np.max(np.multiply(g_abs,theta_max)/lmax)
    return rg    
    
def EM_(c,gammas0,thetas0,maxit=5,printing=False):
    theta_prev = theta_matrix(gammas0,thetas0)
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

def QN2(c,gammas0,thetas0,maxit=100,ftol=1e-6):
    numparams = len(gammas0+thetas0)
    theta = theta_matrix(gammas0,thetas0)
    theta,_ = EM_(c,gammas0,thetas0,maxit=5)
    S = np.matrix(np.zeros((numparams,numparams)))
    g = g_(c,theta)
    gt = gt_(c,theta)
    rg = conv(c,theta,g)
    
    linesearchiterations = 0
    
    t = 0
    print t,theta.T
    while t < maxit and rg >= ftol:
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


def example1_params():
    c = [552,703,454,180,84,23,4]
    gammas0 = [0.3,0.7]
    thetas0 = [1.0,1.5]
    return c,gammas0,thetas0
    
def example1():
    QN2(*example1_params())