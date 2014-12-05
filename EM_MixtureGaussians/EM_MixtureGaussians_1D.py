import numpy as np
import numpy.linalg as la
import scipy as sp
import scipy.stats


s0 = 0.87
s1 = 0.77
s_init = [s0,s1]

m0 = 4.62
m1 = 4.06
m_init = [m0,m1]

p0 = 0.546
p1 = 1-p0
p_init = [p0,p1]

M=10000
N=1
K=2

x0 = np.random.normal(loc=m0,scale=s0**(0.5),size=M)
x1 = np.random.normal(loc=m1,scale=s1**(0.5),size=M)
X = [x0,x1]

P = np.random.multinomial(1,[p0,p1],M)
x = np.zeros((M,N)).T
for j in range(K):
    x += X[j]*P[:,j]
x=x[0]


s0 = 0.87
s1 = 0.77
s = [s0,s1]
A0 = s0
A1 = s1
A = [A0,A1]

m0 = 0
m1 = 0
m = [m0,m1]

p0 = 0.546
p1 = 1-p0
p = [p0,p1]



def Gauss(xi,m,s):
    return sp.stats.norm.pdf(xi,loc=m,scale=s**(0.5))

def Convergence(p,m,s,method):
    tol = 10**(-5)
    converged = True
    for i in range(K):
        if any(abs(s[i]-s_init[i])>tol):
            converged = False
        if any(abs(m[i]-m_init[i])>tol):
            converged = False
        if method == 0:
            if any(abs(p[i]-p_init[i])>tol):
                converged = False
        if abs(p-p_init[0])>tol:
            converged = False
    return converged

t = 0
maxit = 20
print "m =",m[0],m[1]
print "s =",s[0],s[1]
print "p =",p[0],p[1]
while t < maxit:
    t=t+1
    
    #  E-STEP
    gauss = []
    W = []
    pgauss_sum = np.zeros(M)
    for j in range(K):
        gauss.append(Gauss(x,m[j],s[j]))
        #print gauss
        W.append(p[j]*gauss[j])
        pgauss_sum += W[j]
    #print W[0]
    #print W[1]
    W = W/pgauss_sum
    
    
    #  M-STEP
    for j in range(K):
        w = W[j]
        sum_w = np.sum(w)
        m[j] = sum(w*x)/sum_w
        s[j] = sum(w*(x-m[j])**2)/sum_w
        p[j] = sum_w/M
    print "m =",m[0],m[1]
    #print "s =",s
    #print "p =",p

print "###################################"
 
t=0   
print "m =",m0,m1
print "s =",s0,s1
print "p =",p0,p1
while t < maxit:
    t=t+1   
    
    gamma = np.zeros(M)
    gauss1 = Gauss(x,m0,s0)
    gauss2 = Gauss(x,m1,s1)
    gamma = p1*gauss2
    #print gamma
    gamma = gamma/((1-p1)*gauss1+p1*gauss2)
    
    #print gamma
    
    m0 = sum((1-gamma)*x)/sum(1-gamma)
    m1 = sum(gamma*x)/sum(gamma)
    
    s0 = sum((1-gamma)*(x-m0)**2)/sum(1-gamma)
    s1 = sum((gamma*(x-m1)**2))/sum(gamma)
    
    p1 = sum(gamma)/M
    
    print "m =",m0,m1
    