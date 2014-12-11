import PoissonQN2_MultiMixture as qn2


import numpy.random as random
import numpy as np

def problem_generator(num_mixtures=2,num_samples=3000,exp_lambda=2.0,inittype=1):
    lambdas = random.exponential(exp_lambda,num_mixtures)
    samples = random.poisson(lam=lambdas,size=(num_samples,num_mixtures))
    
    pvals = random.uniform(0,1,num_mixtures)
    pvals = pvals/np.sum(pvals)
    selector = random.multinomial(n=1,pvals=pvals,size=num_samples)
    
    selected = np.sum(selector*samples,axis=1)
    
    maxval = max(selected)
    J = np.arange(maxval+1)
    c = np.sum(selected.reshape(-1,1) == J,axis=0)
    
    if inittype == 1:
        gammas0 = [1.0*(1.0+i)/sum(range(1,num_mixtures+1)) for i in range(num_mixtures)]
        thetas0 = range(1,num_mixtures+1)
    else:
        gammas0 = pvals.tolist()
        thetas0 = lambdas.tolist()
    return c,lambdas,pvals,gammas0,thetas0



c,lambdas,pvals,gammas0,thetas0 = problem_generator(
                                    num_mixtures=3,
                                    num_samples=3000,
                                    exp_lambda=5.0,
                                    inittype=1)

print "pvals =  ",np.round(pvals,2)
print "lambdas =",np.round(lambdas,2)


print "############################"
print "EM Results"
(em_gammas,em_thetas,k,rg1,Log1,converged),_ = qn2.EM_(
                                c,gammas0,thetas0,
                                maxit=100,
                                printing=False)
print k,"iterations"
print "gammas =",np.round(np.array(em_gammas),2)
print "thetas =",np.round(np.array(em_thetas),2)
print "merit  =",rg1


print "############################"
print "QN Results"
(qn_gammas,qn_thetas,k,rg2,Log2,converged),_ = qn2.QN2(
                                c,gammas0,thetas0,
                                maxit=20,
                                printing=True)
print k,"iterations"
print "gammas =",np.round(np.array(qn_gammas),2)
print "thetas =",np.round(np.array(qn_thetas),2)
print "merit  =",rg2

def printlast20(Log):
    last20 = Log[-20:]
    for line in last20:
        for item in line:
            print item,
        print '\n',

#printlast20(Log2)
    
def roughexample1():
    pvals =   [ 0.24,  0.74,  0.03]
    lambdas = [ 6.16,  0.04,  2.58]
    c = [2103,  100,   53,   87,   99,  136,  107,  105,   80,   60,   31,
         13,   15,    5,    4,    2]
    num_mixtures=3
    num_samples=3000
