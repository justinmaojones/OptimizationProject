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
    
    return c,lambdas,pvals

def initialize(pvals,lambdas,inittype=1):
    num_mixtures = len(pvals)
    if inittype == 1:
        gammas0 = [1.0*(1.0+i)/sum(range(1,num_mixtures+1)) for i in range(num_mixtures)]
        thetas0 = range(1,num_mixtures+1)
    elif inittype ==2:
        gammas0 = pvals.tolist()
        thetas0 = lambdas.tolist()
    else:
        gammas0 = [1.0/num_mixtures]*num_mixtures
        thetas0 = [1.0]*num_mixtures
    return gammas0,thetas0

def printlast20(Log):
    last20 = Log[-20:]
    for line in last20:
        for item in line:
            print item,
        print '\n',
    

class AlgRuns():
    def __init__(self, params):
        self.gammas = params[0][0]
        self.thetas = params[0][1]
        self.k = params[0][2]
        self.rg = params[0][3]
        #self.Log = params[0][4]
        self.converged = params[0][5]
        self.ll = params[0][6]
        self.time = params[1]
        
class Problem():
    def __init__(self, params,gammas0,thetas0):
        self.c = params[0]
        self.lambdas = params[1]
        self.pvals = params[2]
        self.gammas0 = gammas0
        self.thetas0 = thetas0
        
    def init_EM(self,params):
        self.EM = AlgRuns(params)
    
    def init_QN2(self,params):
        self.QN2 = AlgRuns(params)
        
        

RunData1 = []
RunData2 = []
RunData3 = []
num_mixtures = 5
num_samples = 30000
exp_lambda = 10.0
T = 100

for t in range(T):
    print "Begin t =",t
    print "   RUN 1",
    params_init = problem_generator(
                        num_mixtures=num_mixtures,
                        num_samples=num_samples,
                        exp_lambda=exp_lambda,
                        inittype=1)
    
    c,lambdas,pvals = params_init
    gammas0,thetas0 = initialize(lambdas,pvals,inittype=1)
    problem = Problem(params_init,gammas0,thetas0) 
    
    params_EM = qn2.EM_(
                    c,gammas0,thetas0,
                    maxit=1e6,
                    ftol = 1e-6,
                    merit_type = 'em',
                    printing=False)
    print "EM",
    
    params_QN2 = qn2.QN2(
                    c,gammas0,thetas0,
                    maxit=1e4,
                    ftol = 1e-6,
                    merit_type = 'rg',
                    printing=False)
    print "QN2"
    
    problem.init_EM(params_EM)
    problem.init_QN2(params_QN2)
    RunData1.append(problem)
    print "   EM",problem.EM.converged,problem.EM.k,problem.EM.time
    print "   QN",problem.QN2.converged,problem.QN2.k,problem.QN2.time
    
    print "   RUN 2",
    c,lambdas,pvals = params_init
    gammas0,thetas0 = initialize(lambdas,pvals,inittype=2)
    problem = Problem(params_init,gammas0,thetas0)
    
    params_EM = qn2.EM_(
                    c,gammas0,thetas0,
                    maxit=1e6,
                    ftol = 1e-6,
                    merit_type = 'em',
                    printing=False)
    print "EM",
    
    params_QN2 = qn2.QN2(
                    c,gammas0,thetas0,
                    maxit=1e4,
                    ftol = 1e-6,
                    merit_type = 'rg',
                    printing=False)
    print "QN2"
    
    problem.init_EM(params_EM)
    problem.init_QN2(params_QN2)
    RunData2.append(problem)
    print "   EM",problem.EM.converged,problem.EM.k,problem.EM.time
    print "   QN",problem.QN2.converged,problem.QN2.k,problem.QN2.time
    
    
    print "   RUN 3",
    c,lambdas,pvals = params_init
    gammas0,thetas0 = initialize(lambdas,pvals,inittype=3)
    problem = Problem(params_init,gammas0,thetas0)
    
    params_EM = qn2.EM_(
                    c,gammas0,thetas0,
                    maxit=1e6,
                    ftol = 1e-6,
                    merit_type = 'em',
                    printing=False)
    print "EM",
    
    params_QN2 = qn2.QN2(
                    c,gammas0,thetas0,
                    maxit=1e4,
                    ftol = 1e-6,
                    merit_type = 'rg',
                    printing=False)
    print "QN2"
    
    problem.init_EM(params_EM)
    problem.init_QN2(params_QN2)
    RunData3.append(problem)
    print "   EM",problem.EM.converged,problem.EM.k,problem.EM.time
    print "   QN",problem.QN2.converged,problem.QN2.k,problem.QN2.time
    





