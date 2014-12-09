
import numpy as np
import numpy.linalg as la
from ProblemGenerator import *
from MixtureGaussians import *
from EM import *


M=1000
N=4
K=4
x,p_sol,m_sol = generateProblem(M,N,K)


p,m,S = InitialIterate(N=N,K=K)
EM(x,theta=[p,m,S],solution=[p_sol,m_sol],maxit=1000)





  