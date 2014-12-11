
import utils
import numpy.random as random
import numpy as np

import matplotlib.pyplot as plt
import prettyplotlib as ppl

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
        
        
def RunAnalysis(filename,nummixtures):
    RunData1,RunData2 = utils.pickleLoad(filename)
    
    title = str(nummixtures)+' mixtures\n'
    
    em1_times = np.array([run.EM.time for run in RunData1])
    qn1_times = np.array([run.QN2.time for run in RunData1])
    em1_conv = np.array([run.EM.converged for run in RunData1])
    qn1_conv = np.array([run.QN2.converged for run in RunData1])
    em1_ll = np.array([run.EM.ll for run in RunData1])
    qn1_ll = np.array([run.QN2.ll for run in RunData1])
    em1_k = np.array([run.EM.k for run in RunData1])
    qn1_k = np.array([run.QN2.k for run in RunData1])
    em2_times = np.array([run.EM.time for run in RunData2])
    qn2_times = np.array([run.QN2.time for run in RunData2])
    em2_conv = np.array([run.EM.converged for run in RunData2])
    qn2_conv = np.array([run.QN2.converged for run in RunData2])
    em2_ll = np.array([run.EM.ll for run in RunData2])
    qn2_ll = np.array([run.QN2.ll for run in RunData2])
    em2_k = np.array([run.EM.k for run in RunData2])
    qn2_k = np.array([run.QN2.k for run in RunData2])
    
    numruns = len(RunData1)
    
    em_faster_percent_runs = round(100*np.concatenate((em1_times < qn1_times,em2_times<qn2_times)).mean())
    fig, ax = plt.subplots(1)
    ppl.scatter(ax,np.log(em1_times),np.log(qn1_times),label='A')
    ppl.scatter(ax,np.log(em2_times),np.log(qn2_times),label='B')
    xlim = plt.xlim()
    ylim = plt.ylim()
    minrange = min(xlim[0],ylim[0])
    maxrange = max(xlim[1],ylim[1])
    linerange = np.arange(minrange,maxrange)
    ppl.plot(linerange,linerange,'k--',label='equal time divider')
    plt.title(title+"log runtimes")
    plt.xlabel('EM log runtime')
    plt.ylabel('QN2 log runtime')
    plt.xlim(minrange,maxrange)
    plt.ylim(minrange,maxrange)
    plt.legend(loc='upper left')
    plt.text(minrange+1,(maxrange+minrange)/2.0,'EM faster\n%s%% of runs'%em_faster_percent_runs)
    plt.text((maxrange+minrange)/2.0,minrange+1,'EM slower\n%s%% of runs'%(100-em_faster_percent_runs))
    plt.show()
    
    fig, ax = plt.subplots(1)
    log_runtime_ratios = [np.log(em1_times/qn1_times),np.log(em2_times/qn2_times)]
    labels=['A: log(EM time / QN2 time)','B: log(EM time / QN2 time)']
    ppl.hist(ax,log_runtime_ratios,label=labels)
    plt.title(title+"log runtime ratio EM/QN2")
    plt.legend()
    plt.show()
    
    converged = np.hstack((
                    em1_conv.reshape(-1,1),
                    qn1_conv.reshape(-1,1),
                    em2_conv.reshape(-1,1),
                    qn2_conv.reshape(-1,1))) 
    converged_sum = np.sum(converged,axis=0)
    labels = ['EM A','QN2 A','EM B','QN2 B']
    fig, ax = plt.subplots(1)
    ppl.bar(ax,np.arange(4),converged_sum,xticklabels=labels,annotate=True)
    plt.title(title+"% converged to stationary point")
    plt.legend()
    plt.show()
    
    
    loglik = np.hstack((
                    em1_ll.reshape(-1,1),
                    qn1_ll.reshape(-1,1),
                    em2_ll.reshape(-1,1),
                    qn2_ll.reshape(-1,1)))
    
    loglikmax = np.max(loglik,axis=1).reshape(-1,1)
    loglikdiff = loglikmax - loglik
    loglikdiff[np.abs(loglikdiff)<=1e-3] = 0
    
    converged_overall = np.prod(
                            [converged[:,j] for j in range(converged.shape[1])],
                            axis=0
                            )==1
    
    loglikdiff_converged = loglikdiff[converged_overall,:]
    loglikdiff_conv_nonzero = loglikdiff_converged[np.sum(np.abs(loglikdiff_converged),axis=1)>0]
    loglikdiff_conv_nonzero_list = [loglikdiff_conv_nonzero[:,j] for j in range(loglikdiff_conv_nonzero.shape[1])]
    
    
    
    loglikdiff_ismax = loglikdiff==0
    loglikdiff_ismax_sum = np.sum(loglikdiff_ismax,axis=0)
    labels = ['EM A','QN2 A','EM B','QN2 B']
    fig, ax = plt.subplots(1)
    ppl.bar(ax,np.arange(4),loglikdiff_ismax_sum,xticklabels=labels,annotate=True)
    plt.title(title+"is maximum of log likelihoods\n(not necessarily local max)")
    plt.ylim((0,110))
    plt.legend()
    plt.show()





RunAnalysis('RunDataK2',2)


