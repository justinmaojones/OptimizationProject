
from utils import AlgRuns
from utils import Problem
from utils import pickleLoad
import numpy.random as random
import numpy as np

import matplotlib.pyplot as plt
import prettyplotlib as ppl

#filename = 'RunDataK2' 
filename = "RunData2_K10_init3" 

       
#def RunAnalysis(filename):
RunDataList = pickleLoad(filename)
RunData = [run for run in RunDataList]
numinits = len(RunData)    

nummixtures = 2#RunData[0][0].num_mixtures
title = str(nummixtures)+' mixtures\n'

em_times = []
qn_times = []
em_conv = []
qn_conv = []
em_ll = []
qn_ll = []
em_k = []
qn_k = []
em_log = []
qn_log = []
em_rg = []
qn_rg = []

initnames = ['A','B','C','D','E']

for i in range(numinits):
    em_times.append(np.array([run.EM.time for run in RunData[i]]))
    qn_times.append(np.array([run.QN2.time for run in RunData[i]]))
    em_conv.append(np.array([run.EM.converged for run in RunData[i]]))
    qn_conv.append(np.array([run.QN2.converged for run in RunData[i]]))
    em_ll.append(np.array([run.EM.ll for run in RunData[i]]))
    qn_ll.append(np.array([run.QN2.ll for run in RunData[i]]))
    em_k.append(np.array([run.EM.k for run in RunData[i]]))
    qn_k.append(np.array([run.QN2.k for run in RunData[i]]))
    em_log.append([run.EM.Log for run in RunData[i]])
    qn_log.append([run.QN2.Log for run in RunData[i]])
    em_rg.append(np.array([run.EM.rg for run in RunData[i]]))
    qn_rg.append(np.array([run.QN2.rg for run in RunData[i]]))

time_comparisons = [em_times[i]<qn_times[i] for i in range(numinits)]
em_faster_percent_runs = round(100*np.concatenate(time_comparisons).mean())
fig, ax = plt.subplots(1)
for i in range(numinits):
    #ppl.scatter(ax,np.log(em_times[i]),np.log(qn_times[i]),label=initnames[i])
    ppl.scatter(ax,em_times[i],qn_times[i],label=initnames[i])
ax.set_xscale('log')
ax.set_yscale('log')
xlim = plt.xlim()
ylim = plt.ylim()
minrange = min(xlim[0],ylim[0])
maxrange = max(xlim[1],ylim[1])
linerange = np.arange(minrange,maxrange)
ppl.plot(linerange,linerange,'k--')
plt.title(title+"scatter of log runtimes")
plt.xlabel('EM runtime (s)')
plt.ylabel('QN2 runtime (s)')
plt.xlim(minrange,maxrange)
plt.ylim(minrange,maxrange)
plt.legend(loc='upper left')
plt.text(5,100,'EM faster\n%s%% of runs'%em_faster_percent_runs)
plt.text(100,11,'EM slower\n%s%% of runs'%(100-em_faster_percent_runs))
plt.show()

fig, ax = plt.subplots(1)
log_runtime_ratios = [np.log10(em_times[i]/qn_times[i]) for i in range(numinits)]
#runtime_ratios = [em_times[i]/qn_times[i] for i in range(numinits)]
labels=[initnames[i]+': log(EM time / QN2 time)' for i in range(numinits)]
ppl.hist(ax,log_runtime_ratios,label=labels)
#ppl.hist(ax,runtime_ratios,label=labels)
plt.title(title+"histogram of log runtime ratio EM/QN2")
plt.legend(loc='upper right',bbox_to_anchor=(1.1, 1))
plt.show()

converged = np.hstack([em_conv[i].reshape(-1,1) for i in range(numinits)] 
                    + [qn_conv[i].reshape(-1,1) for i in range(numinits)]) 
converged_sum = np.sum(converged,axis=0)
labels = ['EM '+initnames[i] for i in range(numinits)] + ['QN2 '+initnames[i] for i in range(numinits)]
fig, ax = plt.subplots(1)
ppl.bar(ax,np.arange(numinits*2),converged_sum,xticklabels=labels,annotate=True)
plt.title(title+"% converged to stationary point")
plt.legend()
plt.show()

loglik = np.hstack([em_ll[i].reshape(-1,1) for i in range(numinits)] 
                    + [qn_ll[i].reshape(-1,1) for i in range(numinits)]) 

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
labels = ['EM '+initnames[i] for i in range(numinits)] + ['QN2 '+initnames[i] for i in range(numinits)]
fig, ax = plt.subplots(1)
ppl.bar(ax,np.arange(numinits*2),loglikdiff_ismax_sum,xticklabels=labels,annotate=True)
plt.title(title+"is maximum of log likelihoods\n(not necessarily local max)")
plt.ylim((0,110))
plt.legend()
plt.show()


def extractlogs_EM(log,numrecords=2):
    numiters = len(log)
    numrecordsreturn = min(numiters,numrecords)
    #thetas = np.vstack([np.array(line[1]) for line in log])
    rgs = np.array([line[2] for line in log])
    return rgs[-numrecordsreturn:]#,thetas[-numrecordsreturn:]
    
def extractlogs_QN2(log,numrecords=2):
    numiters = len(log)-2
    numrecordsreturn = min(numiters,numrecords)
    mylog = log[:]
    mylog.remove(mylog[-1])
    mylog.remove(mylog[0])
    #thetas = np.vstack([np.array(line[1]) for line in mylog])
    rgs = np.array([line[4] for line in mylog])
    return rgs[-numrecordsreturn:] #,thetas[-numrecordsreturn:]

def calc_convergence_rate(rgs,numrecords=2):
    n = min(numrecords,len(rgs))
    if n <= 1:
        return 1
    else:
        rg = rgs[-n:]
        r = []
        for i in range(n-1):
            f1 = np.abs(rg[i])
            f2 = np.abs(rg[i+1])
            r.append(np.log(f2)/np.log(f1))
        return np.mean(np.array(r))

em_rate=[]
qn_rate=[]
for r in range(numinits):
    em_rate.append(np.array([calc_convergence_rate(extractlogs_EM(log)) for log in em_log[r]]))
    qn_rate.append(np.array([calc_convergence_rate(extractlogs_QN2(log)) for log in qn_log[r]]))
    em_rate[r] = em_rate[r][converged_overall]
    qn_rate[r] = qn_rate[r][converged_overall]
    em_rate[r] = em_rate[r][
                    np.logical_not(
                        np.logical_or(
                            np.isinf(em_rate[r]),
                            np.isnan(em_rate[r])))]
    qn_rate[r] = qn_rate[r][
                    np.logical_not(
                        np.logical_or(
                            np.isinf(qn_rate[r]),
                            np.isnan(qn_rate[r])))]
    em_rate[r] = em_rate[r][em_rate[r]>=0]
    qn_rate[r] = qn_rate[r][qn_rate[r]>=0]
    em_rate[r] = em_rate[r][em_rate[r]<2]
    qn_rate[r] = qn_rate[r][qn_rate[r]<2]
fig, ax = plt.subplots(1)
#labels=[initnames[i]+': EM' for i in range(numinits)]+[initnames[i]+': QN2' for i in range(numinits)]
labels = ['EM','QN2']
ppl.hist(ax,[np.concatenate(em_rate),np.concatenate(qn_rate)],label=labels)
plt.title(title+"histogram of convergence rates")
plt.legend()
plt.show()


fig, ax = plt.subplots(1)
for i in range(numinits):
    em_k[i][em_k[i]==0] = 1
    qn_k[i][qn_k[i]==0] = 1
em_time_k_ratio = [np.log(em_times[i]/em_k[i]*1e6) for i in range(numinits)]
qn_time_k_ratio = [np.log(qn_times[i]/qn_k[i]*1e6) for i in range(numinits)]
#labels=[initnames[i]+': EM' for i in range(numinits)]+[initnames[i]+': QN2' for i in range(numinits)]
labels = ['EM','QN2']
bins = np.arange(0,15,0.5)
ppl.hist(ax,[np.concatenate(em_time_k_ratio),np.concatenate(qn_time_k_ratio)],bins=bins,label=labels)
plt.title(title+"histogram of log(nanoseconds per iteration)")
plt.legend()
plt.show()

fig, ax = plt.subplots(1)
for i in range(numinits):
    em_k[i][em_k[i]==0] = 1
    qn_k[i][qn_k[i]==0] = 1
em_time_k_ratio = [em_times[i]/em_k[i] for i in range(numinits)]
qn_time_k_ratio = [qn_times[i]/qn_k[i] for i in range(numinits)]
em_qn_time_k_ratio = [em_time_k_ratio[i]/qn_time_k_ratio[i] for i in range(numinits)]
#labels=[initnames[i]+': EM' for i in range(numinits)]+[initnames[i]+': QN2' for i in range(numinits)]
labels = ['A','B','C']
bins = np.arange(0,15,0.5)
ppl.hist(ax,np.concatenate(em_qn_time_k_ratio))#,label=labels)
plt.title(title+"histogram of ratio of\nEM seconds/iteration to QN2 seconds/iteration")
plt.legend()
plt.show()

#RunAnalysis(filename)


