# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 17:48:13 2013

@author: dgevans
"""

from mpi4py import MPI
import primitives

w = MPI.COMM_WORLD
import rational_expectations_log as re
from numpy import *
import bayesian_log as bayesian

rank = w.Get_rank()

Para = primitives.parameters()
Para.lp.a_q_sigma0 = 0.01
norder = [6,6,6]
Para.lp.set_integrate_order(4)
#XRE,VfRE = re.findRE(Para,norder)

#a_q_hat = linspace(min(XRE[:,2]),max(XRE[:,2]),100)
#for i in range(0,36):
#    Xhat = hstack((XRE[i,:2]*ones((100,2)),a_q_hat.reshape(-1,1)))
#    plot(a_q_hat,VfRE(Xhat))

#Now solve bayesian problem
grid = Para.lp.getREgrid(array(norder))
#sHist = Para.lp.drawSample(1000,1000,[mean(XRE[:,0]),mean(XRE[:,1])])[0]
print 'computing posteriors'
s0 =[mean(grid[0]),mean(grid[1])]
bayesian.computePosteriors(Para,s0,2000)
X = bayesian.getX(Para)
#Vs = VfRE(X[:,:3])#map(lambda x: VfRE(x[:3]),X)
primitives.ValueFunction.gamma = Para.gamma
primitives.ValueFunction.beta = Para.beta
primitives.ValueFunction.eta = 0.
Vf = primitives.ValueFunction(X,-10.*ones(len(X)),[3,3,3,3],max_order = 3,normalize = True)
Vf = bayesian.solveBellman(Para,Vf)

if rank == 0:
    fout = open('solution.dat','w')
    import cPickle
    cPickle.dump((Vf),fout)
    fout.close()