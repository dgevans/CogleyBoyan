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
XRE = primitives.makeGrid(grid)
sHist = Para.lp.drawSample(1000,100,[mean(XRE[:,0]),mean(XRE[:,1])])[0]
print 'computing posteriors'
zgrid,qgrid,_ = Para.lp.getREgrid(norder)
sGrid = primitives.makeGrid((zgrid,qgrid))
my_samples = bayesian.computePosteriors(Para,sHist,sGrid,skip=10)
X = bayesian.getX(Para)
Vs = -10*ones(len(X))#VfRE(X[:,:3])#map(lambda x: VfRE(x[:3]),X)
Vf = primitives.ValueFunction(Para,['hermite','hermite','hermite','hermite'],[2,2,2,2],[1]*4,normalize=True)
Vf.fit(X,Vs)

Vf = bayesian.solveBellman(Para,Vf)
if rank == 0:
    fout = open('solution.dat','w')
    import cPickle
    cPickle.dump((Vf,sHist),fout)
    fout.close()