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
#Para.lp.a_q_sigma0 = 0.02
#Para.lp.a_q_bar0 =0.16
Para.lp.integrate_order = 4
norder = [6,6,6]
XRE,VfRE = re.findRE(Para,norder)

#a_q_hat = linspace(min(XRE[:,2]),max(XRE[:,2]),100)
#for i in range(0,36):
#    Xhat = hstack((XRE[i,:2]*ones((100,2)),a_q_hat.reshape(-1,1)))
#    plot(a_q_hat,VfRE(Xhat))

#Now solve bayesian problem
sHist = Para.lp.drawSample(15,1000,[0.3,1.])[0]
print 'computing posteriors'
my_samples = bayesian.computePosteriors(Para,sHist,skip=10)
X = bayesian.getX(Para)
Vs = VfRE(X[:,:3])#map(lambda x: VfRE(x[:3]),X)
Vf = primitives.ValueFunction(Para,['spline','spline','hermite','hermite'],[6,6,2,2],[2]*4)
Vf.fit(X,Vs)
Vf.fitV(Vs)

Vf = bayesian.solveBellman(Para,Vf)