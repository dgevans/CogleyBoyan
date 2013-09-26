# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 17:48:13 2013

@author: dgevans
"""


import primitives

from numpy import *
import bayesian_log as bayesian
from IPython.parallel import Client

c = Client()
v = c[:]

Para = primitives.parameters()
Para.lp.a_q_sigma0 = 0.01
norder = [6,6,6]
v['Para'] = Para
v.execute('Para.lp.a_q_sigma0 = 0.01')

grid = Para.lp.getREgrid(array(norder))

print 'computing posteriors'
s0 =[mean(grid[0]),mean(grid[1])]
Para.domain = bayesian.computePosteriors(Para,s0,500)
X = bayesian.getX(Para)

Vf = primitives.ValueFunction([2,2,2,2],max_order = 2,normalize = True)
Vf.gamma = Para.gamma
Vf.beta = Para.beta
Vf.eta = 0.
#initialize the value functions
Vf.fit(X,-10.*ones(len(X)))

Vf = bayesian.solveBellman_parallel(Para,Vf)
    
fout = open('solution.dat','w')
import cPickle
cPickle.dump((Vf),fout)
fout.close()