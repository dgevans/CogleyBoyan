# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 15:01:15 2013

@author: dgevans
"""

from numpy import *

from mpi4py import MPI
import primitives
import pdb

w = MPI.COMM_WORLD

class BellmanMap(object):
    '''
    The bellman equation map for the rational expectations equilibrium
    '''
    def __init__(self,Para):
        '''
        Inits the bellman map from Para
        '''
        #store Para
        self.Para = Para
        #store the learning problem
        self.lp = Para.lp
        #store the size of the problem
        self.n = 2+self.lp.n_param
        
    def __call__(self,Vf):
        '''
        From a value function gives a new value function
        '''
        self.Vf = Vf
        return self.maximizeObjective
        
    def maximizeObjective(self,state):
        '''
        Maximizes the objective function for a given state param combo.
        '''
        s,mu = state
        #setup parameters        
        gamma = self.Para.gamma
        beta = self.Para.beta
        delta = self.Para.delta
        z,q= exp(s)
        #Choose optimal policy
        EV = mu.getEV(self.Vf)
        alpha = ( (1.-gamma) *beta*EV / q )**(-1.0/gamma)
        g = (z+q*(1.-delta))/(q+alpha)
        c = alpha*g
        V = c**(1.-gamma)/(1.-gamma)+beta*g**(1.-gamma)*EV
        return array([c,g]),V
        
def computePosteriors(Para,sHist,skip=1):
    '''
    Computes the posteriors across states 
    '''
    T,N,_ =  sHist.shape
    samples = primitives.makeGrid([arange(0,N),arange(2,T,skip)])
    #split up samples
    size = w.Get_size()
    rank = w.Get_rank()
    n = len(samples)
    m = n/size
    r = n%size
    my_samples = samples[rank*m+min(rank,r):(rank+1)*m+min(rank+1,r)]
    #constuct domain
    my_domain = []
    for sample in my_samples:
        i,t = sample
        s = sHist[t,i,:]
        mu = Para.lp.getPosterior(sHist[:t,i,:])
        my_domain.append((s,mu))
    Para.my_domain = my_domain
    return my_samples
    
def getX(Para):
    '''
    Computes a matrix of (z,q) and moments based on the samples drawn
    '''
    rank = w.Get_rank()
    #get each indivdual X
    my_X = []
    for state in Para.my_domain:
        s,mu = state
        my_X.append(hstack((s,mu.moments())))
    my_X = vstack(my_X)
    
    X = w.gather(my_X) 
    if rank == 0:
        X = vstack(X)
    
    return w.bcast(X)
    
def solveBellman(Para,Vf):
    '''
    Solves the bellman equation for the states computed need to run 
    computePosteriors before this.
    '''
    rank = w.Get_rank()
    X = getX(Para)
    T = BellmanMap(Para)
    Vs_old = Vf(X).flatten()
    
    diff = 1.
    while diff > 1e-5:
        Vnew = T(Vf)
        #my_Vs = hstack(map(lambda state:Vnew(state)[1],Para.my_domain))
        my_Vs = zeros(len(Para.my_domain))
        for i,state in enumerate(Para.my_domain):
            my_Vs[i] = Vnew(state)[1]
            if isnan(my_Vs[i]):
                pdb.set_trace()
        Vs = w.gather(my_Vs)
        if rank == 0:
            Vs = hstack(Vs)
        #Now fit the new value function
        Vf.fit(X,Vs)
        diff = max(abs((Vs-Vs_old)/Vs_old))
        if rank == 0:
            print diff
        Vs_old = Vs
        
    return Vf
        
    
        