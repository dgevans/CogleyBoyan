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
        
def computePosteriors(Para,s0,n = 1000):
    '''
    Computes the posteriors across states 
    '''
    #split up samples
    size = w.Get_size()
    rank = w.Get_rank()
    m = n/size
    r = n%size
    my_n = m
    if rank < r:
        my_n += 1
    #constuct domain
    my_domain = []
    for j in range(my_n):
        t = int(log(random.rand())/log(Para.beta))+1
        sHist = Para.lp.drawSample(1,t,s0)[0]
        s = sHist[t-1,0,:]
        mu = Para.lp.getPosterior(sHist[:,0,:])
        my_domain.append((s,mu))
    
    #Para.my_domain = primitives.makeGrid_generic((sGrid,my_posteriors))
    #for domain in Para.my_domain:
    #    s,mu = domain
    #    domain[1] = mu.set_state(s)
    #now reset my_domain to ch
    Para.my_domain = my_domain
    print len(Para.my_domain)

    
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
    diff = 100.
    diff_old = diff
    a = 0.05
    n_reset = 10
    while diff > 1e-4:
        Vs = iterateBellman(T,Vf,Para)
        #Now fit the new value function
        c_old = Vf.f.get_c()
        Vf.fit(X,Vs)
        Vf.f.set_c(a*Vf.f.get_c()+(1-a)*c_old)
        diff = max(abs((Vs-Vs_old)/Vs_old))/a
        if diff > diff_old and n_reset >9:
            a /= 2.
            n_reset = 0
        diff_old = diff
        n_reset += 1
        
        if rank == 0:
            print diff
        Vs_old = Vs
        
    return Vf
    
def iterateBellman(T,Vf,Para):
    '''
    '''
    rank = w.Get_rank()
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
    return w.bcast(Vs)
        
    
        