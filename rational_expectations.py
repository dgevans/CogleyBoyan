# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 11:41:24 2013

@author: dgevans
"""
from numpy import *
from cpp_interpolator import *
import primitives
from copy import copy

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
        
    def maximizeObjective(self,state_param):
        '''
        Maximizes the objective function for a given state param combo.
        '''
        state = state_param[:2]
        param = state_param[2:]
        #setup parameters        
        gamma = self.Para.gamma
        beta = self.Para.beta
        delta = self.Para.delta
        z,q= state
        #Choose optimal policy
        EV = self.lp.getEV_RE(param,state,self.Vf)
        alpha = ( (1.-gamma) *beta*EV / q )**(-1.0/gamma)
        g = (z+q*(1.-delta))/(q+alpha)
        c = alpha*g
        V = c**(1.-gamma)/(1.-gamma)+beta*g**(1.-gamma)*EV
        return array([c,g]),V
        
        
def findRE(Para,norder=None):
    '''
    Computes the rational expectations equilibrium
    '''
    T = BellmanMap(Para)
    if norder ==None:
        norder = [5]*T.n
    types = ['spline']*T.n
    k = [3]*T.n
    
    #sets up the interpolation INFO
    #INFO = interpolate_INFO(types,norder,k)
    grid = Para.lp.getREgrid(array(norder))
    X = primitives.makeGrid(grid)
    Vf = primitives.ValueFunction(Para,types,norder,k)
    Vf.fit(X,-10*ones(len(X)))
    #interpolate(X,-10*ones(len(X)),INFO)  
    Vs_old = Vf(X).flatten()
    #solve bellman
    diff = 1
    while diff >0.02:
        Vs = hstack(map(lambda state: T(Vf)(state)[1],X))
        #Vfnew = interpolate(X,Vs,INFO)
        Vf.fit(X,Vs)
        diff = max(abs((Vs-Vs_old)/Vs_old))
        print diff
        Vs_old = Vs
    return X,Vf
    
    
        
                