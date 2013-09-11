# -*- coding: utf-8 -*-
"""
Created on Sun May 19 14:41:35 2013

@author: dgevans
"""
from LearningProblems import a_q_c
from numpy import *
from cpp_interpolator import *
from copy import copy
from scipy.linalg import sqrtm

class parameters(object):
    
    gamma = 2.0
    
    delta = 0.25
    
    beta  = 1.0/1.04
    
    lp = a_q_c
    
    
class ValueFunction(object):
    '''
    Holds the interpolated value functions
    '''
    eta = 1e-8
    def __init__(self,Para,types,order,k,normalize = False):
        '''
        Inits the value function object
        '''
        self.f = None
        self.INFO = interpolate_INFO(types,order,k)
        self.beta = Para.beta
        self.gamma = Para.gamma
        self.normalize = normalize
    
    def fit(self,X,V):
        '''
        Fits the constant consumption per capital equivalent
        '''
        Y = ((1.-self.beta)*(1.-self.gamma)*V)**(1./(1-self.gamma))
        if self.normalize:
            self.mu = mean(X,0).reshape(1,X.shape[1])
            self.S = linalg.inv(real(sqrtm(cov(X.T))))
            self.X = (X-self.mu).dot(self.S)
        else:
            self.X = X
        if self.eta > 0:
            self.f = interpolate(self.X,Y,self.INFO,self.eta)
        else:
            self.f = interpolate(self.X,Y,self.INFO)
    def fitV(self,V):
        '''
        Fits the constant consumption per capital equivalent
        '''
        Y = ((1.-self.beta)*(1.-self.gamma)*V)**(1./(1-self.gamma))
        Phi = self.f.get_Phi(self.X)
        A = (Phi.T.dot(Phi)-self.eta*eye(Phi.shape[1]))
        c = linalg.solve(A,Phi.T.dot(Y))
        self.f.set_c(c)
    def __call__(self,X):
        '''
        Returns the value function at the numpy array X
        '''
        if self.normalize:
            x = (X-self.mu).dot(self.S)
            x = vectorize(lambda z:min(max(z,-2.),2.))(x)
        else:
            x = X    
        return (self.f(x))**(1.-self.gamma)/((1-self.gamma)*(1.-self.beta))
    
    
 
        
def printResults(c,res):
    n=[0]* len(c.ids)
    import time
    import sys
    while not res.ready():
        time.sleep(0.1)
        c.spin()
        for j in range(0,len(n)):
            out = c.metadata[res.msg_ids[j]].stdout.split()
            for i in range(n[j],len(out)):
                print out[i]
                sys.stdout.flush()
                n[j] = len(out)

def makeGrid(x):
    '''
    Makes a grid to interpolate on from list of x's
    '''
    N = 1
    n = []
    n.append(N)
    for i in range(0,len(x)):
        N *= x[i].shape[0]
        n.append(N)
    X = zeros((N,len(x)))
    for i in range(0,N):
        temp = i
        for j in range(len(x)-1,-1,-1):
            X[i,j] = x[j][temp/n[j]]
            temp %= n[j]
    return X
    
def makeGrid_generic(x):
    '''
    Makes a grid to interpolate on from list of x's
    '''
    N = 1
    n = []
    n.append(N)
    for i in range(0,len(x)):
        N *= len(x[i])
        n.append(N)
    X =[]
    for i in range(0,N):
        temp = i
        temp_X = []
        for j in range(len(x)-1,-1,-1):
            temp_X.append(x[j][temp/n[j]])
            temp %= n[j]
        temp_X.reverse()
        X.append(temp_X)
    return X