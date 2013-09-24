# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 20:33:34 2013

@author: dgevans
"""
import numpy as np
from numpy.polynomial.hermite import hermval

class interpolate(object):
    '''
    Interpolates data using hermite polynomials
    '''
    def __init__(self,X,y,order,eta=0.,max_order = None):
        '''
        Fits data X,y using hermite polynomials given by order.  Uses eta for stability
        and max_order to remove high order cross terms
        '''
        #first check that values passed are sane
        X = np.atleast_2d(X)
        if X.shape[1] != len(order):
            X = X.T
        assert len(X) == len(y)
        assert X.shape[1] == len(order)
        #build basis matrix
        self.coef = self.buildBasisMatrix(order,max_order)
        self.eta = eta
        self.N = len(order)
        #apply basis functions to the domain X
        self.fit(X,y)
        
    def fit(self,X,y):
        '''
        Fits the data usng the bais functions
        '''
        #first get Phi matrix bt evaluating basis functions allong each row of X
        Phi = np.vstack(map(lambda x:np.prod(hermval(x,self.coef,False),1),X))
        if self.eta == 0.:
            self.c = np.linalg.lstsq(Phi,y)[0]
        else:
            A = Phi.T.dot(Phi)
            A -= self.eta*np.eye(len(A))
            b = Phi.T.dot(y)
            self.c = np.linalg.solve(A,b)
            
    def __call__(self,X):
        '''
        Evaluates the fit at x
        '''
        X = np.atleast_2d(X)
        if X.shape[1] != self.N:
            X = X.T
        assert X.shape[1] == self.N
        #evaluate basis fuctionslong each rowof X
        B = np.vstack(map(lambda x:np.prod(hermval(x,self.coef,False),1),X))
        return B.dot(self.c)
        
    def get_c(self):
        '''
        Gets the coefficients associated with the fit
        '''
        return self.c
        
    def set_c(self,c):
        '''
        Sets the coefficients associate with the fit
        '''
        self.c = c
        
    @staticmethod
    def buildBasisMatrix(order,max_order):
        '''
        Builds the matrix containing basis combinations for each dimension of X
        '''
        N = len(order)
        orders = []
        for i,d in enumerate(order):
            orders.append(np.arange(d+1))
        basis_orders = np.array(makeGrid_generic(orders))
        if not max_order == None:
            basis_orders_sum = np.sum(basis_orders,1)
            basis_orders = basis_orders[basis_orders_sum <= max_order]
        
        #first dimension is the maximum order hermite polynomial used, second
        #is the number of basis functions last is the dimension
        coef = np.zeros((np.amax(basis_orders)+1,len(basis_orders),N))
        for ib in range(len(basis_orders)):
            for i in range(N):
                coef[basis_orders[ib,i],ib,i] = 1. 
        return coef
                
        
        
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
    
def buildPolyMatrix(order,max_order):
        '''
        Builds the matrix containing basis combinations for each dimension of X
        '''
        orders = []
        for i,d in enumerate(order):
            orders.append(np.arange(d+1))
        coeff = np.array(makeGrid_generic(orders))
        if max_order == None:
            return  coeff.T
        else:
            coeff_sum = np.sum(coeff,1)
            return coeff[coeff_sum <= max_order].T