# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 15:50:39 2013

@author: dgevans
"""
from numpy import *
from scipy import stats
from scipy import integrate


n_param = 1

#Holds the initial mean and variance for the prior
a_q_bar0 = 0.1811
a_q_sigma0 = 0.02#0.1020

#other parameters for the learning problem
rho_q = 0.88
a_z = 0.0424
rho_z = 0.8899
sigma_q = 0.1#0.2878
sigma_z = 0.0122

class posterior_distribution:
    '''
    Holds the posterior distribution, moments of that distribution and functions
    holding the moments of the period ahead distriubtion
    '''
    def __init__(self,a_q_bar,a_q_sigma,q):
        '''
        Initialize with mean and standard deviation
        '''
        self.a_q_bar = a_q_bar
        self.a_q_sigma = a_q_sigma
        self.q = q
        self.density = True
        
    def __call__(self,a_q):
        '''
        Returns the density at a_q
        '''
        return stats.norm(a_q_bar,a_q_sigma).pdf(a_q)
        
    def moments(self,stateprime = None):
        '''
        Returns the moments of the distribution if stateprime == None.  Otherwise
        return the moments of the posterior after new realization stateprime
        '''
        if stateprime==None:
            return array([self.a_q_bar,self.a_q_sigma])
        else:
            _,qprime = stateprime
            S0 = self.a_q_sigma**2
            S = 1./(S0**(-1)+ (t-1)/sigma_q**2)
            a_q_barprime = S*(S0**(-1)*self.a_q_bar + qprime/sigma_q**2)
            return array([a_q_barprime,sqrt(S)])
                

def prior(a_q):
    '''
    Given the prior parameters return the density.
    '''
    return stats.norm(a_q_bar0,a_q_sigma0).pdf(a_q)
    
def likelihood(a_q,state,stateprime):
    '''
    For a given parameter a_q and state today compute the density for the state 
    tomorrow
    '''
    z,q = state
    zprime,qprime = stateprime
    return stats.norm(a_z+rho_z*z,sigma_z).pdf(zprime)*\
    stats.norm(a_q+rho_q*q,sigma_q).pdf(qprime)
    
    
def getEV_RE(a_q,state,Vf):
    '''
    Computes the rational expectations expectation of the value function
    '''
    z,q = state
    mu_z = a_z+rho_z*z
    z_limits = [mu_z-4*sigma_z,mu_z+4*sigma_z]
    mu_q = a_q+rho_q*q
    q_limits = [mu_q-4*sigma_q,mu_q+4*sigma_q]

    #den = integrate.dblquad(lambda zprime,qprime: likelihood(a_q,state,[zprime,qprime]),q_limits[0],q_limits[1]
    #                ,lambda q:z_limits[0],lambda z: z_limits[1])[0]              
    num = integrate.dblquad(lambda zprime,qprime: likelihood(a_q,state,[zprime,qprime])*Vf(hstack((zprime,qprime,a_q))),q_limits[0],q_limits[1]
                    ,lambda q:z_limits[0],lambda z: z_limits[1])[0]
                    
    return num
    
def getREgrid(ngrid):
    '''
    Computes a grid to solve the RE problem over
    '''
    a_q_grid = linspace(a_q_bar0-2*a_q_sigma0,a_q_bar0+2*a_q_sigma0,ngrid[2])
    sig_z = sigma_z/sqrt(1.-rho_z**2)
    sig_q = sigma_q/sqrt(1.-rho_q**2)
    z_grid = linspace(a_z/(1-rho_z)-2*sig_z,a_z/(1-rho_z)+2*sig_z,ngrid[0])  
    q_grid = linspace(a_q_grid.min()/(1-rho_q)-2*sig_q,a_q_grid.max()/(1-rho_z)+2*sig_q,ngrid[1])
    
    return[z_grid,q_grid,a_q_grid]
    
    
    
    
def getPosterior(stateHist,moments):
    '''
    Given a history of states this returns a posterior.  Note to speed computational 
    time this posterior will be special.  Not only will it have moments precomputed but,
    will also store a function which returns moments for each possible state tomorrow
    '''
    qHist = stateHist[:,1]
    t = len(qHist)
    y = qHist[1:] - rho_q*qHist[:t-1]
    S0 = a_q_sigma0**2
    S = 1./(S0**(-1)+ (t-1)/sigma_q**2)
    a_q_sigma = sqrt(S)
    a_q_bar = S*(S0**(-1)*a_q_bar0 + sum(y)/sigma_q**2)
    
    return posterior_distribution(a_q_bar,a_q_sigma)
        