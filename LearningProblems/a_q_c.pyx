# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 15:50:39 2013

@author: dgevans
"""
from numpy import *
cimport numpy as np
from scipy import stats
from scipy import integrate
from sparse_integrator import sparse_integrator
from sparse_integrator import sparse_integrator_normal

cdef extern from "gsl/gsl_randist.h":
    double gsl_ran_gaussian_pdf(double x,double sigma)

n_param = 1

integrate_order = 5

#Holds the initial mean and variance for the prior
a_q_bar0 = -0.015#0.1811
a_q_sigma0 = 0.001#0.005#0.1020

#other parameters for the learning problem
rho_q = 0.95
a_z = 0.0
rho_z = 0.95
sigma_q = 0.025#0.2878
sigma_z = 0.025

Integrate2d = sparse_integrator(2,integrate_order)
Integrate3d = sparse_integrator_normal(3,integrate_order)

def set_integrate_order(i_o):
    global integrate_order,Integrate2d,Integrate3d
    integrate_order = i_o
    Integrate2d = sparse_integrator(2,integrate_order)
    Integrate3d = sparse_integrator_normal(3,integrate_order)



cdef class posterior_distribution:
    '''
    Holds the posterior distribution, moments of that distribution and functions
    holding the moments of the period ahead distriubtion
    '''
    cdef double a_q_bar,a_q_sigma,q,z
    cdef np.ndarray state_moments #placeholder so don't keep reallocating during integration
    
    def __init__(self,a_q_bar,a_q_sigma,state):
        '''
        Initialize with mean and standard deviation
        '''
        self.a_q_bar = a_q_bar
        self.a_q_sigma = a_q_sigma
        self.z = state[0]
        self.q = state[1]
        self.state_moments = zeros(4)
        
    def set_state(self,state):
        return posterior_distribution(self.a_q_bar,self.a_q_sigma,state)
        
    def __call__(self,a_q):
        '''
        Returns the density at a_q
        '''
        return stats.norm(self.a_q_bar,self.a_q_sigma).pdf(a_q)
        
        
    cdef moments_prime(self,double qprime):
        '''
        Fast function for computing tomorrows moments
        '''
        cdef double S0,S,a_q_barprime
        S0 = self.a_q_sigma**2
        S = 1./(S0**(-1)+ 1./sigma_q**2)
        a_q_barprime = S*(S0**(-1)*self.a_q_bar + qprime/sigma_q**2)
        return [a_q_barprime,sqrt(S)]
        
    cdef double posterior_likelihood(self,double a_q,double zprime,double qprime ):
        '''
        Fast function or computing the likelihood over a_q,zprime and qprime
        '''
        return gsl_ran_gaussian_pdf(a_q-self.a_q_bar,self.a_q_sigma)*likelihood_c(a_q,self.z,self.q,zprime,qprime)
        
    cdef double posterior_integrand(self,Vf,double a_q,double zprime,double qprime ):
        '''
        Fast function or computing the likelihood over a_q,zprime and qprime
        '''
        mom =  self.moments_prime(qprime)
        self.state_moments[0] = zprime
        self.state_moments[1] = qprime
        self.state_moments[2:] = mom
        return self.posterior_likelihood(a_q,zprime,qprime )*Vf(self.state_moments)
        
    cdef double posterior_integrand2(self,Vf,double a_q,double zprime,double qprime ):
        '''
        Fast function or computing the likelihood over a_q,zprime and qprime
        '''
        mom =  self.moments_prime(qprime)
        self.state_moments[0] = zprime
        self.state_moments[1] = qprime
        self.state_moments[2:] = mom
        return Vf(self.state_moments)
            
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
            S = 1./(S0**(-1)+ 1./sigma_q**2)
            a_q_barprime = S*(S0**(-1)*self.a_q_bar + qprime/sigma_q**2)
            return array([a_q_barprime,sqrt(S)])
            
    def getEV(self,Vf):
        '''
        Computes the expected value of the value function from the posterior distribution
        '''
        
        #a_q_limits,z_limits,q_limits = self.getEVlimits()

        #compute the integral
        #den = integrate_fixed3(lambda a_q,z,q: self.posterior_likelihood(a_q,z,q),a_q_limits,z_limits,q_limits)
        #num = integrate_fixed3(lambda a_q,z,q: self.posterior_integrand(Vf,a_q,z,q),a_q_limits,z_limits,q_limits)
        #den = Integrate3d(lambda x: self.posterior_likelihood(x[0],x[1],x[2]),a_q_limits,z_limits,q_limits)
        mu = array([self.a_q_bar,a_z+rho_z*self.z,self.a_q_bar+rho_q*self.q])
        a_q_var = self.a_q_sigma**2
        Sigma = array([[a_q_var,0,a_q_var],[0.,sigma_z,0.],[a_q_var,0.,a_q_var+sigma_q**2]])
        num = Integrate3d(lambda x: self.posterior_integrand2(Vf,x[0],x[1],x[2]),mu,Sigma)
        
        #return expected value
        return num
        
    def getEVlimits(self):
        '''
        Computes the limits for the integration
        '''
                #compute the lmits of integrations
        zgrid,qgrid,a_q_grid = getREgrid([2]*3)
        a_q_limits = [max(self.a_q_bar-2*self.a_q_sigma,a_q_grid[0]),min(self.a_q_bar+2*self.a_q_sigma,a_q_grid[1])]
        mu_z = a_z+rho_z*self.z
        z_limits = [min(mu_z-2*sigma_z,zgrid[0]),max(mu_z+2*sigma_z,zgrid[1])]
        mu_q_l = a_q_limits[0]+rho_q*self.q
        mu_q_u = a_q_limits[1]+rho_q*self.q
        q_limits = [min(mu_q_l-2*sigma_q,qgrid[0]),max(mu_q_u+2*sigma_q,qgrid[1])]   
        return a_q_limits,z_limits,q_limits

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
    
cdef double likelihood_c(double a_q,double z,double q,double zprime, double qprime):
    '''
    For a given parameter a_q and state today compute the density for the state 
    tomorrow, using gsl for and c
    '''
    cdef double mu_z,mu_q
    mu_z = a_z+rho_z*z
    mu_q = a_q+rho_q*q
    return gsl_ran_gaussian_pdf(zprime-mu_z,sigma_z)*gsl_ran_gaussian_pdf(qprime-mu_q,sigma_q)
    
    
def getEV_RE_accurate(a_q,state,Vf):
    '''
    Computes the rational expectations expectation of the value function
    '''
    #z,q = state
    cdef double z,q
    z = state[0]
    q = state[1]
    zgrid,qgrid,_ = getREgrid([2]*3)
    mu_z = a_z+rho_z*z
    z_limits = [max(mu_z-4*sigma_z,zgrid[0]),min(mu_z+4*sigma_z,zgrid[1])]
    mu_q = a_q+rho_q*q
    q_limits = [max(mu_q-4*sigma_q,qgrid[0]),min(mu_q+4*sigma_q,qgrid[1])]


    den = integrate.dblquad(lambda zprime,qprime: likelihood_c(a_q,z,q,zprime,qprime),q_limits[0],q_limits[1]
                    ,lambda q:z_limits[0],lambda z: z_limits[1])[0]
                    
    state_param_prime = zeros(3)
    state_param_prime[2] = a_q
    def integrand(double zprime,double qprime):
        state_param_prime[:2] = zprime,qprime
        return likelihood_c(a_q,z,q,zprime,qprime)*Vf(state_param_prime)
        
    num = integrate.dblquad(integrand,q_limits[0],q_limits[1]
                    ,lambda q:z_limits[0],lambda z: z_limits[1])[0]
                    
    return num/den
    
def getEV_RE(a_q,state,Vf):
    '''
    Computes the rational expectations expectation of the value function
    '''
    #z,q = state
    cdef double z,q
    z = state[0]
    q = state[1]
    zgrid,qgrid,_ = getREgrid([2]*3)
    mu_z = a_z+rho_z*z
    z_limits = [max(mu_z-4*sigma_z,zgrid[0]),min(mu_z+4*sigma_z,zgrid[1])]
    mu_q = a_q+rho_q*q
    q_limits = [max(mu_q-4*sigma_q,qgrid[0]),min(mu_q+4*sigma_q,qgrid[1])]


    #den = integrate.dblquad(lambda zprime,qprime: likelihood_c(a_q,z,q,zprime,qprime),q_limits[0],q_limits[1]
    #                ,lambda q:z_limits[0],lambda z: z_limits[1])[0]
    den = integrate_fixed(lambda zprime,qprime: likelihood_c(a_q,z,q,zprime,qprime),z_limits,q_limits)               
    state_param_prime = zeros(3)
    state_param_prime[2] = a_q
    def integrand(double zprime,double qprime):
        state_param_prime[:2] = zprime,qprime
        return likelihood_c(a_q,z,q,zprime,qprime)*Vf(state_param_prime)
        
    #num = integrate.dblquad(integrand,q_limits[0],q_limits[1]
    #                ,lambda q:z_limits[0],lambda z: z_limits[1])[0]
    num = integrate_fixed(integrand,z_limits,q_limits)                
    return num/den
    
def integrate_fixed(f,z_limit,q_limit):
    '''
    Do fixed order intergration
    '''
    def integrate_z(q):
        return integrate.fixed_quad(vectorize(lambda z: f(z,q)),z_limit[0],z_limit[1],n=integrate_order)[0]
    
    return integrate.fixed_quad(vectorize(integrate_z),q_limit[0],q_limit[1],n=integrate_order)[0]
    
def integrate_fixed3(f,a_q_limit,z_limit,q_limit):
    '''
    Do fixed order intergration
    '''
    def integrate_a_q(z,q):
        return integrate.fixed_quad(vectorize(lambda a_q: f(a_q,z,q)),a_q_limit[0],a_q_limit[1],n=integrate_order)[0]
        
    def integrate_z(q):
        return integrate.fixed_quad(vectorize(lambda z: integrate_a_q(z,q)),z_limit[0],z_limit[1],n=integrate_order)[0]
    
    return integrate.fixed_quad(vectorize(integrate_z),q_limit[0],q_limit[1],n=integrate_order)[0]
    
def getREgrid(ngrid):
    '''
    Computes a grid to solve the RE problem over
    '''
    a_q_grid = linspace(a_q_bar0-4*a_q_sigma0,a_q_bar0+4*a_q_sigma0,ngrid[2])
    sig_z = sigma_z/sqrt(1.-rho_z**2)
    sig_q = sigma_q/sqrt(1.-rho_q**2)
    z_grid = linspace(a_z/(1-rho_z)-4*sig_z,a_z/(1-rho_z)+4*sig_z,ngrid[0])  
    q_grid = linspace(a_q_grid.min()/(1-rho_q)-4*sig_q,a_q_grid.max()/(1-rho_z)+4*sig_q,ngrid[1])
    
    return[z_grid,q_grid,a_q_grid]
    
    
    
    
def getPosterior(stateHist):
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
    
    return posterior_distribution(a_q_bar,a_q_sigma,stateHist[t-1,:])
    
def drawSample(N,T,state0):
    '''
    Draws N time series of length T starting from state0
    '''
    a_q = a_q_bar0+a_q_sigma0*random.randn(N)#makes N draws from the prior

    #draw epsilon shocks
    epsilon = zeros((T-1,N,2))
    epsilon[:,:,0] = sigma_z*random.randn(T-1,N)
    epsilon[:,:,1] = sigma_q*random.randn(T-1,N)
    
    #initialize
    stateHist = zeros((T,N,2))
    stateHist[0,:,:] = state0
    #simulate
    for t in range(1,T):
        stateHist[t,:,0] = a_z + rho_z*stateHist[t-1,:,0] + epsilon[t-1,:,0]
        stateHist[t,:,1] = a_q + rho_q*stateHist[t-1,:,1] + epsilon[t-1,:,1]
        
    return stateHist,a_q
    
    
        