"""

"""
from numpy import *
cimport numpy as np
from scipy.linalg import sqrtm

cdef extern from "sparse_grid_hw.h":
    void gqu ( int n, double *x, double *w)
    int gqu_order ( int l )
    void gqn ( int n, double x[], double w[] )
    int gqn_order ( int l )
    void nwspgr ( void rule ( int n, double *x, double *w ), 
                 int rule_order ( int l ), int dim, int k, int r_size, int *s_size, 
                 double *nodes, double *weights )
    int nwspgr_size ( int rule_order ( int l ), int dim, int k )

cdef class sparse_integrator:
    '''
    Class holds nodes for sparse integration 
    '''
    cdef np.ndarray X
    cdef np.ndarray W
    
    def __init__(self,d,k):
        '''
        creates nodes for integration with 
        '''
        n = nwspgr_size( gqu_order, d, k );
        cdef int n2 = 0
        self.X = zeros((n,d))
        self.W = zeros(n)
        nwspgr ( gqu, gqu_order, d, k, n, &n2,<double*> self.X.data , <double*> self.W.data )
        self.X = resize(self.X,(n2,d))
        self.W = resize(self.W,(n2))
        
    def __call__(self,f,*args):
        '''
        Integrates f over the limits provided by *args
        '''
        d = self.X.shape[1]
        assert d == len(args)
        limits = vstack(args)
        l_limits = limits[:,0].reshape((1,d))
        diff_limits = (limits[:,1]-limits[:,0]).reshape(1,d)
        
        X = l_limits + diff_limits*self.X
        
        return self.W.dot(map(f,X))*prod(diff_limits)
        
cdef class sparse_integrator_normal:
    '''
    Class holds nodes for sparse integration 
    '''
    cdef np.ndarray X
    cdef np.ndarray W
    
    def __init__(self,d,k):
        '''
        creates nodes for integration with 
        '''
        n = nwspgr_size( gqu_order, d, k );
        cdef int n2 = 0
        self.X = zeros((n,d))
        self.W = zeros(n)
        nwspgr ( gqn, gqn_order, d, k, n, &n2,<double*> self.X.data , <double*> self.W.data )
        self.X = resize(self.X,(n2,d))
        self.W = resize(self.W,(n2))
        
    def __call__(self,f,mu,Sigma):
        '''
        Integrates f over the limits provided by *args
        '''
        d = self.X.shape[1]
        assert d == len(mu)
        assert (d,d) == Sigma.shape
        mu = mu.reshape((1,d))
        S = sqrtm(Sigma)
        
        X = mu + self.X.dot(S)
        
        fs = hstack(map(f,X))
        
        return self.W.dot(fs)
        
    def test(self):
        '''
        Returns X and W
        '''
        return self.X,self.W
        
    