import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

#FROM HEAD
def sigmoid(X):
    return 1/(1+np.exp(-X))

def sigmoid_temp(input,temperature):
    b = 1/temperature
    return sigmoid(input*b)

def inv_sigmoid(X):
    return -(np.log( (1-X)/X ))
    
def compute_sgn_rate(x,t):
    '''
    COMPUTES PSEUDO SIGN FUNCTION FOR RATE NEURON FORMULATION
    x is assumed to be an np.floatX of one dimension and one element 
    x should be the sum of h_t
    t is the threshoLd for the neuron
    '''
    x_2 = x + t
    x_3 = 1 if x_2 > 0 else 0

    return x_3

def compute_sgn_prob_rate(x,t,tmp):
    '''
    COMPUTES PSEUDO SIGN FUNCTION FOR RATE NEURON FORMULATION
    x is assumed to be an np.floatX of one dimension and one element 
    x should be the sum of h_t
    t is the threshoLd for the neuron
    '''
    x_2 = x + t
    x_3 = sigmoid_temp(x_2,tmp)
    x_4 = np.random.binomial(1,x_3,1)

    return x_4

#U Matrix of patterns first dim is num patterns 
def make_U_rate(n_p, n_n):
    '''
    RETURNS MATRIX OF PATTERNS
    n_p is number of patterns
    n_n is number of units/neuros
    '''
    X_a = np.random.randn(n_p,n_n)
    X = np.where(X_a > 0, 1, 0)
    return X

#U Matrix of anti0Ferromagnet
def make_U_antiferrous_rate(n_p,n_n):
    '''
    RETURNS MATRIX OF PATTERNS IN
    SHAPE OF ANTIFERRO MAGNET
    n_p is number of patterns
    n_n is number of units/neuros
    NOTE: THIS IS THE RATE VERSION WITH BINARY {0,1} UNITS
    '''
    X_a = np.array([1,0])
    n = int(n_n/2)
    X_b = np.hstack([X_a]*n)
    X = np.vstack([X_b]*n_p)
    return X  

def make_W_rate(m):
    '''
    RETURNS A HEBBIAN WT MATRIX
    m is matrix of patterns (see make_U)
    m_x is a transformed 2x-1 matrix
    mat_collect is the weight matrix (accumulates over Us)
    NOTE: FOR THE RATE MODEL WITH BINARY {0,1} NEURONS
    '''
    a = m.shape[0] #number of patterns
    b = m.shape[1] #number of nodes

    m_x = m.copy()
    m_x = m_x*2 - 1

    mat_collect = np.zeros((b,b))

    for i in range(0,a):
        mat_tmp = np.outer(m_x[i],m_x[i].transpose())
        #print('mat_tmp: ')
        #print(mat_tmp) 
        mat_tmp = (1/b) * mat_tmp
        #print('mat_tmp: ')
        #print(mat_tmp)
        mat_collect = mat_collect + mat_tmp
        #print('mat_collect')
        #print(mat_collect)
        
    
    return mat_collect

def make_T_rate(m):
    '''
    RETURNS A THRESHOLD VECTOR, EACH ELEMENT INDEX IS MAPPED TO 
    CORRESPONDING NEURON INDEX VIA U(m)
    m is matrix of patterns (see make_U)
    m_x is a transformed 2x-1 matrix
    f is the scaling factor
    '''
    a = m.shape[0] #number of patterns for scaling ts
    
    m_x = m.copy()
    m_x = m_x*2 - 1
    
    t_collect = m_x.sum(axis=0)
    weighted_t = t_collect*(1/a)
    
    return weighted_t

def sim_patt_U_rate_if(v1,wts,t,n_u):
    '''
    RETURNS A HISTORY OF A SIM OF ONE PATTERN
    AND OVERWRITES S[i] n_u TIMES BY v1
    v1 is the initialization or input state vector
    wts is a weight vector between elements of v1
    t is a threshold vector, one per unit (indexed as expected)
    n_u is number of updates to the system; in i for loop below
    
    *GENERATED BELOW:
       n_n is the number of units/neuros in v1
       v2 is the ordered index of v1
       The i loop:
           Is the loop over the number of random unit updates
           Where S_i is a single neuron updates
           The j loop:
               Is the computation of h_t for a single neuron update
               In_j is computing one interaction i <- j between two
               neurons, h_t is the sum across j.
               
    '''
    n_n = len(v1)
    v2 = np.arange(n_n)
    S_hist = v1.copy()

    for i in np.random.randint(0,n_n,n_u):
        #print("SHISTLOOP, i:",i)
        S_i = v1[i].copy()
        
        h_t = np.zeros(1)
    
        for j in v2[np.logical_not(np.in1d(v2,i))]:
        
            In_j = wts[i,j]*v1[j]
            h_t = np.append(h_t,In_j)
        
        v1[i] = compute_sgn_rate(h_t.sum(),t[i])
        
        S_hist = np.vstack((S_hist,v1))
        
    return S_hist

def sim_patt_U_rate_hb(v1,wts,t,n_u):
    '''
    RETURNS A HISTORY OF A SIM OF ONE PATTERN
    AND OVERWRITES S[i] n_u TIMES BY v1
    v1 is the initialization or input state vector
    wts is a weight vector between elements of v1
    t is a threshold vector, one per unit (indexed as expected)
    n_u is number of updates to the system; in i for loop below
    
    *GENERATED BELOW:
       n_n is the number of units/neuros in v1
       v2 is the ordered index of v1
       The i loop:
           Is the loop over the number of random unit updates
           Where S_i is a single neuron updates
           The j loop:
               Is the computation of h_t for a single neuron update
               In_j is computing one interaction i <- j between two
               neurons, h_t is the sum across j.
               
    '''
    n_n = len(v1)
    v2 = np.arange(n_n)
    S_hist = v1.copy()

    for i in np.random.randint(0,n_n,n_u):
        #print("SHISTLOOP, i:",i)
        S_i = v1[i].copy()
        
        h_t = np.zeros(1)
    
        for j in v2[np.logical_not(np.in1d(v2,i))]:
        
            In_j = wts[i,j]*v1[j]
            h_t = np.append(h_t,In_j)
        
        v1[i] = compute_sgn_rate(h_t.sum(),t[i])
        
        S_hist = np.vstack((S_hist,v1))
        
    return S_hist

def sim_patt_U_prob_rate_if(v1,wts,t,n_u,tmp):
    '''
    RETURNS A HISTORY OF A SIM OF ONE PATTERN
    AND OVERWRITES S[i] n_u TIMES BY v1
    v1 is the initialization or input state vector
    wts is a weight vector between elements of v1
    t is a threshold vector, one per unit (indexed as expected)
    n_u is number of updates to the system; in i for loop below
    tmp is the temperature for the logistic  
    *GENERATED BELOW:
       n_n is the number of units/neuros in v1
       v2 is the ordered index of v1
       The i loop:
           Is the loop over the number of random unit updates
           Where S_i is a single neuron updates
           The j loop:
               Is the computation of h_t for a single neuron update
               In_j is computing one interaction i <- j between two
               neurons, h_t is the sum across j.
               
    '''
    n_n = len(v1)
    v2 = np.arange(n_n)
    S_hist = v1.copy()

    for i in np.random.randint(0,n_n,n_u):
        #print("SHISTLOOP, i:",i)
        S_i = v1[i].copy()
        
        h_t = np.zeros(1)
    
        for j in v2[np.logical_not(np.in1d(v2,i))]:
        
            In_j = wts[i,j]*v1[j]
            h_t = np.append(h_t,In_j)
        
        v1[i] = compute_sgn_prob_rate(h_t.sum(),t[i],tmp)
        
        S_hist = np.vstack((S_hist,v1))
        
    return S_hist

def sim_patt_U_prob_rate_hb(v1,wts,t,n_u,tmp):
    '''
    RETURNS A HISTORY OF A SIM OF ONE PATTERN
    AND OVERWRITES S[i] n_u TIMES BY v1
    v1 is the initialization or input state vector
    wts is a weight vector between elements of v1
    t is a threshold vector, one per unit (indexed as expected)
    n_u is number of updates to the system; in i for loop below
    tmp is the temperature for the logistic function    
    *GENERATED BELOW:
       n_n is the number of units/neuros in v1
       v2 is the ordered index of v1
       The i loop:
           Is the loop over the number of random unit updates
           Where S_i is a single neuron updates
           The j loop:
               Is the computation of h_t for a single neuron update
               In_j is computing one interaction i <- j between two
               neurons, h_t is the sum across j.
               
    '''
    n_n = len(v1)
    v2 = np.arange(n_n)
    S_hist = v1.copy()

    for i in np.random.randint(0,n_n,n_u):
        #print("SHISTLOOP, i:",i)
        S_i = v1[i].copy()
        
        h_t = np.zeros(1)
    
        for j in v2[np.logical_not(np.in1d(v2,i))]:
        
            In_j = wts[i,j]*v1[j]
            h_t = np.append(h_t,In_j)
        
        v1[i] = compute_sgn_prob_rate(h_t.sum(),t[i],tmp)
        
        S_hist = np.vstack((S_hist,v1))
        
    return S_hist

def energy(v1,wts,t):
    '''
    RETURNS THE ENERGY FOR A VECTOR OF STATES 
    GIVEN THE WEIGHTS AND THRESHOLDS
    v1 = state vector
    wts = is the weight matrix
    t = the threshold vector
    '''
    #print('v1: ', v1)
    mat_outer = np.outer(v1,v1.transpose())
    #print('mat_outer: ')
    #print(mat_outer)
    #print(mat_outer.shape)
    
    mat_wts = wts.copy()
    print('matwts: ')
    print(mat_wts)

    mat_e_js = mat_outer*mat_wts
    print('mat_e_js:')
    print(mat_e_js)
    np.fill_diagonal(mat_e_js,0)
    print('mat_e_js with FILL')
    print(mat_e_js)
    
    mat_e_js_flat = mat_e_js.flatten()
    print('len mat_e_js_flat: ',len(mat_e_js_flat))
    print('shape mat_e_js_flat: ',mat_e_js_flat.shape)
    print('mat_e_js_flat')
    print(mat_e_js_flat)
    
    mat_e_t = v1*t
    print("matt_e_t")
    print(mat_e_t)

    mat_return = mat_e_js_flat.sum() + mat_e_t.sum()
    print("mat_return")
    print(mat_return)
    
    return mat_return
    
    
#EOF
