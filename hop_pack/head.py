import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random


#HELPER FUNCS
def plot_hist(img):
    imgplot = plt.imshow(img,cmap="binary",aspect='auto') 
    plt.colorbar()  
    
def plot_catch(catch,b):
    plt.hist(catch,bins=b)

def ham_compare(u_i,U):
    '''
    RETURNS A VECTOR OF HAMMING DISTANCES BETWEEN A GIVEN VECTOR
    AND ALL THE VECTORS IN THE MATRIX
    u_i is a vector of states
    U is a matrix of u_is
    '''
    catch = np.array([])
    for i in range(0,len(U)):
        catch = np.append(catch,np.sum(np.bitwise_xor(u_i.astype(int),U[i].astype(int))))

    return catch


#STATE VECTOR i is a node

#U Matrix of patterns first dim is num patterns 
def make_U(n_p, n_n):
    '''
    RETURNS MATRIX OF PATTERNS
    n_p is number of patterns
    n_n is number of units/neuros
    '''
    X_a = np.random.randn(n_p,n_n)
    X = np.where(X_a > 0, 1, -1)
    return X

#U Matrix of anti0Ferromagnet
def make_U_antiferrous(n_p,n_n):
    '''
    RETURNS MATRIX OF PATTERNS IN
    SHAPE OF ANTIFERRO MAGNET
    n_p is number of patterns
    n_n is number of units/neuros
    '''
    X_a = np.array([1,-1])
    n = int(n_n/2)
    X_b = np.hstack([X_a]*n)
    X = np.vstack([X_b]*n_p)
    return X  

def make_W(m):
    '''
    RETURNS A HEBBIAN WT MATRIX
    m is matrix of patterns (see make_U)
    '''
    a = m.shape[0] #number of patterns
    b = m.shape[1] #number of nodes
    mat_collect = np.zeros((b,b))
                
    for i in range(0,a):
        mat_tmp = np.outer(m[i],m[i].transpose())
        mat_tmp = (1/b) * mat_tmp
        mat_collect = mat_collect + mat_tmp
    
    return mat_collect

def compute_M(v1,v2,a):
    '''
    RETURNS THE OVERLAP OF TWO VECTORS
    v1 is vector 1
    v2 is vector 2
    a is the number of elements in v1 or v2
    '''
    return np.dot(v1,v2)*(1/a)

def compute_sgn(x):
    '''
    COMPUTES SIGN FUNCTION WITH ZERO MAPPED TO 1
    x is assumed to be an np.floatX of one dimension and one element 
    x should be the sum of h_t
    '''
    x_2 = -1 if x < 0 else 1
        
    return x_2

def compute_p_sgn(x):
    '''
    COMPUTES SIGN FUNCTION WITH ZERO MAPPED TO 1
    x is assumed to be an np.floatX of one dimension and one element
    x should be the sum of h_t
    '''
    if x < 0:
        x_2 = -1
    elif x == 0:
        x_2 = random.choice([-1,1])
    else:
        x_2 = 1
        
    return x_2

def sim_patt_U(v1,wts,n_u):
    '''
    RETURNS A HISTORY OF A SIM OF ONE PATTERN
    AND OVERWRITES S[i] n_u TIMES BY v1
    v1 is the initialization or input state vector
    wts is a weight vector between elements of v1
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
        
        v1[i] = np.sign(h_t.sum())
        
        S_hist = np.vstack((S_hist,v1))
        
    return S_hist

def sim_patt_U_2(v1,wts,n_u):
    '''
    RETURNS A HISTORY OF A SIM OF ONE PATTERN
    AND OVERWRITES S[i] n_u TIMES BY v1
    v1 is the initialization or input state vector
    wts is a weight vector between elements of v1
    n_u is number of updates to the system; in i for loop below
    NOTE: THIS CALLS CUSTOM SIGN FUNCTION
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
        
        #v1[i] = np.sign(h_t.sum())
        v1[i] = compute_sgn(h_t.sum())
        
        S_hist = np.vstack((S_hist,v1))
        
    return S_hist

def sim_patt_U_3(v1,wts,n_u):
    '''
    RETURNS A HISTORY OF A SIM OF ONE PATTERN
    AND OVERWRITES S[i] n_u TIMES BY v1
    v1 is the initialization or input state vector
    wts is a weight vector between elements of v1
    n_u is number of updates to the system; in i for loop below
    NOTE: THIS CALLS CUSTOM PROBALISTIC SIGN FUNCTION
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
        
        #v1[i] = np.sign(h_t.sum())
        v1[i] = compute_p_sgn(h_t.sum())
        
        S_hist = np.vstack((S_hist,v1))
        
    return S_hist


#END OF FILE
