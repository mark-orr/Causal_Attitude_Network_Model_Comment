import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


#WRITE U TO CSV FOR R
def flip_bit_array(x):
    '''
    RETURNS FLIPPED STATES FOR AN ARRAY
    FOR ALL ELEMENTS IN THE ARRAY
    ASSUMES {-1,1} only in the array.
    x is an np.array of arbitrary dimension
    '''
    y = np.where(x==-1,1,-1)
    
    return y

def flip_bit_array_rate(x):
    '''
    RETURNS FLIPPED STATES FOR AN ARRAY
    FOR ALL ELEMENTS IN THE ARRAY
    ASSUMES {-1,1} only in the array.
    x is an np.array of arbitrary dimension
    '''
    y = np.where(x==0,1,0)
    
    return y


def write_for_R_wnoise(self,nl):
    '''
    WARNING****NOT WELL TESTED*****
    RETURNS NOTHING
    MAKES A .CSV, X.csv OF A MATRIX AND CONVERTS ALL NUMBERS TO 
    [0,1]
    self is the object passed in (a np.array)
    nl is the noise level (min/max 0/1)
    replicates
    
    
    
    '''
    nl_int = round(self.size*nl) 
    
    #ORIGINAL ARRAY
    X = self.astype(np.int32)            
    
    
    flip_index = np.random.choice(X.size,nl_int,replace=False)
    print('flip index: ',flip_index)
    X.put(flip_index,flip_bit_array(X))
    
    #COMPLIMENT ARRAY
    Y = self.astype(np.int32)
    Y.put(range(0,Y.size),flip_bit_array(Y)) #NOW ITS A COMPLIMENT
    
    flip_index = np.random.choice(Y.size,nl_int,replace=False)
    print('flip index: ',flip_index)
    Y.put(flip_index,flip_bit_array(Y))
    
    X_Y = np.vstack((X,Y))
    X_Y_x = np.where(X_Y < 0, 0, 1)
    
    np.savetxt(f'Graphs/X_nl{nl}.csv',X_Y_x,delimiter=',')
    return None

def write_for_R_wnoise_negpos1(self,nl):
    '''
    WARNING ****NOT WELL TESTED*****
    RETURNS NOTHING
    MAKES A .CSV, X.csv OF A MATRIX AND CONVERTS ALL NUMBERS TO 
    [0,1]
    self is the object passed in (a np.array)
    nl is the noise level (min/max 0/1)
    replicates
    
    
    
    '''
    nl_int = round(self.size*nl) 
    
    #ORIGINAL ARRAY
    X = self.astype(np.int32)            
    
    
    flip_index = np.random.choice(X.size,nl_int,replace=False)
    print('flip index: ',flip_index)
    X.put(flip_index,flip_bit_array(X))
    
    #COMPLIMENT ARRAY
    Y = self.astype(np.int32)
    Y.put(range(0,Y.size),flip_bit_array(Y)) #NOW ITS A COMPLIMENT
    
    flip_index = np.random.choice(Y.size,nl_int,replace=False)
    print('flip index: ',flip_index)
    Y.put(flip_index,flip_bit_array(Y))
    
    X_Y_x = np.vstack((X,Y))
    #X_Y_x = np.where(X_Y < 0, 0, 1)
    
    np.savetxt(f'Graphs/X_posneg1_nl{nl}.csv',X_Y_x,delimiter=',')
    return None


#EXPAND U WHEN U IS RANDOM PATTERNS
def compliment_rate(m):
    '''
    m is the input matrix of patterns
    '''
    #COMPLIMENT FULL MATRIX
    X = m.astype(np.int32).copy()
    X.put(range(0,X.size),flip_bit_array_rate(X)) #NOW ITS A COMPLIMENT
 
    return X

def replicate_combine_rate(m1,m2,n_r):
    '''
    
    '''
    X = m1.copy()
    Y = m2.copy()

    X_r = np.vstack([X]*n_r)
    Y_r = np.vstack([Y]*n_r)
    X_Y_r = np.vstack((X_r,Y_r))

    return X_Y_r

    
#DEV
#U = hd.make_U(100, 8)
#X = np.vstack([U[0]]*100)
#rite_for_R(X)
