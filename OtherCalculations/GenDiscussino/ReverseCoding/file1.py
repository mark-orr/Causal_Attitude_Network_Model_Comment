import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def energy(x_1,x_2,tau_1,tau_2,w):
    
    print('Test is added Xs: ',x_1+x_2)

    a = x_1*tau_1; print('a ',a)
    b = x_2*tau_2; print('b ',b)
    c = x_1*x_2*w; print('c ',c)
    return -(a+b+c)

input_list = [[1,1],[1,-1],[-1,1],[-1,-1]]

#DIFF TAU, NEG WT THEN FLIP WT
'''ORIGINAL DATA'''
w = -1; tau_1 = -1; tau_2 = 1
for i in input_list:
    print(i)
    #print('x_1', i[0]); print('x_2', i[1])
    print(energy(i[0],i[1],tau_1,tau_2,w))
'''RECODE DATA'''
w = 1; tau_1 = -1; tau_2 = 1
for i in input_list:
    print(i)
    #print('x_1', i[0]); print('x_2', i[1])
    print(energy(i[0],i[1],tau_1,tau_2,w))

#SAME POS TAU, NEG WT THEN FLIP WT
'''ORIGINAL DATA'''
w = -1; tau_1 = 1; tau_2 = 1
for i in input_list:
    print(i)
    #print('x_1', i[0]); print('x_2', i[1])
    print(energy(i[0],i[1],tau_1,tau_2,w))
'''RECODE DATA'''
w = 1; tau_1 = 1; tau_2 = 1
for i in input_list:
    print(i)
    #print('x_1', i[0]); print('x_2', i[1])
    print(energy(i[0],i[1],tau_1,tau_2,w))

#SAME NEG TAU, NEG WT THEN FLIP WT
'''ORIGINAL DATA'''
w = -1; tau_1 = -1; tau_2 = -1
for i in input_list:
    print(i)
    #print('x_1', i[0]); print('x_2', i[1])
    print(energy(i[0],i[1],tau_1,tau_2,w))
'''RECODE DATA'''
w = 1; tau_1 = -1; tau_2 = -1
for i in input_list:
    print(i)
    #print('x_1', i[0]); print('x_2', i[1])
    print(energy(i[0],i[1],tau_1,tau_2,w))


#EOF 