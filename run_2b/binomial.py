import math as math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from math import factorial as fact
def binomial(n,k):
    return fact(n) // fact(k) // fact(n-k)

print(binomial(20,10))

'''NOT WORK IN PYTHON 2.7'''
def binom(n,k):
    return math.comb(n,k)
print(binom(20,10))

def prob_of(n,k,p):
    n_choose_k = binomial(n,k)
    return n_choose_k * p**k * (1-p)**(n-k)

n = 10

#k is i
catch_p01 = np.array([])
for i in range(0,n+1):
    catch_p01 = np.append(catch_p01,prob_of(n,i,0.01))

catch_p01_df = pd.DataFrame(catch_p01)
catch_p01_df.to_csv('catch_p01.csv',header=False)

catch_p01_ge_k = np.array([])
for i in range(0,n+1):
    catch_p01_ge_k = np.append(catch_p01_ge_k,catch_p01[i:].sum())

catch_p01_ge_k_df = pd.DataFrame(catch_p01_ge_k)
catch_p01_ge_k_df.to_csv('catch_p01_ge_k.csv',header=False)

print('Catch_p01, Binomial Sum: ',catch_p01.sum())
print('Catch_p01, >5: ', catch_p01[6:].sum())



catch_p10 = np.array([])
for i in range(0,n+1):
    catch_p10 = np.append(catch_p10,prob_of(n,i,0.10))

catch_p10_df = pd.DataFrame(catch_p10)
catch_p10_df.to_csv('catch_p10.csv',header=False)

catch_p10_ge_k = np.array([])
for i in range(0,n+1):
    catch_p10_ge_k = np.append(catch_p10_ge_k,catch_p10[i:].sum())

catch_p10_ge_k_df = pd.DataFrame(catch_p10_ge_k)
catch_p10_ge_k_df.to_csv('catch_p10_ge_k.csv',header=False)

print('Catch_p10, Binomial Sum: ',catch_p10.sum())
print('Catch_p10, >5: ', catch_p10[6:].sum())



catch_p20 = np.array([])
for i in range(0,n+1):
    catch_p20 = np.append(catch_p20,prob_of(n,i,0.20))

catch_p20_df = pd.DataFrame(catch_p20)
catch_p20_df.to_csv('catch_p20.csv',header=False)

catch_p20_ge_k = np.array([])
for i in range(0,n+1):
    catch_p20_ge_k = np.append(catch_p20_ge_k,catch_p20[i:].sum())

catch_p20_ge_k_df = pd.DataFrame(catch_p20_ge_k)
catch_p20_ge_k_df.to_csv('catch_p20_ge_k.csv',header=False)

print('Catch_p20, Binomial Sum: ',catch_p20.sum())
print('Catch_p20, >5: ', catch_p20[6:].sum())


catch_p40 = np.array([])
for i in range(0,n+1):
    catch_p40 = np.append(catch_p40,prob_of(n,i,0.40))

catch_p40_df = pd.DataFrame(catch_p40)
catch_p40_df.to_csv('catch_p40.csv',header=False)

catch_p40_ge_k = np.array([])
for i in range(0,n+1):
    catch_p40_ge_k = np.append(catch_p40_ge_k,catch_p40[i:].sum())

catch_p40_ge_k_df = pd.DataFrame(catch_p40_ge_k)
catch_p40_ge_k_df.to_csv('catch_p40_ge_k.csv',header=False)

print('Catch_p40, Binomial Sum: ',catch_p40.sum())
print('Catch_p40, >5: ', catch_p40[6:].sum())



catch_p60 = np.array([])
for i in range(0,n+1):
    catch_p60 = np.append(catch_p60,prob_of(n,i,0.60))

catch_p60_df = pd.DataFrame(catch_p60)
catch_p60_df.to_csv('catch_p60.csv',header=False)

catch_p60_ge_k = np.array([])
for i in range(0,n+1):
    catch_p60_ge_k = np.append(catch_p60_ge_k,catch_p60[i:].sum())

catch_p60_ge_k_df = pd.DataFrame(catch_p60_ge_k)
catch_p60_ge_k_df.to_csv('catch_p60_ge_k.csv',header=False)

print('Catch_p60, Binomial Sum: ',catch_p60.sum())
print('Catch_p60, >5: ', catch_p60[6:].sum())


    
catch_p1 = np.array([])
for i in range(0,n+1):
    catch_p1 = np.append(catch_p1,prob_of(n,i,1.0))

catch_p1_df = pd.DataFrame(catch_p1)
catch_p1_df.to_csv('catch_p1.csv',header=False)

catch_p1_ge_k = np.array([])
for i in range(0,n+1):
    catch_p1_ge_k = np.append(catch_p1_ge_k,catch_p1[i:].sum())

catch_p1_ge_k_df = pd.DataFrame(catch_p1_ge_k)
catch_p1_ge_k_df.to_csv('catch_p1_ge_k.csv',header=False)

print('Catch_p1, Binomial Sum: ',catch_p1.sum())
print('Catch_p1, >5: ', catch_p1[6:].sum())


'''LOOKING AT THE DATA'''
p01 = pd.read_csv('catch_p01_ge_k.csv',header=None)
p

p10 = pd.read_csv('catch_p10_ge_k.csv',header=None)

p20 = pd.read_csv('catch_p20_ge_k.csv',header=None)

p40 = pd.read_csv('catch_p40_ge_k.csv',header=None)

p60 = pd.read_csv('catch_p60_ge_k.csv',header=None)

p1 = pd.read_csv('catch_p1_ge_k.csv',header=None)

plt.plot(p01[1])
plt.plot(p10[1])
plt.plot(p20[1])
plt.plot(p40[1])
plt.plot(p60[1])
plt.plot(p1[1])

#GT 5
gt_5 = np.array([p01.iloc[6][1],
                 p10.iloc[6][1],
                p20.iloc[6][1],
                p40.iloc[6][1],
                p60.iloc[6][1],
                p1.iloc[6][1]])
