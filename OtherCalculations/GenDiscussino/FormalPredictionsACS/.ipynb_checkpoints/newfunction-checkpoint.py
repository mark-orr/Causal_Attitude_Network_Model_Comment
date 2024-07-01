import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.arange(0,1,0.01)

def cool_curve3(x):
    f_x = np.array([])
    
    for i in x:
        print(x)
        f_x = np.append(f_x,((i)**2))
        
    return f_x

y = cool_curve3(x)

plt.style.use('fivethirtyeight')
plt.plot(x,y,color='black',linestyle='dashed',linewidth=0.89)
plt.xlabel('Persuasive Influence (Delta)')
plt.ylabel('Automatic Associative Bias (Gamma)')
plt.fill_between(
        x= x, 
        y1= y, 
        where= (0 < x)&(x < 1),
        color= "black",
        alpha= 0.15)

plt.savefig('FormalPredictionCAN_NonAnnotated.png',dpi=300,bbox_inches='tight')



'''SCRATCH BELOW'''
def cool_curve2(x):
    f_x = np.array([])
    
    for i in x:
        print(x)
        f_x = np.append(f_x,-((i)**2))
        
    return f_x

y = cool_curve2(x)

plt.style.use('fivethirtyeight')
plt.plot(x,y,color='black',linestyle='dashed',linewidth=0.89)
plt.xlabel('Persuasive Influence (Delta)')
plt.ylabel('Automatic Associative Bias (Gamma)')
plt.fill_between(
        x= x, 
        y1= -y, 
        where= (0 < x)&(x < 1),
        color= "black",
        alpha= 0.15)

#EOF

