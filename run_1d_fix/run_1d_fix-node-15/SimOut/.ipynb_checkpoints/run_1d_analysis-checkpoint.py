import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import subprocess
import multiprocessing
import logging
import time
import pickle
from imp import reload
''' CUSTOM PACKAGES'''
import sys
sys.path.insert(1,'/Users/biocomplexity/Projects/SocioCognitiveModeling/NSF-NetCogSys/Products/CAN_Model/CAN_GitRepo/IsingFitStudy/Hopfield/hop_pack')
import head as hd
import rate


#TESTING OUTPUT
with open('catch_energy_l_out3000019','rb') as filehandle:
    catch_energy_l = pickle.load(filehandle)
    
for i in range(0,len(catch_energy_l)): plt.plot((-1*catch_energy_l[i]))    


catch_endstate = np.load('catch_endstate_out_3000019.npy')
hd.plot_hist(catch_endstate[1:])

