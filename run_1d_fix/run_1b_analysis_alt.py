import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import subprocess
from numpy import genfromtxt
from imp import reload
import networkx as nx
import pandas as pd
''' CUSTOM PACKAGES'''
import sys
sys.path.insert(1,'/Users/biocomplexity/Projects/SocioCognitiveModeling/NSF-NetCogSys/Products/CAN_Model/CAN_GitRepo/IsingFitStudy/Hopfield/hop_pack')
import head as hd
import rate
import stat
reload(stat) #FOR SOME REASON NEED THIS
import pickle

def convert_dec_to_array(x):
    mystr = format(x, '022b') #WITH PADDING
    return np.fromiter(mystr, dtype=int)

'''ANALYSIS FOR THE ARTICLE SUBMISSION IS ANNOTATED BELOW'''

save_get_dir = '/Users/biocomplexity/Projects/SocioCognitiveModeling/NSF-NetCogSys/Products/CAN_Model/CAN_GitData/Hopfield/run_1b_sim_out/'


#GEN DATA
#IMPOR AND PROCESS WITH R
print('START R PROCESS')
subprocess.check_call(['Rscript', 'Graphs/IsingFit.r'], shell=False)
print('END R PROCESS')
#LOAD CSV TO PYTHON
#CONVERT TO W_if
from numpy import genfromtxt
W_if = genfromtxt('Graphs/graph_from_R.csv', delimiter=',')
T_if = genfromtxt('Graphs/thresholds_from_R.csv', delimiter=',')
U_if = genfromtxt('Graphs/data_from_R.csv',delimiter=',')


'''NETWORK ANALYSIS DEV'''
G = nx.from_numpy_matrix(W_if, create_using=nx.Graph) #SHOULD BE UNDIRECTD

#PLOT IT
nx.draw_networkx(G)                                                        
limits = plt.axis('on')                                                   
plt.show(G)
g_deg = nx.degree_centrality(G)
g_close = nx.closeness_centrality(G)
g_between = nx.betweenness_centrality(G)


'''................
LOAD RESULTS AND PARAMS FROM SIM
....................'''
a = np.zeros((22,22))
np.fill_diagonal(a,1)
W = W_if.copy()
T = T_if.copy()
N = 22
temp = 0.1
replicates = 1000 

with open(f'{save_get_dir}catch_catch_endstate_clamp_temp0.1_replicates1000_list','rb') as filehandle:
    catch_catch_endstate_clamp = pickle.load(filehandle)
    
with open(f'{save_get_dir}catch_energy_l_temp0.1_replicates1000_list','rb') as filehandle:
    catch_energy_l = pickle.load(filehandle)
    
with open(f'{save_get_dir}catch_s_hist_l_temp0.1_replicates1000_list','rb') as filehandle:
    catch_s_hist_l = pickle.load(filehandle)

catch_endstate_clamp_sum = np.load(f'{save_get_dir}catch_endstate_clamp_sum_temp0.1_replicates1000.npy')



'''SUMMARY PLOTS'''
catch_endstate_clamp_ave = catch_endstate_clamp_sum/replicates
hd.plot_hist(catch_endstate_clamp_ave)

'''USED FOR FIRST SUMBISSION'''
plt.style.use('fivethirtyeight')
imgplot = plt.imshow(catch_endstate_clamp_ave,cmap="binary",aspect='auto',interpolation='none')
plt.xlabel('State Vector Index')
plt.ylabel('Fixed Node Index')
plt.colorbar()
plt.savefig('Fig_Endstates_NodePerturbation.png',dpi=400,bbox_inches='tight')



'''NODE SPECIFIC END STATE ANALYSIS'''
#FOR J OF FIRST INDICE [J][] FROM RUN ABOVE
catch_catch_endstate_clamp_arr = np.array(catch_catch_endstate_clamp)
catch_catch_endstate_clamp_arr = catch_catch_endstate_clamp_arr[:,1:,:] #SLICE OFF THE 99s
hd.plot_hist(catch_catch_endstate_clamp_arr[:,9,:]) #SECOND INDEX GIVES ENDSTATES ACROSS REPLICATES FOR NODE i
        
'''............................
COMPUTING THE MEASUER OF EACH NODES EFFECT, 
---THE FREQUENCY WEIGHTED HAMMING DISTANCE TO EACH ATTRACTOR
WHICH IS THEN AVERAGED OVER ALL ATTRACTORS FOR A NODE
'''

'''PREP FOR THE HAMMING DIST MEASURES'''
#HAMMING PREP
'''ENTERNED BY HAND WITHOUT ORDER IN MIND'''
k_states_entered =      np.array([85,139349,4191598,4194218,1146880,1548288,3613781,3650629,4191300,69,65605,1114112,1114116,1114197,1674181])
k_states_freq_entered = np.array([801512,464075,446373,2148642,76462,38726,85472,68630,46512,3,2,12,2,1,17879])
#k_states = np.array([85,139349,4191598,4194218])
#k_states_freq = np.array([801512,464075,446373,2148642])
'''ORDER THEM'''
arr1 = np.vstack((k_states_entered,k_states_freq_entered))
arr2 = arr1.T
arr3 = arr2[arr2[:,1].argsort()]
k_states = arr3[:,0]
k_states_freq = arr3[:,1]
#NORM THE FREQ
k_states_freq_norm = k_states_freq / k_states_freq.sum()

    
'''HAMMING DIST FINAL APPROACH'''
'''COMPUTE WEIGHTED HAMMING DISTANCE FROM ALL FIXED-POINT ATTRACTORS'''
#NUMBER OF END STATES WITH WEIGHTED HAMMING 
catch_weighted_ham = []
catch_dist = []
catch_catch_hammings_arr = []
catch_arr1 = []
catch_arr2 = []
catch_arr3 = []
catch_arr4 = []
catch_arr5 = []
catch_arr6 = []
catch_arr7 = []
catch_arr8 = []
catch_weighted_ham_2 = []
catch_dist_2 = []
#for j in range(0,N):
for j in range(0,1): #FOR TESTING ONLY
    print('NODE: ',j)
    target_node_no = j
    catch_each_decimal = np.array([])
    target_node_endstates = catch_catch_endstate_clamp_arr[:,target_node_no,:]
    for i in range(1,len(target_node_endstates)):
        tmp0 = target_node_endstates[i].astype(int)
        tmp1 = list(tmp0.astype(str))
        tmp2 = int(''.join(tmp1),2)
        catch_each_decimal = np.append(catch_each_decimal,tmp2)
        
    uniq_endstates_decimal = np.unique(catch_each_decimal)
    uniq_endstates_decimal_freq = np.unique(catch_each_decimal,return_counts=True)[1]
    print('Endstates: ', uniq_endstates_decimal)
    print('Freq of Endstates: ', uniq_endstates_decimal_freq)
    print('Len(Endstate): ', len(uniq_endstates_decimal))
    
    #hd.plot_hist(target_node_endstates)
    '''Target nodes endstates for vis'''
    #for i in range(0,len(uniq_endstates_decimal)):
        #x = uniq_endstates_decimal[i].astype(int)
        #print('vector as decimal:')
        #print(x)
        #print('vector as vector:')
        #print(convert_dec_to_array(x))
        #print('Energy', rate.energy(convert_dec_to_array(x),W,T))
    
    '''COMPUTE WEIGHTED HAMMING'''
    #catch_weighted_ham_node = np.array([]) #DONT NEED AS IS A SCALAR IN NEW CODE
    catch_hammings = np.zeros(len(uniq_endstates_decimal))
    catch_freqs = np.zeros(len(uniq_endstates_decimal))
    for i in k_states:
        print('k state: ', i)
        k_state = convert_dec_to_array(i)
        h_vect = np.array([])
        f_vect = np.array([])
        for k in range(0,len(uniq_endstates_decimal)):
            e_state = convert_dec_to_array(uniq_endstates_decimal[k].astype(int))
            #print('HAMMING: ', np.sum(np.bitwise_xor(k_state.astype(int),e_state.astype(int))))
            #print('FREQ: ', uniq_endstates_decimal_freq[k])
            h_vect = np.append(h_vect,np.sum(np.bitwise_xor(k_state.astype(int),e_state.astype(int))))
            f_vect = np.append(f_vect,uniq_endstates_decimal_freq[k])
        
        catch_hammings = np.vstack((catch_hammings,h_vect))
        catch_freqs = np.vstack((catch_freqs,f_vect))
        
    catch_hammings_arr = catch_hammings[1:]
    catch_freqs_arr = catch_freqs[1]
    #FOR TESTING
    catch_catch_hammings_arr.append(catch_hammings_arr)
        
    '''NEW METHOD'''
    #FIRST LOOP TO FREQ DIST BY MIN HAMM PER COL
    arr1 = np.zeros(catch_hammings_arr.shape)
    for i in range(0,arr1.shape[1]): #looping over cols   
        loc_min_of_col = np.where(catch_hammings_arr[:,i]==min(catch_hammings_arr[:,i]))
        list_loc = list(loc_min_of_col[0])
        arr1[list_loc,i] = catch_freqs_arr[i]/len(list_loc)
    catch_arr1.append(arr1)
    #SECOND LOOP TO GET FREQ DIST BY MIN HAMM PER ROW 
    arr2 = np.array([])
    arr3 = np.array([])
    for i in range(0,arr1.shape[0]):
        loc_min_of_row = np.where(catch_hammings_arr[i,:]==min(catch_hammings_arr[i,:]))
        list_loc = list(loc_min_of_row)
        arr2 = np.append(arr2,arr1[i,list_loc].sum())
        arr3 = np.append(arr3,catch_hammings_arr[i,list_loc].mean())
    catch_arr2.append(arr2)
    catch_arr3.append(arr3)
    '''MEAN HAMMING, WEIGHTED AND NORMALIZED BY HIT SUMS NOT NECESSARILY TOTAL SUM'''
    wt_ham = np.dot(arr2,arr3)/arr2.sum()

    'DIST NORMALIZED BY FULL DENOM'
    arr4 = arr2/1000
    catch_arr4.append(arr4)

    '''COMPARE DISTS'''
    pd.DataFrame([arr4,k_states_freq_norm]).round(4).T

    '''NORM BOTH VECTOR DISTRIBUTIONS'''
    arr4_len = np.sqrt(np.dot(arr4,arr4))
    arr4_norm = arr4/arr4_len
    #CHECK
    np.sqrt(np.dot(arr4_norm,arr4_norm))#should equal 1

    k_states_freq_norm_len = np.sqrt(np.dot(k_states_freq_norm,k_states_freq_norm))
    k_states_freq_norm_norm = k_states_freq_norm/k_states_freq_norm_len
    np.sqrt(np.dot(k_states_freq_norm_norm,k_states_freq_norm_norm))

    '''COMPUTE DIST BETWEEN THE NORMED VECTORS'''
    v1_min_v2 = k_states_freq_norm_norm - arr4_norm
    v1_d_v2 = np.sqrt(np.dot(v1_min_v2,v1_min_v2))

    '''SIMPLER METHOD'''
    arr5 = np.zeros(catch_hammings_arr.shape)
    #arr5[arr5==0]=None #FOR TESTING ONLY
    for i in range(0,arr5.shape[1]): #looping over cols   
        loc_min_of_col = np.where(catch_hammings_arr[:,i]==min(catch_hammings_arr[:,i]))
        list_loc = list(loc_min_of_col[0])
        arr5[list_loc,i] = catch_hammings_arr[list_loc,i]
    catch_arr5.append(arr5)
    #APPLY OVER ARR_5 FOR AVE HAMMING 
    arr6 = arr5.mean(axis=1)
    catch_arr6.append(arr6)
    #APPLY OVER ARR_1 FOR SUM FREQ
    arr7 = arr1.sum(axis=1)
    catch_arr7.append(arr7)
    #DIST NORMALIZED BY FULL DENOM
    arr8 = arr7/arr7.sum()
    catch_arr8.append(arr8)
    
    #WEIGHTED HAM
    wt_ham_2 = np.dot(arr6,arr8) #DIFF THAN wt_ham bc did denom when created arr8 from arr7
    
    #NORM ARR 8
    arr8_len = np.sqrt(np.dot(arr8,arr8))
    arr8_norm = arr8/arr8_len
    #CHECK
    np.sqrt(np.dot(arr8_norm,arr8_norm))#should equal 1
    #COMPUTE DIST BETWEEN ARR7 and Kstates
    v3_min_v4 = k_states_freq_norm_norm - arr8_norm
    v3_d_v4 = np.sqrt(np.dot(v3_min_v4,v3_min_v4))
    
    catch_weighted_ham_2.append(wt_ham_2)
    catch_dist_2.append(v3_d_v4)
    '''END SIMPLER METHOD'''
    #PRIOR CODE
    catch_weighted_ham.append(wt_ham)
    catch_dist.append(v1_d_v2)

'''END LOOP'''

'''PLOTS OF HAMMINGS ETC.'''
#MERGE WITH GRAPH INFO IN DATAFRAME
df = pd.DataFrame([catch_weighted_ham_2,catch_dist_2]).T
 
#ADD GRAPH INFORMATION
series_g_deg = pd.Series(g_deg)
series_g_close = pd.Series(g_close)
series_g_between = pd.Series(g_between)

df['g_deg'] = series_g_deg
df['g_close'] = series_g_close
df['g_between'] = series_g_between
df['mean_centrality'] = (df.g_deg + df.g_close + df.g_between) / 3

df_sorted = df.sort_values(by='mean_centrality')

'''USE THIS FOR INITIAL SUBMISSION'''
plt.style.use('fivethirtyeight')
plt.xlabel('Centrality')
plt.ylabel('Weighted Hamming')
plt.scatter(df_sorted.mean_centrality,df_sorted[0],color = 'black')
plt.savefig('Fig_WeightHam.png',dpi=400,bbox_inches='tight')


plt.style.use('fivethirtyeight')
plt.xlabel('Centrality')
plt.ylabel('Euclidian Distance')
plt.scatter(df_sorted.mean_centrality,df_sorted[1],color = 'black')
plt.savefig('Fig_EuclidianDist.png',dpi=400,bbox_inches='tight')

'''
ADDING THE DISTRIBUTION GRAPH
ORIGNIATED FROM ONE OF THE SANITY CHECKS BELOW
MAY CONSIDER THIS FOR FINAL SUBMISSION
'''
#PLOT WEIGHTED HAM AGAINST NETWORK MEASURES
new_arr = np.array(catch_arr8)
df_new_arr = pd.DataFrame(new_arr,columns=k_states)
df_new_arr.T.plot.bar(stacked=False,legend=False,ylim=(0,1),width=1.1,colormap='gray')
plt.plot(k_states_freq_norm,linewidth=0.8,color='red')

from matplotlib.lines import Line2D

for i in range(0,len(catch_arr8)):
    if i==9: 
        plt.plot(catch_arr8[i],'k--',linewidth=0.9,color='green')
    else: 
        plt.plot(catch_arr8[i],'k-.',linewidth=0.5,color='black')
    
    plt.plot(k_states_freq_norm,linewidth=1.1,color='red')
    plt.xlabel('Referent Fixed-Points (Decimal Value)')
    plt.ylabel('Proportion')
    plt.xticks(ticks=range(0,len(df_new_arr.columns)),labels=df_new_arr.columns,rotation=80)
    
    custom_lines = [Line2D([0], [0], color='red', lw=1),
                Line2D([0], [0], linestyle='--',color='green', lw=1),
                Line2D([0], [0], linestyle='-.',color='black', lw=1)]
    plt.legend(custom_lines, ['Referent', 'Node 9', 'Other Nodes'])
plt.savefig('Fig_DistributionByNode.png',dpi=400,bbox_inches='tight')

'''SANITY CHECKS'''
'''SEE THE _dev.py version of this file'''

#EOF