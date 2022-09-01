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

#WEIGHT VALUES MUST BE TAKEN INTO ACCOUNT
for i in range(0,len(W_if)):
    print('node ',i,' sum wts: ',W_if[i].sum())
'''WEIGHT VALUES AS SUMS PER NODE TELL PARTIAL STORY'''


    
'''
TURN ON EACH OF THE UNITS INDEPENDENTLY + CLAMPING
'''
a = np.zeros((22,22))
np.fill_diagonal(a,1)
W = W_if.copy()
T = T_if.copy()
N = 22
temp = 0.1
replicates = 1000 
catch_endstate_clamp_sum = np.zeros((22,22))
catch_catch_endstate_clamp = []
catch_energy_l = []
catch_s_hist_l = []

for j in range(0,replicates):  #TO GENERATE SAMPLES OF THE PROCESS
    print('j',j)
    catch_overlap = np.array([]) 
    catch_ham = np.array([])
    catch_endstate_clamp = np.full(22,99)

    U_a = a
    for i in range(0,len(U_a)):
        S = U_a[i].copy()
        T_clamp = T.copy() #FIX THRESHOLD REALL HIGH FOR NODE J
        T_clamp[i] = 1000
        S[i] = 1 #FIX S FOR Node J
        S_hist = rate.sim_patt_U_prob_rate_if(S,W,T_clamp,2000,temp)
        catch_s_hist_l.append(S_hist)
        #print('ran catch_s_hist_l')
        catch_overlap = np.append(catch_overlap, hd.compute_M(S,U_a[i],N))
        catch_ham = np.append(catch_ham,np.sum(np.bitwise_xor(S.astype(int),U_a[i].astype(int))))
        catch_endstate_clamp = np.vstack((catch_endstate_clamp,S))
        catch_energy = np.array([])
        for j in range(0,len(S_hist)):
            catch_energy = np.append(catch_energy,rate.energy(S_hist[j],W,T))
        catch_energy_l.append(catch_energy)
    
        #print(' i:',i)
        
    catch_endstate_clamp_sum = catch_endstate_clamp_sum + catch_endstate_clamp[1:]
    catch_catch_endstate_clamp.append(catch_endstate_clamp)
    
print("DONE")

'''SAVE RESULTS FOR LATER
...LATEST SAVE was March 12
'''
with open(f'{save_get_dir}catch_catch_endstate_clamp_temp{temp}_replicates{replicates}_list','wb') as filehandle:                            
    pickle.dump(catch_catch_endstate_clamp, filehandle)  

with open(f'{save_get_dir}catch_energy_l_temp{temp}_replicates{replicates}_list','wb') as filehandle:                            
    pickle.dump(catch_energy_l, filehandle) 
    
with open(f'{save_get_dir}catch_s_hist_l_temp{temp}_replicates{replicates}_list','wb') as filehandle:                            
    pickle.dump(catch_s_hist_l, filehandle) 

np.save(f'{save_get_dir}catch_endstate_clamp_sum_temp{temp}_replicates{replicates}',catch_endstate_clamp_sum)

'''....................................................................................................
FROM HERE BELOW HAS BEEN CLEANED UP AND PUT IN <RUN_1B_ANALYSIS.PY
WILL KEEP BELOW FOR REFERENCE IN CASE THE ANALYSIS FILE FAILS 
....................................................................................................'''
'''SUMMARY PLOTS'''

'''USED FOR FIRST SUMBISSION'''
plt.style.use('fivethirtyeight')
imgplot = plt.imshow(catch_endstate_clamp_ave,cmap="binary",aspect='auto',interpolation='none')
plt.xlabel('State Vector Index')
plt.ylabel('Fixed Node Index')
plt.colorbar()
plt.savefig('Fig_Endstates_NodePerturbation.png',dpi=400,bbox_inches='tight')


#hd.plot_catch(catch_overlap,70)
#hd.plot_catch(catch_ham,70)
catch_endstate_clamp_ave = catch_endstate_clamp_sum/replicates
hd.plot_hist(catch_endstate_clamp_ave)
hd.plot_hist(catch_endstate_clamp[1:]) #LAST J FROM ABOVE
#hd.plot_hist(catch_endstate[1:]) #THIS ONE IS FROM SIMPLE NONCLAMP ABOVE
#for i in range(0,len(catch_energy_l)): plt.plot((-1*catch_energy_l[i])); plt.savefig('test_fig.png')
for i in range(0,len(catch_energy_l)): plt.plot((-1*catch_energy_l[i]))    

#CAHTCH ENERGY FOR A SPECIFIC NODE:
e_node = 9
for i in range(e_node,replicates*len(a),22): plt.plot((-1*catch_energy_l[i])) 
    

'''NODE SPECIFIC END STATE ANALYSIS'''
#FOR J OF FIRST INDICE [J][] FROM RUN ABOVE
catch_catch_endstate_clamp_arr = np.array(catch_catch_endstate_clamp)
catch_catch_endstate_clamp_arr = catch_catch_endstate_clamp_arr[:,1:,:] #SLICE OFF THE 99s
hd.plot_hist(catch_catch_endstate_clamp_arr[:,9,:]) #SECOND INDEX GIVES ENDSTATES ACROSS REPLICATES FOR NODE i

#NUMBER OF END STATES
for k in range(0,N):
    print('NODE: ',k)
    target_node_no = k
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
    #print('Len(Endstate): ', len(uniq_endstates_decimal))
    
    #hd.plot_hist(target_node_endstates)
    '''Target nodes endstates for vis'''
    for i in range(0,len(uniq_endstates_decimal)):
        x = uniq_endstates_decimal[i].astype(int)
        #print('vector as decimal:')
        #print(x)
        #print('vector as vector:')
        #print(convert_dec_to_array(x))
        #print('Energy', rate.energy(convert_dec_to_array(x),W,T))

        
        
'''COMPUTE WEIGHTED HAMMING DISTANCE FROM ALL BASIS ATTRACTORS'''
#HAMMING PREP
#k_states =      np.array([85,139349,4191598,4194218,1146880,1548288,3613781,3650629,4191300,69,65605,1114112,1114116,1114197,1674181])
#k_states_freq = np.array([801512,464075,446373,2148642,76462,38726,85472,68630,46512,3,2,12,2,1,17879])
k_states = np.array([85,139349,4191598,4194218])
k_states_freq = np.array([801512,464075,446373,2148642])
k_states_freq_norm = k_states_freq / k_states_freq.sum()
#SEE THE k_STATES
for i in k_states:
    print(i)
    print(type(i))
    print(convert_dec_to_array(i))

#NUMBER OF END STATES WITH WEIGHTED HAMMING 
catch_weighted_ham = []
for j in range(0,N):
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
    catch_weighted_ham_node = np.array([])
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
        
        sum_wgt_ham = np.array([(h_vect * f_vect)*(1/1000)]).sum()
        catch_weighted_ham_node = np.append(catch_weighted_ham_node,sum_wgt_ham)
        #print('k state: ', i)
        #print('HAM X FREQ: ', h_vect * f_vect)
        #print('HAM X FREQ/N: ', (h_vect * f_vect)*(1/1000))
        print('SUM WEIG HAM SCORE: ', np.array([(h_vect * f_vect)*(1/1000)]).sum())
    
    catch_weighted_ham.append(catch_weighted_ham_node)

'''END LOOP'''

#PLOT WEIGHTED HAM AGAINST NETWORK MEASURES
catch_weighted_ham_arr = np.array(catch_weighted_ham)
catch_weighted_ham_arr[:,0]

'''NEED BARPLOT FOR EACH NODE'''
df = pd.DataFrame(catch_weighted_ham_arr,columns=k_states)
#MAKE WEIGHTED HAMMINGS
df['wt_ham'] = df[k_states[0]]*k_states_freq_norm[0] + \
df[k_states[1]]*k_states_freq_norm[1] + df[k_states[2]]*k_states_freq_norm[2] + \
df[k_states[3]]*k_states_freq_norm[3] + df[k_states[4]]*k_states_freq_norm[4] + \
df[k_states[5]]*k_states_freq_norm[5] + df[k_states[6]]*k_states_freq_norm[6] + \
df[k_states[7]]*k_states_freq_norm[7] + df[k_states[8]]*k_states_freq_norm[8] + \
df[k_states[9]]*k_states_freq_norm[9] + df[k_states[10]]*k_states_freq_norm[10] + \
df[k_states[11]]*k_states_freq_norm[11] + df[k_states[12]]*k_states_freq_norm[12] + \
df[k_states[13]]*k_states_freq_norm[13] + df[k_states[14]]*k_states_freq_norm[14]

#ADD GRAPH INFORMATION
series_g_deg = pd.Series(g_deg)
series_g_close = pd.Series(g_close)
series_g_between = pd.Series(g_between)

df['g_deg'] = series_g_deg
df['g_close'] = series_g_close
df['g_between'] = series_g_between
df['mean_centrality'] = (df.g_deg + df.g_close + df.g_between) / 3

df_sorted = df.sort_values(by='mean_centrality')


#PLOTS
#df_sorted_bar_cols = df_sorted[df_sorted.columns[0:15]]
df_sorted_bar_cols = df_sorted[df_sorted.columns[0:4]]
df_sorted_bar_cols.T.plot.bar(stacked=False,legend=True,ylim=(0,24),width=0.75)
plt.xlabel('Basis Point-Attractor')
plt.ylabel('Weighted Hamming Distance')
plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=8,fontsize=10)
plt.savefig('WeightHam_4_Major_Basis_Attractors_Bar.png',dpi=400,bbox_inches='tight')


'''USE FOR INITIAL SUBMISSION'''
plt.style.use('fivethirtyeight')
plt.xlabel('Centrality')
plt.ylabel('Weighted Hamming')
plt.scatter(df.wt_ham,df.mean_centrality,color = 'black')
plt.savefig('WeightedHamByCentrality.png',dpi=400,bbox_inches='tight')





'''
WEIGHTED HAM OVER K for each I, call it weighted ham 2.
'''
#NUMBER OF END STATES WITH WEIGHTED HAMMING 
#NUMBER OF END STATES WITH WEIGHTED HAMMING 
catch_weighted_ham = []
catch_weighted_ham_sum = []
catch_weighted_ham_max = []
for j in range(0,N):
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
    catch_weighted_ham_node = np.array([])
    for k in range(0,len(uniq_endstates_decimal)):
        e_state = convert_dec_to_array(uniq_endstates_decimal[k].astype(int))
        h_vect = np.array([])
        f_vect = np.array([])
        i_dex = 0
        for i in k_states:
            print('k state: ', i)
            k_state = convert_dec_to_array(i)
            print('HAMMING: ', 1+np.sum(np.bitwise_xor(k_state.astype(int),e_state.astype(int))))
            print('FREQ: ', uniq_endstates_decimal_freq[k])
            h_vect = np.append(h_vect,np.sum(np.bitwise_xor(k_state.astype(int),e_state.astype(int))))
            f_vect = np.append(f_vect,uniq_endstates_decimal_freq[k])
            i_dex =+ 1
            
        h_vect_min_index = np.where(h_vect == h_vect.min())
        h_vect_min = h_vect[h_vect_min_index]+1
        f_vect_min = f_vect[h_vect_min_index]
            
        f_denom = uniq_endstates_decimal_freq.sum()
        #sum_wgt_ham = np.array([(h_vect * f_vect)*(1/f_denom)]).sum()
        sum_wgt_ham = np.array([(h_vect_min * f_vect_min)*(1/f_denom)]).sum()
        catch_weighted_ham_node = np.append(catch_weighted_ham_node,sum_wgt_ham)
        #print('k state: ', i)
        #print('HAM X FREQ: ', h_vect * f_vect)
        #print('HAM X FREQ/N: ', (h_vect * f_vect)*(1/1000))
        '''NEED NEW N_k'''
        print('SUM WEIG HAM SCORE: ', np.array([(h_vect_min * f_vect_min)*(1/f_denom)]).sum())
    
    catch_weighted_ham.append(catch_weighted_ham_node)
    catch_weighted_ham_sum.append(catch_weighted_ham_node.sum())
    catch_weighted_ham_max.append(catch_weighted_ham_node.max())
    
tmp = np.where(h_vect == h_vect.min())
'''END LOOP'''
#PLOT WEIGHTED HAM AGAINST NETWORK MEASURES
catch_weighted_ham
catch_weighted_ham_arr[:,0]

#HAND TESTING
ks = convert_dec_to_array(4194218)
es = convert_dec_to_array(85)
np.sum(np.bitwise_xor(ks,es))
freq = 994/999
k_states_freq_norm
h = np.array([0,2,17,22])
h_by_k = h*k_states_freq_norm
tmp = h_by_k*freq
tmp.sum()
h_by_k

'''
NOTES:
--Temperature has an effect on endstates per each clamped node
----Fixing temp to 0.001, only three nodes {0,1 and 5} have any variability (with 1K replicates)
------Node 0 has two endstates, Node 1 has 2 (similar in kind to Node 0s), Node 5 has two, similar to Node 2.
------The energy profile is fucked up; never dips below -8...; still get distinct bands:  But the clamped node is fucking things up
--------for the natural attractor states; how formalize this?
----Fixing temp to 0.1, with 1K replicates
------Compared to 0.001 temp, we now get Node 9 having variablity
------Some nodes now have a set of nodes (e.g., nodes 0 and 9) have about 15 endstates; some of the endstates which correspond well 
--------with the final endstates in the space of 2^22 simple inputs (no clamping)
------Energy distribution corresponds to the space of 2^22 simple inputs (no clapming)
------Its like only 3 or so nodes, when clamped, can push the system to its natural attractors; is this what is meant by Dalege
--------influential nodes?
------Would like to know the frequencies of the endstates: Done, and can see the most freq.
------SEEMS LIKE NEED TO DOUBLE THE SIM TIME t0 2000 WHEN LOOK AT ENERGY TRAJECTORIES FOR EACH RUN.
----Would like to know the ranking of the graph properties mentioned in Dalege:
----
'''


            
'''
TESTING AND DEV CODE
'''
#PRINTING INDEX FOR 9th UNIT IN NETWORK FOR EACH REPLICATE
#TO SEE BEHAVIOR OF THIS SUPPOSEDLY INFLUENTIAL UNIT FOR ONE RUN
for i in range(0,99):
    j = (i*22)+9
    print(j)
hd.plot_hist(catch_s_hist_l[1065][:])
        
#TEST ON CLAMPING
catch_hist_arr = np.array(catch_s_hist_l)
catch_hist_arr[:,:,1].min() #FIX THIRD ELEMENT ON j AS DEFINED ABOVE
catch_hist_arr[:,:,1].max() # "" 

#TEST ON ENDSTATES
len(catch_unique_endstate_l)
set(catch_unique_endstate_l)
catch_unique_endstate_l


'''EXTRA DEPRECIATED ANALYSIS'''
'''
TURN ON EACH OF THE UNITS ON INDEPENDENTLY
'''
catch_overlap = np.array([]) 
catch_ham = np.array([])
catch_endstate = np.full(22,99)
catch_energy_l = []
catch_s_hist_l = []
a = np.zeros((22,22))
np.fill_diagonal(a,1)
W = W_if.copy()
T = T_if.copy()
N = 22
U_a = a
for i in range(0,len(U_a)):
    S = U_a[i].copy()
    S_hist = rate.sim_patt_U_prob_rate_if(S,W,T,1000,.1)
    catch_s_hist_l.append(S_hist)
    catch_overlap = np.append(catch_overlap, hd.compute_M(S,U_a[i],N))
    catch_ham = np.append(catch_ham,np.sum(np.bitwise_xor(S.astype(int),U_a[i].astype(int))))
    catch_endstate = np.vstack((catch_endstate,S))
    catch_energy = np.array([])
    for j in range(0,len(S_hist)):
        catch_energy = np.append(catch_energy,rate.energy(S_hist[j],W,T))
    catch_energy_l.append(catch_energy)
    
    print(' i:',i)
print("DONE")

'''SUMMARY PLOTS'''

#hd.plot_catch(catch_overlap,70)
#hd.plot_catch(catch_ham,70)
hd.plot_hist(catch_endstate[1:])

for i in range(0,len(catch_energy_l)): plt.plot((-1*catch_energy_l[i]))
    
hd.plot_hist(catch_s_hist_l[0][:300])

'''END STATE ANALYSIS'''
#NUMBER OF END STATES
catch_each_decimal = np.array([])
for i in range(1,len(catch_endstate)):
    tmp0 = catch_endstate[i].astype(int)
    tmp1 = list(tmp0.astype(str))
    tmp2 = int(''.join(tmp1),2)
    catch_each_decimal = np.append(catch_each_decimal,tmp2)

uniq_endstates_decimal = np.unique(catch_each_decimal)
print('Endstates: ', uniq_endstates_decimal)
print('Len(Endstate): ', len(uniq_endstates_decimal))
'''WHICH ONES WERE LOW ENERGY??????'''

#ENDSTATS IN BINARY
for i in range(0,len(uniq_endstates_decimal)):
    x = uniq_endstates_decimal[i].astype(int)
    print(x)
    print(convert_dec_to_array(x))



#EOF