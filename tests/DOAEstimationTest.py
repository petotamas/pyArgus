# -*- coding: utf-8 -*-
"""
********************************************************
*****          Demonstration and test              *****
********************************************************

PyARGUS 

Demonstration functions written to test the proper operation of the 
direction of arrival estimation functions.

TODO: Write demonstration functions for the spatial smoothing technique
TODO: Write demonstration functions circular antenna arrays

Tamás Pető
21, february 2018

"""
import numpy as np
from pyargus.directionEstimation import *
import matplotlib.pyplot as plt

def demo_single_signal(M=4, d=0.5):
    """
     Description:
     ------------
         
    
        Parameters:
        -----------          
            M     : (int) Number of antenna elements in the antenna array. Default value: 4
            d     : (float) Distance between antenna elements. [lambda] Default value: 0.5
    """
    
    N = 2**12  # sample size          
    theta = 90 # Incident angle of the test signal
    
    # Array response vectors of the test signal
    a = np.exp(np.arange(0,M,1)*1j*2*np.pi*d*np.cos(np.deg2rad(theta)))
       
    # Generate multichannel test signal 
    soi = np.random.normal(0,1,N)  # Signal of Interest
    soi_matrix  = np.outer( soi, a).T 
    
    # Generate multichannel uncorrelated noise
    noise = np.random.normal(0,np.sqrt(10**-10),(M,N))
    
    # Create received signal
    rec_signal = soi_matrix + noise
    
    ## R matrix calculation
    R = corr_matrix_estimate(rec_signal.T, imp="mem_eff")
    
    # Generate scanning vectors
    array_alignment = np.arange(0, M, 1) * d
    incident_angles= np.arange(0,181,1)
    ula_scanning_vectors = gen_ula_scanning_vectors(array_alignment, incident_angles)
      
    # DOA estimation           
    Bartlett= DOA_Bartlett(R, ula_scanning_vectors)    
    Capon = DOA_Capon(R, ula_scanning_vectors)
    MEM = DOA_MEM(R, ula_scanning_vectors)
    LPM = DOA_LPM(R, ula_scanning_vectors, element_select = 0)
    MUSIC = DOA_MUSIC(R, ula_scanning_vectors, signal_dimension = 1)
    
    # Plot results
    axes = plt.axes()
    DOA_plot(Bartlett, incident_angles, log_scale_min = -50, axes=axes)
    DOA_plot(Capon, incident_angles, log_scale_min = -50, axes=axes)
    DOA_plot(MEM, incident_angles, log_scale_min = -50,axes=axes)
    DOA_plot(LPM, incident_angles, log_scale_min = -50, axes=axes)
    DOA_plot(MUSIC, incident_angles, log_scale_min = -50, axes=axes)
    axes.legend(("Bartlett","Capon","MEM","LPM","MUSIC"))
    
   
def demo_coherent_signals(M = 4, d=0.5):
    """
     Description:
     ------------
         Basic demonstration for the forward-backward averaging.         
    
        Parameters:
        -----------          
            M     : (int) Number of antenna elements in the antenna array. Default value: 4
            d     : (float) Distance between antenna elements. [lambda] Default value: 0.5
    """
    
    N = 2**10  # sample size          
    theta_list=[50, 80]   # Incident angles of test signal   
       
    # Generate multichannel test signal 
    soi = np.random.normal(0,1,N)   # Signal of Interest        
    
    soi_matrix  = np.zeros((M,N), dtype=complex)    

    for p in range(len(theta_list)):        
        a = np.exp(np.arange(0,M,1)*1j*2*np.pi*d*np.cos(np.deg2rad(theta_list[p])))    
        soi_matrix  += (np.outer( soi, a)).T 
    
    # Generate multichannel uncorrelated noise
    noise = np.random.normal(0,np.sqrt(10**-3),(M,N))
    
    # Create received signal
    rec_signal = soi_matrix + noise
    
    ## R matrix calculation
    R = corr_matrix_estimate(rec_signal.T, imp="mem_eff")
    
    R = forward_backward_avg(R)
    
    # Generate scanning vectors        
    array_alignment = np.arange(0, M, 1) * d
    incident_angles= np.arange(0,181,1)
    ula_scanning_vectors = gen_ula_scanning_vectors(array_alignment, incident_angles)
    
    # DOA estimation
    Bartlett = DOA_Bartlett(R, ula_scanning_vectors)    
    Capon = DOA_Capon(R, ula_scanning_vectors)
    MEM = DOA_MEM(R, ula_scanning_vectors,  column_select = 0)
    LPM = DOA_LPM(R, ula_scanning_vectors, element_select = 1)
    MUSIC = DOA_MUSIC(R, ula_scanning_vectors, signal_dimension = 3)
    
    # Plot results
    axes = plt.axes()
    DOA_plot(Bartlett, incident_angles, log_scale_min = -50, axes=axes)
    DOA_plot(Capon, incident_angles, log_scale_min = -50, axes=axes)
    DOA_plot(MEM, incident_angles, log_scale_min = -50,axes=axes)
    DOA_plot(LPM, incident_angles, log_scale_min = -50, axes=axes)
    DOA_plot(MUSIC, incident_angles, log_scale_min = -50, axes=axes)
    axes.legend(("Bartlett","Capon","MEM","LPM","MUSIC"))
    # Mark nominal incident angles
    for p in range(len(theta_list)):
        axes.axvline(linestyle = '--',linewidth = 2,color = 'black',x = theta_list[p])
    #axes.axvline(linestyle = '--',linewidth = 2,color = 'black',x = 115)
    

#demo_coherent_signals()
#demo_single_signal(M=4)