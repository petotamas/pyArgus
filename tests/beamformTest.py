 #-*- coding: utf-8 -*-
"""
********************************************************
*****          Demonstration and test              *****
********************************************************

PyARGUS 

Demonstration functions written to test the different beamformer algorithms

Tamás Pető
1, may 2017

"""
from pyargus.antennaArrayPattern import array_rad_pattern_plot
import numpy as np
import matplotlib.pyplot as plt
import pyargus.beamform as bf


def demo_fixed_max_sir():
    """
        Demonstration function for the maximum signal to interference ratio beamformer
    """
    
    # Incident angle of the signal of interest is: 130 [deg]
    # Incident angle of interferences are: 50,60,70 [deg]
    par_angles = np.array([130, 50, 60, 70])  # Incident angle definition
    par_constraints = np.array([1, 0, 0, 0]).reshape(4,1)  # Constraint vector definition
    par_array_alignment = np.array([0.5, 1, 1.5, 2])  # Vector used to describe the antenna system
    
    w_maxsir = bf.fixed_max_sir_beamform(par_angles, par_constraints, par_array_alignment)  # Calculate coefficients
    
    # Plot the obtained beampattern
    figure = plt.figure()        
    ax = figure.add_subplot(111)
       
    # mark incident angles on the figure
    ax.axvline(linestyle = '--',linewidth = 2,color = 'r',x = 130)    
    ax.axvline(linestyle = '--',linewidth = 2,color = 'black',x = 50)
    ax.axvline(linestyle = '--',linewidth = 2,color = 'black',x = 60)
    ax.axvline(linestyle = '--',linewidth = 2,color = 'black',x = 70)
    
    array_alignment = np.array(([0.5, 1, 1.5, 2],[0, 0, 0, 0]))
    array_rad_pattern_plot(w=w_maxsir, array_alignment = array_alignment, axes =ax)
    
def demo_fixed_max_sir_mra():
    """
        Demonstration function for the maximum signal to interference ratio beamformer
        The anttena array is aligned in minimum redundance alginment (MRA)
                                x  x  o  x
        - "x" denotes the active antenna positions
        - "0" denotes the inactive antenna position
    """

    par_angles = np.array([130, 50, 70])
    par_constraints = np.array([1, 0, 0]).reshape(3,1)
    par_array_alignment = np.array([0.5, 1, 2])
    
    w_maxsir = bf.fixed_max_sir_beamform(par_angles, par_constraints, par_array_alignment)
    

    array_alignment = np.array(([0.5, 1, 2],[0, 0, 0]))
    array_rad_pattern_plot(w=w_maxsir, array_alignment = array_alignment)

def demo_fixed_max_sir_Godara():
    """
        Demonstration function for the Godara's maximum signal to interference ratio beamformer
    """
    
    angles      = np.array([130,80])
    constraints = np.array([1,0])
    array_alignment = np.array([0.5, 1, 2])
    
    # Calculate coefficient vector
    w_Godara_maxsir = bf.Goadar_max_sir_beamform(angles, constraints, array_alignment) 
    
    # Plot the obtained beampattern
    figure = plt.figure()        
    ax = figure.add_subplot(111)
       
    # mark incident angles on the figure
    ax.axvline(linestyle = '--',linewidth = 2,color = 'r',x = angles[0])    
    ax.axvline(linestyle = '--',linewidth = 2,color = 'black',x = angles[1])    
        
    p_array_alignment = np.array(([0.5, 1, 2],[0, 0, 0]))
    array_rad_pattern_plot(w = w_Godara_maxsir, array_alignment = p_array_alignment, axes =ax) 


def simulate_test_signal(theta_soi, theta_interf, power_soi, power_interf, 
                         power_noise, N, M, d):
    """
        This function used to generate test signal for the demonstration of the 
        adaptive beamformermers. Two type of signals is distinguised. One is 
        the signal of interest (SOI), while the others are interference sources.       
        Use the "theta" parameters to set the incident angles of the signals and
        use the "power" parameters to set the average power of these signals.
        
        Implementation notes:
        ---------------------
        
            Test signals including the desired signal and the interferences 
            are drawn from complex normal distribution with having zero mean 
            and sqrt(power_***) variance, where *** = soi or interf[k].
            
            This test function is only applicable for uniform linear antenna
            arrays.
        
        Parameters:
        -----------
        
            theta_soi    : (float) Incident angle of the SOI
            theta_interf : (float numpy array) Incident angles of the 
                           interference sources
            power_soi    : (float) power of the SOI
            power_interf : (float numpy array) Powers of the 
                            interference sources
            power_noise  : (float) Power of the thermal noise.
            N            : (int) Number of samples in the generated simulation
                           signal
            M            : (int) Number of antenna elements
            d            : (float) distance between antenna elements
        Return values:
        --------------
        
            X: (M x N complex numpy array) Simulated signal, that could be received
               from "M" antenna element. N is the number of samples received from 
               each elements.
            SOI: (N element complex numpy array) Simulated signal of interest
    """   
   
    # Allocate array for the generated simulation signal
    X = np.zeros((M,N), dtype=complex)   
   
    # --Create SOI    
    i = np.arange(M)
    # Create array response vector for SOI
    aS = np.exp(i*1j*2*np.pi*d*np.cos(np.deg2rad(theta_soi))) 
    
    # Generate desired signal
    s  = np.random.normal(0, np.sqrt(power_soi)/2) + \
         1j* np.random.normal(0, np.sqrt(power_soi)/2, N) 
    
    X += np.outer(aS, s)
    
    # --Create interference signals
    for k in np.arange(np.size(theta_interf)):    
        # Array response vector for the k-th interference signal
        aI_k = np.exp(i*1j*2*np.pi*d*np.cos(np.deg2rad(theta_interf[k])))  
        
        # Generate k-th interference signal
        i_k = np.random.normal(0, np.sqrt(power_interf[k])/2) + \
              1j* np.random.normal(0, np.sqrt(power_interf[k])/2, N) 
        X [:] += np.outer(aI_k, i_k)          
    
    # -- Add noise
    X += np.random.normal(0, np.sqrt(power_noise)/2) + \
         1j* np.random.normal(0, np.sqrt(power_noise)/2, N) 
    
    return X, s

def demo_optimal_Wiener_beamform():
    """
        Demonstrates the opertion of the optimal Wiener beamformer.
        
    """
    
    # -- Parameters --
    
    theta_soi = 90
    theta_interf = np.array([ 120, 140])
    power_soi = 10**-5
    power_interf = np.array([10**4, 10**4])
    power_noise  = 1
    N = 10**5  # Number of simulated signal samples
    M = 4  # Number of antenna elements 
    d = 0.5  # Distance between antenna elements
    
    # -- Calculation --
    
    # Create simulation signal
    X_rec, x_soi = simulate_test_signal(theta_soi, theta_interf, power_soi, power_interf, power_noise, N, M ,d)
    
    # Calculate cross correlation matrix
    R = bf.estimate_corr_matrix(X_rec.T, imp="fast")
    
    # Create array resonse vector for the desired signal angle
    aS = np.exp(np.arange(M)*1j*2*np.pi*d*np.cos(np.deg2rad(theta_soi))) 
    aS = np.matrix(aS).reshape(M,1)
    
    # Calculate optimal weight coefficient vector
    w_opt = bf.optimal_Wiener_beamform(R,aS )
    w_opt /= np.sqrt(np.dot(w_opt,w_opt.conj()))  # normalize coefficients
    
    # -- Display results --    
    
    # Create a figure instance to plot on
    figure = plt.figure()
    
    # Create an axis object
    ax = figure.add_subplot(111)
    
    # Array alignment matrix for radiation pattern plot
    p_array_alignment = np.array((np.arange(M) * d, np.zeros(M)))
    #print(p_array_alignment)
    # Plot radiation pattern with the calculated coeffcients
    array_rad_pattern_plot(w = w_opt, array_alignment = p_array_alignment, axes =ax)
    
    # Place angle markers
    ax.axvline(linestyle = '--',linewidth = 2,color = 'r',x = theta_soi)    
    for k in np.arange(np.size(theta_interf)):
        ax.axvline(linestyle = '--',linewidth = 2,color = 'black',x = theta_interf[k])
  
    w_mmse = bf.MMSE_beamform(np.transpose(X_rec), np.transpose(x_soi))    
    w_mmse /= np.sqrt(np.dot(w_mmse,w_mmse.conj()))  # normalize coefficients
    array_rad_pattern_plot(w = w_mmse, array_alignment = p_array_alignment, axes =ax)

def demo_msinr(theta_soi, theta_interf):
    """
        Demonstration code for MSINR method with known Rss and Rnunu
    
    """
    Pnoise   = 0.001              # noise variance  
    d        = 0.5                # distance between antenna elements [lambda]
    N        = 4                  # number of antenna elements
    
    i = np.arange(N)
    
    # Create array response vector for SOI
    aS = np.exp(i*1j*2*np.pi*d*np.cos(np.deg2rad(theta_soi))) 
    aS = np.matrix(aS).reshape(N,1)
    # Create SOI autocorrelation matrix
    Rss = aS * aS.getH() 
    
    # Create interference autocorrelation matrix
    Rnunu = np.matrix(np.zeros((N,N)))
    
    for k in np.arange(np.size(theta_interf)):    
        aI = np.exp(i*1j*2*np.pi*d*np.cos(np.deg2rad(theta_interf[k])))  
        aI = np.matrix(aI).reshape(N,1)
        
        # Create interference autocorrelation matrix ( interferece signals are not correlated ) 
        Rnunu = Rnunu + aI * aI.getH()
    
    # Create noise autocorrelation matrix
    Rnn = np.matrix(np.eye(N)) * Pnoise
    
    # Create noise + interferences autocorr matrix (interferences and thermal noise are not correlated)    
    Rnunu = Rnunu + Rnn
    
    # a figure instance to plot on
    figure = plt.figure()
    
    # create an axis
    ax = figure.add_subplot(111)
       
    # mark incident angles on the figure
    ax.axvline(linestyle = '--',linewidth = 2,color = 'r',x = theta_soi)
    
    for k in np.arange(np.size(theta_interf)):
        ax.axvline(linestyle = '--',linewidth = 2,color = 'black',x = theta_interf[k])
    
    
    #Calculate MSINR solution
    SINR,w_msinr = bf.MSINR_beamform(Rss,Rnunu)
    w_msinr /= np.sqrt(np.dot(w_msinr,w_msinr.conj()))
    
    p_array_alignment = np.array(([0.5, 1, 1.5, 2],[0, 0, 0, 0]))
    array_rad_pattern_plot(w = w_msinr,axes = ax, array_alignment = p_array_alignment) 
    print('Signal to interference and noise ratio :',np.abs(SINR)) 
    
    #plt.hold
    #Calculate MSINR solution with an alternate form
    w_msinr = bf.optimal_Wiener_beamform(Rnunu = Rnunu,aS = aS)
    w_msinr /= np.sqrt(np.dot(w_msinr,w_msinr.conj()))
    array_rad_pattern_plot(w = w_msinr,axes = ax, array_alignment = p_array_alignment)

def demo_peigen(heta_soi, theta_interf, power_soi, power_noise, power_interf):
    """
        Demonstrates the opertion of the optimal Wiener beamformer.
        
        Implementation notes:
        ---------------------
        This simulation use a quad channel linear antenna array model with 
        half-wave interelement spacing. The number of simulated signal samples
        is 10^5.

        Parameters:
        -----------        
        theta_soi: Incident angle of the signal of interest (float)
        theta_inter: Incident angles of the interference sources (numpy array)
        power_soi:  Averaged power of the soi
        power_noise: Averaged power of the uncorrelated noise
        power_interf: Averaged power of the interference sources       
        
        Return values:
        --------------        
        None
        
        The obtained radiation pattern is plotted onto a matplotlib figure.
        
    """
    
    # -- Parameters --    
    N = 10**5  # Number of simulated signal samples
    M = 4  # Number of antenna elements 
    d = 0.5  # Distance between antenna elements
    
    # -- Calculation --
    
    # Create simulation signal
    X_rec, x_soi = simulate_test_signal(theta_soi, theta_interf, power_soi, power_interf, power_noise, N, M ,d)
    
    # Calculate cross correlation matrix
    R = bf.estimate_corr_matrix(X_rec.T, imp="fast")
    
    # Create array resonse vector for the desired signal angle
    aS = np.exp(np.arange(M)*1j*2*np.pi*d*np.cos(np.deg2rad(theta_soi))) 
    aS = np.matrix(aS).reshape(M,1)
    
    # Calculate optimal weight coefficient vector
    w_pe = bf.peigen_bemform(R,aS,2 )
    w_pe /= np.sqrt(np.dot(w_pe,w_pe.conj()))  # normalize coefficients
  
    # -- Display results --    
    
    # Create a figure instance to plot on
    figure = plt.figure()
    
    # Create an axis object
    ax = figure.add_subplot(111)
    
    # Array alignment matrix for radiation pattern plot
    p_array_alignment = np.array((np.arange(M) * d, np.zeros(M)))
    #print(p_array_alignment)
    # Plot radiation pattern with the calculated coeffcients
    array_rad_pattern_plot(w = w_pe, array_alignment = p_array_alignment, axes =ax)
   
    # Place angle markers
    ax.axvline(linestyle = '--',linewidth = 2,color = 'r',x = theta_soi)    
    for k in np.arange(np.size(theta_interf)):
        ax.axvline(linestyle = '--',linewidth = 2,color = 'black',x = theta_interf[k])

theta_soi = 30  # Incident angle of the signal of interest
theta_interf = np.array([ 156, 80])  # Incident angles of the interferences 
power_soi = 10**-5  #  Power of the SOI
power_interf = np.array([10**4, 10**4])  # Power of the interferences
power_noise  = 1  # Power of the uncorrelated thermal noise    

#demo_peigen(theta_soi, theta_interf, power_soi, power_noise, power_interf)
#demo_fixed_max_sir_Godara()
#demo_msinr(75, np.array([120,150]))
#demo_optimal_Wiener_beamform()