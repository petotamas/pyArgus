 #-*- coding: utf-8 -*-
import numpy as np
import numpy.linalg as lin
import matplotlib.pyplot as plt
"""

                                                   PyArgus
                                                 Beamfroming


     Description:
     ------------
        The beamform module of the PyArgus library implements antenna array processing methods which are intended to
        calculate weight coefficients for beamformers. These techniques can operate either in a fixed or in an adaptive
        manner.
        The following functions implement fixed beamforming strategies:
            - fixed_max_sir_beamform()
            - Godara_max_sir_beamform()

        The implemented adaptive beamformers are the following:
            - Minimum Variance Distortionless Response (MVDR)

     Authors: Pető Tamás

     License: GPLv3

     Changelog :
         - Ver 1      : Initial version (2016)
         - Ver 1.1    : Reformated code (2017 04 23)


     Version format: Ver X.ABCD
                         A - Complete code reorganization
                         B - Processing stage modification or implementation
                         C - Algorithm modification or implementation
                         D - Error correction, Status display modification
 """


def fixed_max_sir_beamform(angles, constraints, array_alignment):
    """
        Description:
        ------------

            The fixed maximum signal to interference (MaxSIR) beamformer calculates the weight coefficient in a way to
            ensure that the resulting radiation pattern satisfies the preliminary specified requirements. The calculated
            beampattern will have the required responses in the specified directions.
            The detailed description of the algorithm can be found in: 
            "Frank B Gross: Smart Antennas With MATLAB, 2nd edition 2015 , section 8.3.1"

        Parameters:
        -----------

            angles: (M by 1 numpy array) List of the angles that are constrained. Must have M elements for an M element
                     antenna array
            constraints: (M by 1 numpy array) List of the requested responses. Has the same size as the angle list has.
            array_alignment: Contains the positions of the antenna elements in the linear antenna array. The distance
                             unit is lambda, where lambda is the center wavelength of the processed signal.



        Return values:
        --------------
            W: (M by 1 complex numpy array) weight coefficient vector. M is the number antenna elements in the array.
            np.empty(0) : If the input parameters are not meet the requirements.
    """

    # --- Input check ---
    if np.size(constraints) != np.size(angles):
        print("ERROR : The angle and the constraint vector do not have the same size")
        return np.empty(0)

    N = np.size(array_alignment)  # number of antenna elements

    # --- Calculation ---

    # Create array response matrix: A
    A = np.zeros((N,N),dtype=complex)
    for k in range(N):
        A[:, k] = np.exp(array_alignment*1j*2*np.pi*np.cos(np.deg2rad(angles[k])))
    print(A)
    # Calculate coefficient vector
    A = np.array(A) # convert to matrix
    u = np.array(constraints) # convert to vector
    A_inv = lin.inv(A) # invert A matrix
    w_max_sir = np.conj((u.T @ A_inv).T) # solve linear equation

    return w_max_sir

    
def Goadar_max_sir_beamform(angles, U, array_alignment):
    """
                    Godara Fixed Maximum Signal to Interference Ratio Beamformer
    
        Description:
         ------------    
           Calculates the weight coefficient using the constraints for fixed incident angles
           The number of constraints and angels can be less than the degree of freedeom af the antenna system
           The detailed description of the algorithm can be found in: 
           "Frank B Gross: Smart Antennas With MATLAB, 2nd edition 2015 , section 8.3.1"

        Implementation notes:
        --------------------
            Noise with variance of 0.001 is added to signal to prevent singularity.
    
        Parameters:
        ------------------
          
            angles          : list of the incident angles [deg]            
            U               : Constraint vector       
            
            
       Return values:
       -----------------
    
            w_Godara_maxsir : (complex numpy array) Calculated coefficient vector
    """
    noc  = np.size(U)  # Number of constraints
        
    if(np.size(angles) != np.size(U)):
        print('Beamformer : Not enough incident angles or constraints')
        return 0 
    
    N = np.size(array_alignment)    
    
    # -- Calculation --    
    A = np.zeros((N,noc),dtype=complex)  # Array response matrix    

    # Create array response matrix
    for k in range(noc):        
        A[:, k] = np.exp(array_alignment*1j*2*np.pi*np.cos(np.deg2rad(angles[k])))
       
    A = np.array(A) # Change to matrix object
    U = np.array(U) # Change to matrix object
    
    I = np.array(np.eye(N)) # Identity matrix
    
    sigmaN2 = 0.001
    aux    = (A@np.conj(A.T) + sigmaN2 * I)
    aux_inv = lin.inv(aux)
    
    w_Godara_maxsir = U@(np.conj(A.T) @ aux_inv)
    
    w_Godara_maxsir = np.conj(w_Godara_maxsir.T)

    return w_Godara_maxsir
    


def MSINR_beamform(Rss,Rnunu,aS = None):
    """                                 
                    Maximum Signal to Noise and Interference Ratio Beamformer
    
        Description:
        ------------    
           
            Calculates the weight coefficient vector in such a manner that it optimizes the Signal to interference + noise ratio                      
            The detailed description of the algorithm can be found in:  
            "Frank B Gross: Smart Antennas With MATLAB, 2nd edition 2015 , section 8.3.1"        
        
            This function it solves the eigenvalue equation using "Rss" and "Rnu" to obtain the optimal MSINR coefficent vector.
            

        
        Parameters:
        ----------- 
          
            Rss   : (N x N complx numpy matrix) Autocorrelation matrix of the signal of interest
                    (where N is the number of antenna elements)
            Rnunu : (N x N complex numpy matrix) Autocorrelation matrix of the noise + interference signals
            aS    : (N x 1 complex numpy vector) Array response vector of the signal of interest 
       Return values:
        -----------------
            
            max sinr: (float) Maximum signal to interference plus noise ratio value
            w_msinr: (complex numpy array) Calculated coefficient vector    

    """
    # -- Calculation --    
        
    # Convert input arrays to matrix form    
    Rss =  np.array(Rss)  
    Rnunu =  np.array(Rnunu)  
    
    # Calcaulating product matrix
    prodMatrix = lin.inv(Rnunu) @ Rss
    
    #calcaulte eigenvectors and eigenvalues
    w,v = np.linalg.eig(prodMatrix)
    
    # search max eigenvalue - this value is equal to the SINR
    maxEigenVal = w[np.argmax(np.abs(w))]
    
    # eigenvector belongs to the maximum eigenvalue is equal to the optimal solution
    w_msinr = v[:,np.argmax(np.abs(w))]
       
    return maxEigenVal, w_msinr
        
    


def optimal_Wiener_beamform(Rnunu, aS):
    """                                 
                    Optimal soluation of the Wiener filtering for beamforiming
    
        Description:
        ------------    
            This function calculates the optimum Wiener solution. To obtain this it uses the
            array resonse of the signal of intereset and the autocorrelation matrix of the interference
            plus noise signals. In many real situation this is substituted with the autocorrelation matrix 
            of the received signal which is comprised of the signal of inteters, the interferences and the noise.
            "Frank B Gross: Smart Antennas With MATLAB, 2nd edition 2015 , section 8.3.1"   
            
            Note that the optimal Wiener beamformer coincides with MSINR or the MVDR beamformer coefficents
            excpeting a scalar multiplier value!
            
            To calculate the coefficients the incident angle of the desired signal must be known 
            preliminary.
            
        Parameters:
        -----------           

            Rnunu : (M x M complex numpy matrix) Autocorrelation matrix of the noise + interference signals                    
            aS    : (M x 1 complex numpy vector) Array response vector of the signal of interest 
       
       Return values:
        -----------------            
            
            w_msinr: (complex numpy array) Calculated coefficient vector    
            None   : Input data was not consistens
    """
    # -- Input check --
    # Number of antenna elements must be the same    
    if np.size(Rnunu,0) != np.size(aS, 0):
        print("ERROR: Input data matrices do not have consistent shape")
        print("ERROR: Weight coefficient vector is not calculated")
        return None
    
    # The correlation matrix must be qudratic
    if np.size(Rnunu,0) != np.size(Rnunu,1):
        print("ERROR: The correlation matrix is not quadratic")
        print("ERROR: Weight coefficient vector is not calculated")
        return None
    
    
    # The correlation vector must have the shape of M x 1
    if np.size(aS,1) != 1:
        print("ERROR: aS must be a vector")
        print("ERROR: Weight coefficient vector is not calculated")
        return None
    
    # -- Calculation --    
    # Change Rnunu array to matrix form
    Rnunu = np.array(Rnunu) # noise+interference autocorrelation matrix
    
    # Change array response vector to matrix form
    aS =  np.array(aS)
    
    w_msinr = lin.inv(Rnunu) @ aS    

    return w_msinr


def MMSE_beamform(received_signal, desired_signal):
    """                                 
                    Minimum Mean Square Error beamformer
    
        Description:
        ------------    
            Calculates the optimal coefficient vector in a way to minimize the
            mean square of the difference between the desired signal and the 
            signal at the output of the beamformer.
            
            The desired signal must be known preliminary to calculate the
            coefficient.
            
        Parameters:
        -----------           

            received_signal : (N x M complex numpy matrix) Time samples of the received multichannel signal
            desired_signal  : (N x 1 complex numpy vector) Time samples of the signal of interest 
       
       Return values:
        -----------------            
            
            w_mmse: (complex numpy array) Calculated coefficient vector    
            None   : Input data was not consistens
    """
    
    # -- Input check --
    
    # -- Calculation --
    nae = np.size(received_signal,1) # Number of antenna elements
    nos = np.size(received_signal,0) # Number of samples
    
    #Calculate cross-correlation matrix
    Rxx = np.zeros((nae,nae),dtype=complex) # Correlation matrix allocation
    for sampleIndex in range(nos):
        Rxx += np.outer(received_signal[sampleIndex,:],np.conj(received_signal[sampleIndex,:]))
        np.divide(Rxx,nos) # normalization
    
    RxxInv = np.linalg.inv(Rxx)
    
    #Calculate cross-correlation vector
    rxd = np.zeros((1,nae),dtype = complex)
    for sampleIndex in range(nos):
        rxd += np.inner(np.conjugate(desired_signal[sampleIndex]),received_signal[sampleIndex,:])
 
    np.divide(rxd,nos)    
    
    w_mmse = np.inner(rxd,RxxInv)

    return w_mmse[0]


def peigen_bemform(Rnunu, aS, peigs):
    """                                 
                                Principal eigenvalue beamformer
    
        Description:
        ------------    
            This beamformer selects the principal eigenvectors of the spatial interference
            correlation matrix. Then it projects the spatial signature vector of the signal
            of interest to the subspace orthogonal to it.
            It allows to select the most dominant interference component only from the 
            interference correlation matrix.
            
            This method is proposed by Michelangelo Villano et al. in
            Antenna Array for Passive Radar: Configuration Design and Adaptive Approaches to Disturbance Cancellation
            International Journal of Antennas and Propagation Volume 2013 
            
        Parameters:
        -----------           

            Rnunu : (M x M complex numpy matrix) Autocorrelation matrix of the noise + interference signals                    
            aS    : (M x 1 complex numpy vector) Array response vector of the signal of interest 
            peigs : (int) Number of principal eigenvalues.
       
       Return values:
        -----------------            
            
            w_eig  : (complex numpy array) Calculated coefficient vector    
            None   : Input data was not consistens
    """
    # -- Input check --
    # Number of antenna elements must be the same    
    if np.size(Rnunu,0) != np.size(aS, 0):
        print("ERROR: Input data matrices do not have consistent shape")
        print("ERROR: Weight coefficient vector is not calculated")
        return None
    
    # The correlation matrix must be qudratic
    if np.size(Rnunu,0) != np.size(Rnunu,1):
        print("ERROR: The correlation matrix is not quadratic")
        print("ERROR: Weight coefficient vector is not calculated")
        return None    
    
    # The correlation vector must have the shape of M x 1
    if np.size(aS,1) != 1:
        print("ERROR: aS must be a vector")
        print("ERROR: Weight coefficient vector is not calculated")
        return None
    
    M = np.size(aS,0)
    
    # -- Calculation --    
    # Change Rnunu array to matrix form
    Rnunu = np.array(Rnunu) # noise+interference autocorrelation matrix
    
    # Change array response vector to matrix form
    aS =  np.array(aS)
    
    I = np.array(np.eye(M))
    
    # Determine eigenvectors and eigenvalues
    sigmai, vi = lin.eig(Rnunu)
    # Sorting    
    eig_array = []
    for i in range(M):
        eig_array.append([np.abs(sigmai[i]),vi[:,i]])
    eig_array = sorted(eig_array, key=lambda eig_array: eig_array[0],
                       reverse=True)
    
    # Generate clutter subspace matrix
    clutter_dim = peigs
    q = np.array(np.zeros((M, clutter_dim),dtype=complex))
    for i in range(clutter_dim):             
        q[:,i] = eig_array[i][1]    
    
    w_pe = (I-q@np.conj(q.T)) @ aS  # Noise subspace
    
    return w_pe
    
def estimate_corr_matrix(X, imp="mem_eff"):
    """
        Estimates the spatial correlation matrix.    
    
    Implementation notes:
    --------------------
        Two different implementation exist for this function call. One of them use a for loop to iterate through the
        signal samples while the other  use a direct vector product from numpy. The latter consumes more memory
        but much faster for large arrays. The implementation can be selected using the "imp" function parameter.
        Set imp="mem_eff" to use the memory efficient implementation with a for loop or set to "fast" in order to use
        the faster direct matrix product implementation.
    
        
    Input parameters:
    ----------------
        X : (N x M complex numpy array) Received signal matrix from the antenna 
            array. N is the number of samples, M is the number of antenna elements.
        imp: (string) Selects the implementation. Valid values are "mem_eff" 
             and "fast". The default value is "mem_eff".
            
    Return values:
    -------------
    
            R : (M x M complex numpy array) Spatial correlation matrix
        None  : Unidentified implementation method was specified
    """      
    N = np.size(X, 0)
    M = np.size(X, 1)
    R = np.zeros((M, M), dtype=complex)
    
    # --input check--
    if N < M:
        print("WARNING: Number of antenna elements is greather than the number of time samples")
        print("WARNING: You may flipped the input matrix")
    
    # --calculation--
    if imp =="mem_eff":            
        for n in range(N):
            R += np.outer(X[n, :], np.conjugate(X[n, :]))
    elif imp == "fast":
            X = X.T 
            R = np.dot(X, X.conj().T)
    else:
        print("ERROR: Unidentified implementation method")
        print("ERROR: No output is generated")
        
    R = np.divide(R, N)
    return R
