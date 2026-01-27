# -*- coding: utf-8 -*-
"""
                                                   PyArgus
                                            Direction Estimation 


     Description:
     ------------
       Implements Direction of Arrival estimation methods for antenna arrays.

        Implemented DOA methods:

            - Bartlett method
            - Capon's method
            - Burg's Maximum Entropy Method (MEM)
            - Multiple Signal Classification (MUSIC)
            - Multi Dimension MUSIC (MD-MUSIC)

        Corr matrix estimation functions:    
            - Sample Matrix Inversion (SMI)
            - Froward-Backward averaging
            - Spatial Smoothing
        

     Authors: Tamás Pető

     License: GPLv3

     Changelog :
         - Ver 1.0000    : Initial version (2016 12 26)
         - Ver 1.1000    : Reformated code (2017 06 02)
	     - Ver 1.1001    : Improved documentation and comments (2018 02 21) 
         - Ver 1.1500    : Algorithms now expects scanning vector matrix insted of array alignment 
                           to support more generic anntenna alignments (2018 09 01)
    
    
"""

import numpy as np
import numpy.linalg as lin
import matplotlib.pyplot as plt

def DOA_Bartlett(R, scanning_vectors):
    """                                 
                    Fourier(Bartlett) - DIRECTION OF ARRIVAL ESTIMATION

        
        
        Description:
        ------------    
           The function implements the Bartlett method for direction estimation
     
           Calculation method : 
		                                                  H         
		                PAD(theta) = S(theta) * R_xx * S(theta)  
    
    
        Parameters:
        ----------- 
          
            :param R: spatial correlation matrix
            :param scanning_vectors : Generated using the array alignment and the incident angles 
                       
            :type R: 2D numpy array with size of M x M, where M is the number of antennas in the antenna system
            :tpye scanning vectors: 2D numpy array with size: M x P, where P is the number of incident angles 
               
       Return values:
       --------------
    	
            :return PAD: Angular distribution of the power ("Power angular densitiy"- not normalized to 1 deg)    
	        :rtype PAD: numpy array
         
            :return -1, -1: Input spatial correlation matrix is not quadratic
            :return -2, -2: dimension of R not equal with dimension of the antenna array 
    
"""
     
    # --- Parameters ---      
        
    # --> Input check
    if np.size(R, 0) != np.size(R, 1):
        print("ERROR: Correlation matrix is not quadratic")
        return -1, -1
    
    if np.size(R, 0) != np.size(scanning_vectors, 0):
        print("ERROR: Correlation matrix dimension does not match with the antenna array dimension")
        return -2, -2
        
    PAD = np.zeros(np.size(scanning_vectors, 1),dtype=complex) 
    
    # --- Calculation ---     
    theta_index=0
    for i in range(np.size(scanning_vectors, 1)): 
        S_theta_ = scanning_vectors[:, i]
        PAD[theta_index]=np.dot(np.conj(S_theta_),np.dot(R,S_theta_))
        theta_index += 1
         
    return PAD

def DOA_Capon(R, scanning_vectors):
    """                                 
                    Capon's method - DIRECTION OF ARRIVAL ESTIMATION

        
        
        Description:
        ------------    
            The function implements Capon's direction of arrival estimation method

            Calculation method : 
        	
                                                  1
                          SINR(theta) = ---------------------------            	                                        
                                            H        -1        		                         
                                     S(theta) * R_xx * S(theta)  
        
        Parameters:
        -----------                
            :param R: spatial correlation matrix
            :param scanning_vectors : Generated using the array alignment and the incident angles 
                       
            :type R: 2D numpy array with size of M x M, where M is the number of antennas in the antenna system            
            :tpye scanning vectors: 2D numpy array with size: M x P, where P is the number of incident angles 
            
       Return values:
       --------------
       
            :return ADSINR:  Angular dependenet signal to noise ratio     
	        :rtype ADSINR: numpy array
         
            :return -1, -1: Input spatial correlation matrix is not quadratic
            :return -2, -2: dimension of R not equal with dimension of the antenna array 
            :return -3, -3: Spatial correlation matrix is singular
    """
    # --- Parameters ---  
    
    # --> Input check
    if np.size(R, 0) != np.size(R, 1):
        print("ERROR: Correlation matrix is not quadratic")
        return -1, -1
    if np.size(R, 0) != np.size(scanning_vectors, 0):
        print("ERROR: Correlation matrix dimension does not match with the antenna array dimension")
        return -2, -2        

     
    ADSINR = np.zeros(np.size(scanning_vectors, 1),dtype=complex)

    # --- Calculation ---  
    try:
        R_inv  = np.linalg.inv(R) # invert the cross correlation matrix
    except:
        print("ERROR: Signular matrix")
        return -3, -3
       
    theta_index=0
    for i in range(np.size(scanning_vectors, 1)):             
        S_theta_ = scanning_vectors[:, i]
        ADSINR[theta_index]=np.dot(np.conj(S_theta_),np.dot(R_inv,S_theta_))
        theta_index += 1
    
    ADSINR = np.reciprocal(ADSINR)
        
    return ADSINR


def DOA_MEM(R, scanning_vectors, column_select = 0 ):
    """                                 
                    Maximum Entropy Method - DIRECTION OF ARRIVAL ESTIMATION

        
        
        Description:
         ------------    
            The function implements the MEM method for direction estimation

    
            Calculation method : 
            
                                                  1
                        PAD(theta) = ---------------------------
                                             H        H 
                                      S(theta) * rj rj  * S(theta)     
        Parameters:
        -----------                
            :param R: spatial correlation matrix
            :param scanning_vectors : Generated using the array alignment and the incident angles                         
            :param column_select: Selects the column of the R matrix used in the MEM algorithm (default : 0)                    
            
            :type R: 2D numpy array with size of M x M, where M is the number of antennas in the antenna system                        
            :tpye scanning vectors: 2D numpy array with size: M x P, where P is the number of incident angles                       
            :type column_select: int
            
       Return values:
       --------------
       
            :return PAD: Angular distribution of the power ("Power angular densitiy"- not normalized to 1 deg)    
	        :rtype : numpy array
         
            :return -1, -1: Input spatial correlation matrix is not quadratic
            :return -2, -2: dimension of R not equal with dimension of the antenna array 
            :return -3, -3: Spatial correlation matrix is singular
    """
    # --- Parameters ---  
    
    # --> Input check
    if np.size(R, 0) != np.size(R, 1):
        print("ERROR: Correlation matrix is not quadratic")
        return -1, -1
    
    if np.size(R, 0) != np.size(scanning_vectors, 0):
        print("ERROR: Correlation matrix dimension does not match with the antenna array dimension")
        return -2, -2                
    
    PAD = np.zeros(np.size(scanning_vectors,1),dtype=complex)

    # --- Calculation ---                            
    try:
        R_inv  = np.linalg.inv(R) # invert the cross correlation matrix
    except:
        print("ERROR: Signular matrix")
        return -3, -3
        
    # Create matrix from one of the column of the cross correlation matrix with
    # dyadic multiplication    
    R_invc = np.outer( R_inv [:,column_select],np.conj(R_inv[:,column_select]))   

    theta_index=0
    for i in range(np.size(scanning_vectors,1)):             
        S_theta_ = scanning_vectors[:, i]
        PAD[theta_index]=np.dot(np.conj(S_theta_),np.dot(R_invc,S_theta_))
        theta_index += 1
    
    PAD = np.reciprocal(PAD)
        
    return PAD
    

def DOA_LPM(R, scanning_vectors, element_select, angle_resolution = 1):
    """                                 
                    LPM - Linear Prediction method

        
        
        Description:
         ------------    
           The function implements the Linear prediction method for direction estimation

           Calculation method : 
                                                  H    -1
                                                 U    R    U
                        PLP(theta) = ---------------------------
                                          |    H   -1           |2
                                          |   U * R  * S(theta) | 
    
            
        Parameters:
        -----------                
            :param R: spatial correlation matrix
            :param scanning_vectors : Generated using the array alignment and the incident angles                         
            :param element_select: Antenna element index used for the predection.
            :param angle_resolution: Angle resolution of scanning vector s(theta) [deg] (default : 1)
           
            :type R: 2D numpy array with size of M x M, where M is the number of antennas in the antenna system            
            :tpye scanning vectors: 2D numpy array with size: M x P, where P is the number of incident angles                       
            :type element_select: int
            :type angle_resolution: float      
   
       Return values:
       --------------
       
            :return PLP : Angular distribution of the power ("Power angular densitiy"- not normalized to 1 deg)    
	        :rtype : numpy array
         
            :return -1, -1: Input spatial correlation matrix is not quadratic
            :return -2, -2: dimension of R not equal with dimension of the antenna array 
            :return -3, -3: Spatial correlation matrix is singular
    """
    # --- Parameters ---  
    
    # --> Input check
    if np.size(R, 0) != np.size(R, 1):
        print("ERROR: Correlation matrix is not quadratic")
        return -1, -1
    
    if np.size(R, 0) != np.size(scanning_vectors, 0):
        print("ERROR: Correlation matrix dimension does not match with the antenna array dimension")
        return -2, -2                
        
    PLP = np.zeros(np.size(scanning_vectors,1),dtype=complex)
    
    # --- Calculation ---      
    try:
        R_inv  = np.linalg.inv(R) # invert the cross correlation matrix
    except:
        print("ERROR: Signular matrix")
        return -3, -3

    R_inv = np.array(R_inv)
    M = np.size(scanning_vectors,0)
    
    # Create element selector vector
    u = np.zeros(M,dtype=complex)
    u[element_select] = 1
    u = np.array(u).T       
    
    theta_index=0
    for i in range(np.size(scanning_vectors,1)):             
        S_theta_ = scanning_vectors[:, i]
        S_theta_ = np.array(S_theta_).T
        PLP[theta_index]=  np.real(np.conj(u.T) @ R_inv @ u) / np.abs(np.conj(u.T)@ R_inv @ S_theta_)**2            
        theta_index += 1   
 
    return PLP



def DOA_MUSIC(R, scanning_vectors, signal_dimension, angle_resolution = 1):
    """                                 
                    MUSIC - Multiple Signal Classification method

        
        
        Description:
         ------------    
           The function implements the MUSIC method for direction estimation
           
           Calculation method : 

                                                    1
                        ADORT(theta) = ---------------------------
                                             H        H 
                                      S(theta) * En En  * S(theta)
         Parameters:
        -----------                
            :param R: spatial correlation matrix            
            :param scanning_vectors : Generated using the array alignment and the incident angles                                                 
            :param signal_dimension:  Number of signal sources    
                       
            :type R: 2D numpy array with size of M x M, where M is the number of antennas in the antenna system                            
            :tpye scanning vectors: 2D numpy array with size: M x P, where P is the number of incident angles                                   
            :type signal_dimension: int
            
       Return values:
       --------------
       
            :return  ADORT : Angular dependent orthogonality. Expresses the orthongonality of the current steering vector to the 
                    noise subspace
            :rtype : numpy array
         
            :return -1, -1: Input spatial correlation matrix is not quadratic
            :return -2, -2: dimension of R not equal with dimension of the antenna array 
            :return -3, -3: Spatial correlation matrix is singular
    """
    # --- Parameters ---  
    
    # --> Input check
    if np.size(R, 0) != np.size(R, 1):
        print("ERROR: Correlation matrix is not quadratic")
        return -1, -1
    
    if np.size(R, 0) != np.size(scanning_vectors, 0):
        print("ERROR: Correlation matrix dimension does not match with the antenna array dimension")
        return -2, -2                    
    
    ADORT = np.zeros(np.size(scanning_vectors, 1),dtype=complex)
    M = np.size(R, 0)
    
    # --- Calculation ---
    # Determine eigenvectors and eigenvalues
    sigmai, vi = lin.eig(R)
    # Sorting    
    eig_array = []
    for i in range(M):
        eig_array.append([np.abs(sigmai[i]),vi[:,i]])
    eig_array = sorted(eig_array, key=lambda eig_array: eig_array[0], reverse=False)
    
    # Generate noise subspace matrix
    noise_dimension = M - signal_dimension    
    E = np.zeros((M,noise_dimension),dtype=complex)
    for i in range(noise_dimension):     
        E[:,i] = eig_array[i][1]     
        
    E = np.array(E)    
    
    theta_index=0
    for i in range(np.size(scanning_vectors, 1)):             
        S_theta_ = scanning_vectors[:, i]
        S_theta_  = np.array(S_theta_).T
        ADORT[theta_index]=  1/np.abs(np.conj(S_theta_.T)@(E@np.conj(E.T))@S_theta_)
        theta_index += 1
         
    return ADORT

def DOAMD_MUSIC(R, array_alignment, signal_dimension, coherent_sources=2, angle_resolution = 1,):
    """                                 
                    MD-MUSIC - Multi Dimensional Multiple Signal Classification method

        
        
         Description:
         ------------    
           The function implements the MD-MUSIC method for direction estimation
           
           Calculation method : 
        
                                                    1
                        ADORT(theta) = ---------------------------
                                            H H       H 
                                           A*c * En En  * A c 
                                           
                        A  - Array response matrix
                        C  - Liner combiner vector
                        En - Noise subspace matrix
           
        Implementation notes:
        ---------------------
        
            This function works only for two coherent signal sources. Note that, however the algorithm works
            for arbitrary number of coherent sources, the computational cost increases exponentially, thus
            using this algorithm for higher number of sources is impractical.
        
        Parameters:
        -----------                
         
            :param R: spatial correlation matrix
            :param array_alignment : Array containing the antenna positions measured in the wavelength             
            :param signal_dimension: Number of signal sources    
            :param coherent_sources: Number of coherent sources
            :param angle_resolution: Angle resolution of scanning vector s(theta) [deg] (default : 1)
           
            :type R: 2D numpy array with size of M x M, where M is the number of antennas in the antenna system                            
            :tpye array_alignment: 1D numpy array with size: M x 1 
            :type signal_dimension: int
            :type: coherent_sources: int
            :type angle_resolution: float      
   
       Return values:
       --------------
       
            :return  ADORT : Angular dependent orthogonality. Expresses the orthongonality of the current steering vector to the 
                    noise subspace
            :rtype : L dimensional numpy array, where L is the number of coherent sources
         
            :return -1, -1: Input spatial correlation matrix is not quadratic
            :return -2, -2: dimension of R not equal with dimension of the antenna array 
            
    """
    
    # --- Parameters ---  
    
    # --> Input check
    if np.size(R, 0) != np.size(R, 1):
        print("ERROR: Correlation matrix is not quadratic")
        return -1, -1
    
    if np.size(R, 0) != np.size(array_alignment, 0):
        print("ERROR: Correlation matrix dimension does not match with the antenna array dimension")
        return -2, -2                

    incident_angles = np.arange(0,180+angle_resolution,angle_resolution)
    ADORT = np.zeros((int(180/angle_resolution+1), int(180/angle_resolution+1)), dtype=float)
                     
    M = np.size(R, 0) # Number of antenna elements
    
    # --- Calculation ---
    # Determine eigenvectors and eigenvalues
    sigmai, vi = lin.eig(R)
    # Sorting    
    eig_array = []    
    for i in range(M):
        eig_array.append([np.abs(sigmai[i]),vi[:,i]])
    eig_array = sorted(eig_array, key=lambda eig_array: eig_array[0],
                       reverse=False)    
   
    # Generate noise subspace matrix
    noise_dimension = M - signal_dimension    
    E = np.zeros((M,noise_dimension),dtype=complex)
    for i in range(noise_dimension):     
        E[:,i] = eig_array[i][1]
           
    E = np.array(E)   
    
    theta_index  = 0
    theta2_index = 0
    
         
    for theta in incident_angles:                
        S_theta_  = np.exp(array_alignment*1j*2*np.pi*np.cos(np.radians(theta))) # Scanning vector      
        theta2_index=0
        for theta2 in incident_angles[0:theta_index]:            
            S_theta_2_ = np.exp(array_alignment*1j*2*np.pi*np.cos(np.radians(theta2))) # Scanning vector                                  
            a = np.array(S_theta_+S_theta_2_).T # Spatial signiture vector
            ADORT[theta_index,theta2_index]=  np.real(1/np.abs(np.conj(a.T)@(E@np.conj(E.T))@a))            
            theta2_index += 1
        theta_index += 1
    
    return ADORT, incident_angles


#********************************************************
#*****       CORRELATION MATRIX ESTIMATIONS         *****
#********************************************************
def corr_matrix_estimate(X, imp="mem_eff"):
    """
        Estimates the spatial correlation matrix with sample averaging    
    
    Implementation notes:
    --------------------
        Two different implementation exist for this function call. One of them use a for loop to iterate through the
        signal samples while the other use a direct matrix product from numpy. The latter consumes more memory
        (as all the received coherent multichannel samples must be available at the same time)
        but much faster for large arrays. The implementation can be selected using the "imp" function parameter.
        Set imp="mem_eff" to use the memory efficient implementation with a for loop or set to "fast" in order to use
        the faster direct matrix product implementation.
    
        
    Parameters:
    -----------
        :param X : Received multichannel signal matrix from the antenna array.         
        :param imp: Selects the implementation method. Valid values are "mem_eff" and "fast". The default value is "mem_eff".
        :type X: N x M complex numpy array N is the number of samples, M is the number of antenna elements.
        :type imp: string
            
    Return values:
    -------------
    
        :return R : Estimated spatial correlation matrix
        :rtype R: M x M complex numpy array
        
        :return -1 : When unidentified implementation method was specified
    """      
    N = np.size(X, 0)
    M = np.size(X, 1)
    R = np.zeros((M, M), dtype=complex)    
    
    # --input check--
    if N < M:
        print("WARNING: Number of antenna elements is greather than the number of time samples")
        print("WARNING: You may flipped the input matrix")
    
    # --calculation--
    if imp == "mem_eff":            
        for n in range(N):
            R += np.outer(X[n, :], np.conjugate(X[n, :]))
    elif imp == "fast":
            X = X.T 
            R = np.dot(X, X.conj().T)
    else:
        print("ERROR: Unidentified implementation method")
        print("ERROR: No output is generated")
        return -1
        
    R = np.divide(R, N)
    return R

def extened_mra_corr_mtx(R):
    """
               
        Fill the defficient correlation matrix when the antenna array is 
        placed in MRA (Minimum Redundancy Alignment). To fill the deficient
        elements the Toeplitz and Hermitian property of the correlation matrix
        is utilized.
        
        Currently it works only for quad element linear arrays.
        TODO: Implementation of general cases
        
        Implementation notes:
        ---------------------    
            Correlation coeffcients corresponding to the blind antenna elements
            must be zero in the spatial correlation matrix.
            
            example: Quad element linear array with blind element at the third position
                                    | R11 R12 0   R14 |
                                R=  | R21 R22 0   R24 |
                                    | 0   0   0   0   |
                                    | R41 R42 0   R44 |
        Parameters:
        -----------
        
            :param R : Spatial correlation matrix
            :type  R : M x M complex numpy array, M is the number of antenna elements.        
        
        Return values:
        --------------
        
            :return R: Extended correlation matrix
            :rtype R: 4 x 4 complex numpy array
        
            :return -1, -1: Input spatial correlation matrix is not quadratic
            
    """          
    # --> Input check
    if np.size(R, 0) != np.size(R, 1):
        print("ERROR: Correlation matrix is not quadratic")
        return -1, -1     
    
    if np.size(R,0) == 3 and np.size(R,1) == 3:
        R = np.insert(R,2,0,axis=1)
        R = np.insert(R,2,0,axis=0)
    
    # Fill deficient correlation matrix (Toeplitz matrix)
    R[0, 2] = R[1, 3]
    R[2, 0] = np.conjugate(R[0, 2])

    R[1, 2] = (R[0, 1] + np.conjugate(R[1, 0])) / 2
    R[2, 1] = np.conjugate(R[1, 2])

    R[2, 2] = (R[0, 0] + R[1, 1] + R[3, 3]) / 3

    R[3, 2] = (R[2, 1] + R[1, 0] + np.conjugate(R[1, 2]) + np.conjugate(
        R[0, 1])) / 4
    R[2, 3] = np.conjugate(R[3, 2])
    return R

def forward_backward_avg(R):
    """
        Calculates the forward-backward averaging of the input correlation matrix
        
    Parameters:
    -----------
        :param R : Spatial correlation matrix
        :type  R : M x M complex numpy array, M is the number of antenna elements.        
            
    Return values:
    -------------
    
        :return R_fb : Forward-backward averaged correlation matrix
        :rtype R_fb: M x M complex numpy array           
        
        :return -1, -1: Input spatial correlation matrix is not quadratic
            
    """          
    # --> Input check
    if np.size(R, 0) != np.size(R, 1):
        print("ERROR: Correlation matrix is not quadratic")
        return -1, -1 
    
    # --> Calculation
    M = np.size(R, 0)  # Number of antenna elements
    R = np.array(R)

    # Create exchange matrix
    J = np.eye(M)
    J = np.fliplr(J) 
    J = np.array(J)
    
    R_fb = 0.5 * (R + J@np.conjugate(R)@J)

    return np.array(R_fb)
         
    
def spatial_smoothing(X, P, direction="forward"): 
    """ 
     
        Calculates the forward and (or) backward spatially smoothed correlation matrix
        
    Parameters:
    -----------
        :param X : Received multichannel signal matrix from the antenna array.         
        :param P : Size of the subarray
        :param direction: 

        :type X: N x M complex numpy array N is the number of samples, M is the number of antenna elements.
        :type P : int
        :type direction: string
            
    Return values:
    -------------
    
        :return R_ss : Forward-backward averaged correlation matrix
        :rtype R_ss: P x P complex numpy array    
        
        -1: direction parameter is invalid
    """
    # --input check--
    N = np.size(X, 0)  # Number of samples 
    M = np.size(X, 1)  # Number of antenna elements    

    if N < M:
        print("WARNING: Number of antenna elements is greather than the number of time samples")
        print("WARNING: You may flipped the input matrix")
    L = M-P+1 # Number of subarrays     
    Rss = np.zeros((P,P), dtype=complex) # Spatiali smoothed correlation matrix 
     
    if direction == "forward" or direction == "forward-backward":            
        for l in range(L):             
            Rxx = np.zeros((P,P), dtype=complex) # Correlation matrix allocation 
            for n in np.arange(0,N,1): 
                Rxx += np.outer(X[n,l:l+P],np.conj(X[n,l:l+P])) 
            np.divide(Rxx,N) # normalization 
            Rss+=Rxx                 
    if direction == "backward" or direction == "forward-backward":         
        for l in range(L): 
            Rxx = np.zeros((P,P), dtype=complex) # Correlation matrix allocation 
            for n in np.arange(0,N,1): 
                d = np.conj(X[n,M-l-P:M-l] [::-1]) 
                Rxx += np.outer(d,np.conj(d)) 
            np.divide(Rxx,N) # normalization 
            Rss+=Rxx         
    if not (direction == "forward" or direction == "backward" or direction == "forward-backward"):     
        print("ERROR: Smoothing direction not recognized ! ") 
        return -1 
        
    # normalization            
    if direction == "forward-backward": 
        np.divide(Rss,2*L)  
    else: 
        np.divide(Rss,L)  
         
    return Rss 

def estimate_sig_dim(R):
    """
        Estimates the signal subspace dimension for the MUSIC algorithm
        
        Notes: Identifying the subspace dimension with K-mean clustering is not
               verified nor theoretically nor experimentally, thus using this 
               function is not recommended. 
        
     Parameters:
    -----------
        :param R : Spatial correlation matrix
        :type  R : M x M complex numpy array, M is the number of antenna elements.        
            
    Return values:
    -------------
    
            :return signal_dimension : Estimated signal dimension
            :rtype signal_dimension: int
            
            :return -1, -1: Input spatial correlation matrix is not quadratic
            
    """     
    from scipy.cluster import vq     
    
    # --> Input check
    if np.size(R, 0) != np.size(R, 1):
        print("ERROR: Correlation matrix is not quadratic")
        return -1, -1     
    print("WARNING: This function is experimental")
    
    # Identify dominant eigenvalues a.k.a signal subspace dimension with K-Mean clutering
    sigmai, vi = lin.eig(R)    
    eigenvalues = np.abs(sigmai)
    centroids, variance = vq.kmeans(eigenvalues,2)
    identified, distance = vq.vq(eigenvalues, centroids)
    
    cluster_1 = eigenvalues[identified == 0]
    cluster_2 = eigenvalues[identified == 1]
    print(cluster_1)
    print(cluster_2)
    print(centroids)
    
    if centroids[0] > centroids[1]:
        signal_dimension = len(eigenvalues[identified == 0])
    else:
        signal_dimension = len(eigenvalues[identified == 1])
    
    return signal_dimension


#********************************************************
#*****            ARRAY UTIL FUNCTIONS              *****
#********************************************************
def gen_ula_scanning_vectors(array_alignment, thetas):
    """
    Description:
    ------------
        This function prepares scanning vectorors for Linear array antenna systems
        
    Parameters:
    -----------

        :param array_alignment : A vector containing the distances between the antenna elements.
                                e.g.: [0, 0.5*lambda, 1*lambda, ... ]
        :param  thetas : A vector containing the incident angles e.g.: [0deg, 1deg, 2deg, ..., 180 deg]
        
        :type array_alignment: 1D numpy array
        :type thetas: 1D numpy array
            
    Return values:
    -------------
    
        :return scanning_vectors : Estimated signal dimension
        :rtype scanning_vectors: 2D numpy array with size: M x P, where P is the number of incident angles
        
    """
    M = np.size(array_alignment, 0)  # Number of antenna elements    
    scanning_vectors = np.zeros((M, np.size(thetas)), dtype=complex)
    for i in range(np.size(thetas)):    
        scanning_vectors[:, i] = np.exp(array_alignment*1j*2*np.pi*np.cos(np.radians(thetas[i]))) # Scanning vector      
        
    return scanning_vectors
    
def gen_uca_scanning_vectors(M, r, thetas):    
    """
    Description:
    ------------
        This function prepares scanning vectorors for Uniform Circular Array antenna systems
        
    Parameters:
    -----------

        :param M : Number of antenna elements on the circle
        :param r : radius of the antenna system
        :param thetas : A vector containing the incident angles e.g.: [0deg, 1deg, 2deg, ..., 180 deg]
        
        :type M: int
        :type R: float
        :type thetas: 1D numpy array
            
    Return values:
    -------------
    
        :return scanning_vectors : Estimated signal dimension
        :rtype scanning_vectors: 2D numpy array with size: M x P, where P is the number of incident angles
        
    """
    scanning_vectors = np.zeros((M, np.size(thetas)), dtype=complex)
    for i in range(np.size(thetas)):    
        for j in np.arange(0,M,1):   
            scanning_vectors[j, i] = np.exp(1j*2*np.pi*r*np.cos(np.radians(thetas[i]-j*(360)/M))) # UCA   
        
    return scanning_vectors

def gen_scanning_vectors(M, x, y, thetas):
    """
    Description:
    ------------
        This function prepares scanning vectorors for general antenna array configurations        
        
    Parameters:
    -----------

        :param M : Number of antenna elements on the circle
        :param x : x coordinates of the antenna elements on a plane
        :param y : y coordinates of the antenna elements on a plane
        :param thetas : A vector containing the incident angles e.g.: [0deg, 1deg, 2deg, ..., 180 deg]
        
        :type M: int
        :type x: 1D numpy array
        :type y: 1D numpy array
        :type R: float
        :type thetas: 1D numpy array
            
    Return values:
    -------------
    
        :return scanning_vectors : Estimated signal dimension
        :rtype scanning_vectors: 2D numpy array with size: M x P, where P is the number of incident angles
        
    """
    scanning_vectors = np.zeros((M, np.size(thetas)), dtype=complex)
    for i in range(np.size(thetas)):        
        scanning_vectors[:,i] = np.exp(1j*2*np.pi* (x*np.cos(np.deg2rad(thetas[i])) + y*np.sin(np.deg2rad(thetas[i]))))    
    
    return scanning_vectors
#********************************************************
#*****          ALIASING UTIL FUNCTIONS             *****
#********************************************************
def alias_border_calc(d):
    """
        Calculate the angle borders of the aliasing region for ULA antenna systems
    Parameters:
    -----------        
        :param d: distance between antenna elements [lambda]
        :type d: float
    
    Return values:
    --------------
        :return anlge_list : Angle borders of the unambious region
        :rtype anlge_list: List with two elements
    """
    theta_alias_min = np.rad2deg(np.arccos(1/(2*d)))
    theta_alias_max = np.rad2deg(np.arccos(1/d -1))
    return (theta_alias_min,theta_alias_max)
   
#********************************************************
#*****                DISPLAY FUNCTIONS             *****
#********************************************************
def DOA_plot(DOA_data, incident_angles, log_scale_min=None, alias_highlight=True, d=0.5, axes=None):
    
    DOA_data = np.divide(np.abs(DOA_data),np.max(np.abs(DOA_data))) # normalization
    if(log_scale_min != None):        
        DOA_data = 10*np.log10(DOA_data)                
        theta_index = 0        
        for theta in incident_angles:                    
            if DOA_data[theta_index] < log_scale_min:
                DOA_data[theta_index] = log_scale_min
            theta_index += 1                     
        
    if axes is None:
        fig = plt.figure()
        axes  = fig.add_subplot(111)
    
    #Plot DOA results  
    axes.plot(incident_angles,DOA_data)    
    axes.set_title('Direction of Arrival estimation ',fontsize = 16)
    axes.set_xlabel('Incident angle [deg]')
    axes.set_ylabel('Amplitude [dB]')   

    # Alias highlight
    if alias_highlight:
        (theta_alias_min,theta_alias_max) = alias_border_calc(d)        
        print('Minimum alias angle %2.2f '%theta_alias_min)
        print('Maximum alias angle %2.2f '%theta_alias_max)
    	
        axes.axvspan(theta_alias_min, theta_alias_max, color='red', alpha=0.3) 
        axes.axvspan(180-theta_alias_min, 180, color='red', alpha=0.3) 
        
        axes.axvspan(180-theta_alias_min, 180-theta_alias_max, color='blue', alpha=0.3) 
        axes.axvspan(0, theta_alias_min, color='blue', alpha=0.3) 

    plt.grid()   
    return axes