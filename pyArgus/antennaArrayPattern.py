# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
"""
                                                      PyArgus
                                        antenna array radiation pattern plot


     Description:
     ------------
        Displays the radiation pattern of the antenna array applying the given weighting coefficients


    Authors: Pető Tamás
    License: GPLv3
"""


def array_rad_pattern_plot(array_alignment, w=None, sing_elem_patterns=None, axes=None, log_scale_min=-50):
    """
        Description:
        ------------
           Displays the radiation pattern of an antenna array applying the given weighting coefficients. This function
           supports planar antenna systems.
	
             - This function supports full antenna array radiation pattern plot
               TOTAL = SINGLE X ARRAY
             - Supports radiation pattern plot from atenna element coordinates
             - Plot to externally specified axes
           
           
        Implementation notes:
        ---------------------
        
            As this plot function use the antenna elements absolut positions to calculate the radiation patern it does not 
            apply the plane wave approximations analyticaly. In order to deal with this, the far-field observer is placed 
            to 10^6 times greather distance than the distant element in the antenna system.
    	 
    
        Parameters:
        -----------          
             
             :param array_alignment : Contains the positions of the antenna elements in a two dimensional numpy array. 
             :param axes : Specify when the radiation pattern is requested to plot on the given axes.
             :param w : Complex valued weight coefficients (default : 1 1 ... 1)
             :param log_scale_min : Minimum plot value in logarithmic scale. (default value is -50 dB)
             :param sing_elem_pattern : (numpy array) Single antenna element radiation pattern [dBi-deg array] (default : 0...0)
                                       This radiation pattern must be contain data values in a range of 0 -180 deg.


    	     :type array_alignment : 2D numpy array.The first row stores the "x", while the second row stores the "y" 
    				     coordinates of the antenna elements. The distance unit is lambda, where lambda 
    				     is the center wavelength of the processed signal.
    	     :type axes : matplotlib generated figure axes object. 
    	     :type w : complex numpy array. Its dimension should be equal with the number of antenna elements.
    	     :type log_scale_min : float	
    	     :type sing_elem_pattern : numpy array
 	
       Return values:
       -----------------

            :return pattern_theta : Absolut value of the calculated radiation pattern in logarithmic scale.
            :rype pattern_theta : numpy array	
    """

    # --- Plot parameters ---

    # (These parameters are not configurable externally)        
    angle_resolution = 0.1  # [deg]
    angle_range = 360  # 180 or 360 [deg]
    far_field_dist = 10**6 * np.max(np.sqrt(array_alignment[0]**2+array_alignment[1]**2))
    
    N = np.size(array_alignment[0])  # Determine the size of the antenna system    
    if w is None:  # Check coefficient vector 
        w = np.ones(N, dtype=complex)  # No coefficient vector is specified. Use uniform.

    # --- Calculation ---
    incident_angles = np.arange(0, angle_range+angle_resolution, angle_resolution)
    
    # Used to handle the single element radiation pattern
    if sing_elem_patterns is not None:
        # TODO: Check sing_elem_patterns dimensions!
        # Create interpolated radiation pattern    
        orig_angle_res = angle_range/(np.size(sing_elem_patterns[0, :])-1)
        orig_angles = np.arange(0, angle_range + orig_angle_res, orig_angle_res)
        
        # Interpolation
        sing_elem_patterns_interp = np.zeros((N, np.size(incident_angles)), dtype=float) # Allocation 
        for m in range(N):
            sing_elem_patterns_interp[m,:]  =  np.interp(incident_angles , orig_angles, sing_elem_patterns[m,:])          
        sing_elem_patterns = sing_elem_patterns_interp
        print("INFO: Single element radiation patterns have fitted with interpolation")
    else:
        sing_elem_patterns = np.zeros((N, np.size(incident_angles)))

    # Calculate radiation pattern    
    AF = np.zeros(int(angle_range/angle_resolution)+1, dtype=complex)
    theta_index = 0
    for theta in incident_angles:                
        r0 = far_field_dist * np.array(([np.cos(np.deg2rad(theta))], [np.sin(np.deg2rad(theta))]))        
        r = np.tile(r0,(1,N)) - array_alignment                
        r_abs = np.sqrt(r[0]**2+r[1]**2)                
        s_theta = np.exp(-1j * 2 * np.pi * r_abs)  # steering vector
        
        # Apply single element pattern
        s_theta *= 10**(sing_elem_patterns[:, theta_index]/10)        
        
        # Applying weight coefficients
        AF[theta_index] = np.inner(np.conjugate(w), s_theta) # Array Factor
        theta_index += 1
    
    # --- Display ---
    #AF = np.divide(AF,np.max(np.abs(AF)))  # normalization
    AF_log = 20*np.log10(abs(AF))
    theta_index = 0
    for theta in incident_angles:
        if AF_log[theta_index] < log_scale_min:
            AF_log[theta_index] = log_scale_min
        theta_index += 1
    
    if axes is None:
        fig = plt.figure()
        axes  = fig.add_subplot(111)
    
    axes.plot(incident_angles,AF_log)
    axes.set_title( "Radiation pattern")
    axes.set_xlabel("Incident angle [deg]")
    axes.set_ylabel("Amplitude [dBi]")
    
    return AF_log
