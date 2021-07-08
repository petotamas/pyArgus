# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

import plotly.graph_objects as go
from plotly.offline import plot

"""
                                                      PyArgus
                                antenna array radiation pattern calculation and plot


     Description:
     ------------
        Calculated and displays the radiation pattern of an antenna array applying the given weighting coefficients


    Authors: Pető Tamás

    License: GPLv3

    Changelog :
        - Ver 1.0000   : Initial version (2015 07 13)
        - Ver 1.0100   : Weight coefficients can be set externally (2015 08 07)
        - Ver 1.0200   : Input parameter set reconfigured (2015 08 21)
        - Ver 1.1000   : Antenna array alignments can be configured (2015 03 23)
        - Ver 1.1100   : Reformatted source code (2017 04 11)
        - Ver 1.2000   : Array factor calculation based on antenna positions (2017 04 21) 
        - Ver 1.2001   : Improved documentation and comments (2018 02 16) 	
        - Ver 1.3000   : Single element pattern for the individual antenna elements (2018 10 23) 
        - Ver 1.4000   : Fix scaling [N. Baranyai] (2021 05 17)
        - Ver 1.5000   : Separate radiation pattern calulcation and plotting, plotly compatible figure (2021 07 08)
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
             
             :param array_alignment   : Contains the positions of the antenna elements in a two dimensional numpy array. 
             :param axes              : Specify when the radiation pattern is requested to plot on the given axes.
             :param w                 : Complex valued weight coefficients (default : 1 1 ... 1)
             :param log_scale_min     : Minimum plot value in logarithmic scale. (default value is -50 dB)
             :param sing_elem_pattern : (numpy array) Single antenna element radiation pattern [dBi-deg array] (default : 0...0)
                                       This radiation pattern must be contain data values in a range of 0 - 360 deg.

    	     :type array_alignment   : 2D numpy array.The first row stores the "x", while the second row stores the "y" 
                    				   coordinates of the antenna elements. The distance unit is lambda, where lambda 
                                       r wavelength of the processed signal.
    	     :type axes              : matplotlib generated figure axes object. 
    	     :type w                 : complex numpy array. Its dimension should be equal with the number of antenna elements.
    	     :type log_scale_min     : float	
    	     :type sing_elem_pattern : numpy array
 	
       Return values:
       -----------------

            :return pattern_theta : Absolut value of the calculated radiation pattern in logarithmic scale.
            :rype pattern_theta : numpy array	
    """
    print("WARNING: Using this function is obsolete, please consider using the 'array_rad_pattern()' and 'pattern_plot()' functions instead")
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
        s_theta *= 10**(sing_elem_patterns[:, theta_index]/20)        
        
        # Applying weight coefficients
        AF[theta_index] = np.inner(np.conjugate(w), s_theta) # Array Factor
        theta_index += 1
    
    # --- Display ---
    #AF = np.divide(AF,np.max(np.abs(AF)))  # normalization    
    AF_log  = 20*np.log10(abs(AF))
    AF_log -= 10*np.log10(np.sum(np.abs(w)))
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

def array_rad_pattern(array_alignment, incident_angles, w=None, sing_elem_patterns=None):
    """
     Description:
        ------------
           Calculates the radiation pattern of an antenna array applying the given weighting coefficients and taking into
           consideration the radiation patterns of the single antenna elements.
           This function supports planar antenna systems.
	
             - This function supports full antenna array radiation pattern plot
               TOTAL = SINGLE X ARRAY
             - Supports radiation pattern plot from atenna element coordinates            
             - The function allows to use different type of single radiation elements in the array.
             - The calculated radiation pattern is normalized in a way that it shows the achiveable 
               signal-to-noise ratio improvement in a given direction. (Calculations does not take into account the inter-element couplings)
                        
           
        Implementation notes:
        ---------------------
        
            This function use the antenna elements absolut positions to calculate the radiation patern, it does not 
            apply the plane wave approximations analyticaly. In order to deal with this, the far-field observer is placed 
            to 10^6 times greather distance than the distant element in the antenna system.
    	 
    
        Parameters:
        -----------          
             
             :param array_alignment   : Contains the positions of the antenna elements in a two dimensional numpy array.              
             :param w                 : Complex valued weight coefficients (default : 1 1 ... 1)
             :param sing_elem_pattern : (numpy array) Single antenna element radiation pattern [dBi-deg array] (default : 0...0)
                                       This radiation pattern array must contain the pattern values on the range of 0 - 360 deg.
                                       The first column should contain the incident angles, while the remaining columns should specify the directivity
                                       values of the current single radiation element.                                       
                                       

    	     :type array_alignment   : 2D numpy array.The first row stores the "x", while the second row stores the "y" 
                    				   coordinates of the antenna elements. The distance unit is lambda, where lambda 
                                       r wavelength of the processed signal.    	     
    	     :type w                 : complex numpy array. Its dimension should be equal with the number of antenna elements.
    	     :type sing_elem_pattern : (M+1 x D) numpy array, where M is the number of elements and D is the number of incident angles on which
                                       radiation patterns are specified.
 	
       Return values:
       -----------------

            :return AF_log : Absolut value of the calculated radiation pattern in logarithmic scale.
            :rype   AF_log : numpy array	
    """
    # -- Input check --
    M = np.size(array_alignment[0])  # Determine the size of the antenna system    
    
    if sing_elem_patterns is not None and (sing_elem_patterns.shape[0] != M+1):
        print("ERROR: Single element radiation pattern array must have M+1 column, where M is the number of antenna elements")
        print("ERROR: No output is generated")
        return None
    
    far_field_dist = 10**6 * np.max(np.sqrt(array_alignment[0]**2+array_alignment[1]**2))
    
    if w is None:  # Check coefficient vector 
        w = np.ones(M, dtype=complex)  # No coefficient vector is specified. Use uniform.
    
    # Used to handle the single element radiation pattern
    if sing_elem_patterns is not None:
        # TODO: Check sing_elem_patterns dimensions!
        
        # Interpolation
        sing_elem_patterns_interp      = np.zeros((M+1, np.size(incident_angles)), dtype=float) # Allocation        
        for m in range(M):
            sing_elem_patterns_interp[m+1,:]  =  np.interp(incident_angles , sing_elem_patterns[0,:], sing_elem_patterns[m+1,:])          
        sing_elem_patterns = sing_elem_patterns_interp
        print("INFO: Single element radiation patterns have fitted with interpolation")
    else:
        sing_elem_patterns = np.zeros((M+1, np.size(incident_angles)))
    
    sing_elem_patterns[0, :] = incident_angles[:]

    # Calculate radiation pattern    
    AF = np.zeros(len(incident_angles), dtype=complex)    
    for theta_index, theta in enumerate(incident_angles):
        r0 = far_field_dist * np.array(([np.cos(np.deg2rad(theta))], [np.sin(np.deg2rad(theta))]))        
        r = np.tile(r0,(1,M)) - array_alignment                
        r_abs = np.sqrt(r[0]**2+r[1]**2)                
        s_theta = np.exp(-1j * 2 * np.pi * r_abs)  # steering vector
        
        # Apply single element pattern
        s_theta *= 10**(sing_elem_patterns[1:, theta_index]/20)        
        
        # Applying weight coefficients
        AF[theta_index] = np.inner(np.conjugate(w), s_theta) # Array Factor
        theta_index += 1
    
    AF_log  = 20*np.log10(abs(AF))
    AF_log -= 10*np.log10(np.sum(np.abs(w)))

    return AF_log

def plot_pattern(incident_angles, pattern, fig=None, log_scale_min=-50):
    """
        Plots the radiation pattern on a polar plot using the plotly library
        
        Parameters:
        ----------
            :param pattern       : Radiation pattern to plot. The first column should be the incident angles while the second 
                                   should be the directivity values in dBi
            :param fig           : Plotly compatible figure object, if not specified a new object will be created [default:None]
            :param log_scale_min : Radiation pattern values that are less than this threshold will be truncated
            
            :type pattern      : 2D numpy array
            :type fig          : Plotly compatible figure object
            :type log_scale_min: float
       
        Return values:
       -----------------

            :return fig : Figure object
            :rype   fig : Plotly compatible figure object
    """    # Remove extreme low values
    pattern = [log_scale_min if pattern_i < log_scale_min else pattern_i for pattern_i in pattern]

    if fig is None:
        fig =go.Figure()

    fig.add_trace(go.Scatterpolar(r = pattern, theta = incident_angles))
    fig.update_layout(title_text="Radiation pattern")
    plot(fig)
    
    return fig


