# pyArgus

This python package aims to implement signal processing algorithms applicable in antenna arrays. The implementation mainly focuses on the beamforming and
direction finding algorithms.
For array synthesis and radiation pattern optimization please check the "arraytool" python package.
https://github.com/zinka/arraytool and https://zinka.wordpress.com/ by S. R. Zinka

Named after Argus the giant from the greek mitology who had hundreds of eyes.

### Package organization:

- pyArgus: Main package
	- antennaArrayPattern: Implements the radiation pattern calculation of antenna arrays
	- beamform: Implements beamformer algorithms.
	- directionEstimation: Implements DOA estimation algorithms and method for estimating the spatial correlation matrix.
- test: Contains demonstration functions for antenna pattern plot, beamforming and direction of arrival estimation. 
- docs: Documentation of the package, written in Jupyter notebook.


### Implemented Algorithms

- Beamforiming:
    - Fixed beamformers:
        - Maximum Signal to Interference Ratio beamformer
        - Maximum Signal to Interference Ratio beamformer with Godara's method
    - Adaptive beamformer:
        - Optimum Wiener beamformer (with known signal of interest direction)
        - MSINR with known covariance matrices
        - MMSE with known signal of interest

- Direction of Arrival Estimation:
    - DOA algorithms:
        - Bartlett (Fourier) method
        - Capon's method
        - Burg's Maximum Entropy Method (MEM)
        - Multiple Signal Classification (MUSIC)
        - Multi Dimension MUSIC (MD-MUSIC)

    - Util functions:
        - Spatial correlation matrix estimation using the sample average technique
        - Forward-backward averaging
        - Spatial smoothing
        - DOA results plot with highlighting the ambiguous regions (Only for Uniform linear arrays)

### Antenna Array Pattern Plot Features
- Arbitrary configured planar antenna systems
- Takes into account the pattern of the signal radiating elements

### Install from the Python Package Index

```
pip install pyargus
```

Personal website: [tamaspeto.com](https://www.tamaspeto.com/pyargus) 

Tamás Pető 2016-2021, Hungary


