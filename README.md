# ERP-beamformer
A beamformer filtering approach to estimating ERP component amplitudes.
Written by Marijn van Vliet.

This is a packaged version of the code used in this paper:

van Vliet, M.; Chumerin, N.; De Deyne, S.; Wiersema, J.R.; Fias, W.; Storms, G.; Van Hulle, M.M., "Single-Trial ERP Component Analysis Using a Spatiotemporal LCMV Beamformer," in Biomedical Engineering, IEEE Transactions on , vol.63, no.1, pp.55-66, Jan. 2016
doi: 10.1109/TBME.2015.2468588
URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7202846&isnumber=7360844

## Installation

It is highly recommended to use the python version of the code. This code has been thouroughly tested. This package depends on the NumPy, SciPy and Scikit-Learn packages. To install the python version of the code, execute:

    python setup.py install

The MATLAB version of the code can be found in the `matlab/` folder. Add this folder to your MATLAB path. This version has less functionality and is less thouroughly tested.
