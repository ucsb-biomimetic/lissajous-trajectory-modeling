There are four python files for demonstrating Lissajous trajectory modeling, along with a set of example data in the folder 'M30D-1977-1980' which needs to be decompressed before use.

Two are infrastructural, model.py and preproc.py.  Model.py defines the general model of periodic motion in tensorflow, while preproc.py provides functions for extracting salient features from data traces that are then passed to the trajectory model.

The script fit_model.py uses a set of 6 recordings from calibration targets (3 for each axis) to fit the model to a particular non-ideal lissajous pattern.  When it is done, it saves the weights in a python pickle file.

The script reconstruct_images.py uses the trained weights from fit_model.py to assemble images from data traces.  It produces images in several configurations, including converting each trace to a series of frames (as in video), and as a sequence of partial reconstructions of a single image frame.  The reconstruction algorithm will be somewhat slow, since it directly uses the model instead of first converting the model into a lookup table.

Configuration of the model complexity and selection of a folder with calibration data are settings found at the top of their respective python files.  No options are taken at execution.