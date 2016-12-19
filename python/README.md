Python codes for MFCM
#####################

This folder contains python codes to perform the MFCM bias field correction method.

The codes are implemented using [ODL](https://github.com/odlgroup/odl), which can be installed via

    pip install odl

To use the data generation scripts that are shipped with this code, you also need [ASTRA-TOOLBOX](https://github.com/astra-toolbox/astra-toolbox) installed. It can be installed using conda with

    conda install -c astra-toolbox astra-toolbox

Content
#######

The files are as follows:

* `make_data.py`, generates a "fake scatter" volume for examples
* `example_coordinate_descent.py`, the method described in the article
* `example_coordinate_descent_l2_reg`, method with l2 regularization of bias
* `example_coordinate_descent_l2_reg_tichonov`, method with smoothness regularization of the bias
