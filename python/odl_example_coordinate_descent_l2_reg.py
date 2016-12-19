import odl
import numpy as np
import pickle


# SELECTABLE PARAMETERS
alpha = 0.1
filter_width = 0.02  # standard deviation of the Gaussian filter
lam = 0.5
I = 3
m = 2
niter = 20

# Load precomputed data
y = pickle.load(open('reconstruction.dmp'))
y.show('data')

# Create domain
domain = y.space

# Create and show the phantom (forbild phantom)
phantom = odl.phantom.forbild(domain)
phantom.show('phantom')

# Create data with some fake "scatter"


# --- Segmentation starts here ---

# Create the "conv" operator that adds neighbor regularization

# neighbor
ft = odl.trafos.FourierTransform(domain)
c = filter_width ** 2 / 4.0 ** 2
gaussian = ft.range.element(lambda x: np.exp(-(x[0] ** 2 + x[1] ** 2) * c))
convolution = ft.inverse * gaussian * ft

diag = odl.IdentityOperator(domain)
conv = diag + alpha * convolution

# Create initial guess
mu = [np.less(y, 0.2),
              np.logical_and(np.greater_equal(y, 0.2), np.less(y, 0.5)),
              np.greater_equal(y, 0.5)]
x = y.copy()
c = [0, 0.35, 0.8]

callback = (odl.solvers.CallbackShow(display_step=1) &
            odl.solvers.CallbackPrintIteration())

# --- Solve problem ---

spatial_domain = y.space
ones = spatial_domain.one()  # ones.inner(x)  gives integral of x

conv_adj_mu_pow_m = [None] * I

for _ in range(niter):
    # Update mu
    for i in range(I):
        summand = spatial_domain.zero()
        convi = conv((x - c[i])**2)
        for j in range(I):
            convj = conv((x - c[j])**2)
            summand += np.power(convi / convj, 1 / (m - 1))
        mu[i] = 1 / summand
        conv_adj_mu_pow_m[i] = conv.adjoint(np.power(mu[i], m))

    # Update c
    for i in range(I):
        dividend = ones.inner(conv_adj_mu_pow_m[i] * x)
        divisor = ones.inner(conv_adj_mu_pow_m[i])
        c[i] = dividend / divisor

    # Update x
    x.assign((lam * y + sum(c[i] * conv_adj_mu_pow_m[i] for i in range(I))) /
             (lam + sum(conv_adj_mu_pow_m[i] for i in range(I))))

    callback(x)
    print(c)


for i in range(3):
    mu[i].show('m[{}]'.format(i))

(y - x).show('b')