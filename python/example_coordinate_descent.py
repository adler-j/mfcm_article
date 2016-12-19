import odl
import numpy as np
import pickle


# SELECTABLE PARAMETERS
alpha = 0.3
filter_width = 0.02  # standard deviation of the Gaussian filter
niter = 10
c = [0.0, 0.9, 1.5]
m = 2

# Load precomputed data
y = pickle.load(open('reconstruction_head.dmp'))
y.show('data')

domain = y.space

# --- Segmentation starts here ---

# Create the "conv" operator that adds neighbor regularization

# neighborhood
ft = odl.trafos.FourierTransform(domain)
const = filter_width ** 2 / 4.0 ** 2
gaussian = ft.range.element(lambda x: np.exp(-(x[0] ** 2 + x[1] ** 2) * const))
convolution = ft.inverse * gaussian * ft

diag = odl.IdentityOperator(domain)
conv = diag + alpha * convolution

# Create initial guess
mu = [np.less(y, (c[0] + c[1]) / 2),
      np.logical_and(np.greater_equal(y, (c[0] + c[1]) / 2), np.less(y, (c[1] + c[2]) / 2)),
      np.greater_equal(y, (c[1] + c[2]) / 2)]
x = y.copy()

callback = (odl.solvers.CallbackShow(display_step=1) &
            odl.solvers.CallbackPrintIteration())

# Store some values
I = len(c)
spatial_domain = y.space
ones = spatial_domain.one()  # ones.inner(x)  gives integral of x
conv_adj_mu_pow_m = [None] * I

# Iteratively solve the coordinate descent equations
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

    # Find lambda
    const = ones.inner(1 / sum(conv_adj_mu_pow_m[i] for i in range(I)))
    integrand = (y -
                 sum(c[i] * conv_adj_mu_pow_m[i] for i in range(I)) /
                 sum(conv_adj_mu_pow_m[i] for i in range(I)))
    lam = ones.inner(integrand) / const

    # Update x
    x.assign((lam + sum(c[i] * conv_adj_mu_pow_m[i] for i in range(I))) /
             sum(conv_adj_mu_pow_m[i] for i in range(I)))

    callback(x)
    print(c)

for i in range(3):
    mu[i].show('m[{}]'.format(i))

y.show('measured values', clim=[1, 1.2])
x.show('estimated true values', clim=[1, 1.2])