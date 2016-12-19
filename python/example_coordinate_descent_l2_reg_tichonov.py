import odl
import numpy as np
import pickle

# SELECTABLE PARAMETERS
alpha = 1.0  # neighbour regularization term
filter_width = 0.03  # standard deviation of the Gaussian filter
lam1 = 0.2   # Penalization of bias
lam2 = 0.01  # Penalization of gradient of bias
niter = 10
c = [0.0, 0.9, 1.5]  # Initial guesses for c
m = 2.0

# Load precomputed data
y = pickle.load(open('reconstruction_head.dmp'))
#y = pickle.load(open('reconstruction_forbild.dmp'))
y.show('data')

domain = y.space

# --- Segmentation starts here ---

# Create the "conv" operator that adds neighbor regularization

# Create convolution with gaussian operator
ft = odl.trafos.FourierTransform(domain)
const = filter_width ** 2 / 4.0 ** 2
gaussian = ft.range.element(lambda x: np.exp(-(x[0] ** 2 + x[1] ** 2) * const))
convolution = ft.inverse * gaussian * ft

# Add extra weight on the diagonal
diag = odl.IdentityOperator(domain)
conv = diag + alpha * convolution

# Create gradient
lap = odl.Laplacian(domain, pad_mode='constant')

# Create initial guess
mu = [np.less(y, (c[0] + c[1]) / 2),
      np.logical_and(np.greater_equal(y, (c[0] + c[1]) / 2),
                     np.less(y, (c[1] + c[2]) / 2)),
      np.greater_equal(y, (c[1] + c[2]) / 2)]
x = y.copy()

callback = (odl.solvers.CallbackShow('bias', display_step=1) &
            odl.solvers.CallbackPrintIteration())

# Store some values
I = len(c)
spatial_domain = y.space
ones = spatial_domain.one()  # ones.inner(x)  gives integral of x
scaling = odl.ScalingOperator(spatial_domain, lam1)
conv_adj_mu_pow_m = [None] * I
cd
# Iteratively solve the coordinate descent equations
for _ in range(niter):
    # Update mu
    for i in range(I):
        summand = spatial_domain.zero()
        convi = conv((x - c[i])**2)
        for j in range(I):
            convj = conv((x - c[j])**2)
            summand += np.power(convi / convj, 1.0 / (m - 1.0))
        mu[i] = 1.0 / summand
        conv_adj_mu_pow_m[i] = conv.adjoint(np.power(mu[i], m))

    # Update c
    for i in range(I):
        dividend = ones.inner(conv_adj_mu_pow_m[i] * x)
        divisor = ones.inner(conv_adj_mu_pow_m[i])
        c[i] = dividend / divisor

    # Update x

    # construct operator
    diag = odl.MultiplyOperator(sum(conv_adj_mu_pow_m[i] for i in range(I)))
    op = scaling - lam2 * lap + diag

    # define right hand side
    rhs = scaling(y) - lam2 * lap(y) + sum(c[i] * conv_adj_mu_pow_m[i] for i in range(I))

    # solve using CG
    odl.solvers.conjugate_gradient(op, x, rhs, niter=256)

    # Display partial results
    callback(y - x)
    print(c)

# display final results

for i in range(3):
    mu[i].show('m[{}]'.format(i))

y.show('measured values', clim=[1, 1.2], saveto='measured.png')
x.show('estimated true values', clim=[1, 1.2], saveto='result.png')