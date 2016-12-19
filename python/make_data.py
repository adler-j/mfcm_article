import odl
import numpy as np
import pickle
import lcr_data  # private, you will not have this.

phantom_type = 'forbild'  # 'head' or 'forbild'
nphoton = 15000.0

# Create domain
domain = odl.uniform_discr([-1, -1], [1, 1], [448, 448])

# Create and show the phantom (forbild phantom)
if phantom_type == 'forbild':
    phantom = odl.phantom.forbild(domain)
elif phantom_type == 'head':
    # This requires access to the private lcr_data repository
    pdata = lcr_data.davids_head_density(shape=(448, 448, 66))
    phantom = domain.element(pdata[..., 30])
else:
    assert False

phantom.show('phantom')

# Create data with some fake "scatter"
angle_partition = odl.uniform_partition(0, np.pi, 1000)
detector_partition = odl.uniform_partition(-2, 2, 1000)
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)
ray_trafo = odl.tomo.RayTransform(domain, geometry, impl='astra_cuda')

sinogram = ray_trafo(phantom)
exp_sinogram = np.random.poisson(nphoton * np.exp(-sinogram)) / nphoton
exp_sinogram = ray_trafo.range.element(exp_sinogram)
sinogram_with_scatter = -np.log(exp_sinogram + 0.3)

fbp_op = odl.tomo.fbp_op(ray_trafo, filter_type='Hamming', filter_cutoff=0.6)

y = fbp_op(sinogram_with_scatter)
y *= np.sum(phantom) / np.sum(y)
y.show()

target_file = open('reconstruction_{}.dmp'.format(phantom_type), 'w')
pickle.dump(y, target_file)
target_file.close()
