import numpy as np
import sigpy.mri.rf as rf
import scipy.io as sio
from algo import InterleavedShim
import matplotlib.pyplot as pyplot

# Demo to test algorithm performance. Initialize with 10 random initializers,
# and compare performance of algorithm using "full batch" iterations vs highly undersampled iterations.


data = sio.loadmat('data/demo_invivo_b1_slice.mat')
mask_slice = data['mask']
b1 = data['b1']  # [x y nslice ncoil] dims

test_slice = np.transpose(b1,(2, 0, 1))
test_slice = test_slice * mask_slice
datasize = 80
nSlices = 1

nTrials = 15
nIter = 300
beta = 1E-8 # small enough that RF power cost is not really a factor. Will see error reducing in batch cost.
batchsize = 12 # number of rows used in inexact iterations; this is 1.5 x 8 channels (column dimension)

plotting = False
verbose = False

convergence_array = np.zeros((nTrials,nIter-1))
cost_mat_minibatch = np.zeros((nSlices, nTrials))
cost_mat_batch = np.zeros((nSlices, nTrials))

n_nonzero_rows = np.count_nonzero(mask_slice)
print(f'n_nonzero_rows = {n_nonzero_rows}')

# set complex target magnetization 
y = np.expand_dims(mask_slice.flatten()+0j*np.zeros(np.size(mask_slice)),1)

# 'single spoke' trajectory for shimming - we're just staying @ the center of k-space
k1 = np.expand_dims(np.array((0, 0, 0)), 0)
A = rf.PtxSpatialExplicit(test_slice, k1, dt=0,
                          img_shape=np.shape(test_slice)[1:],
                          ret_array=True)
Nc = np.shape(A)[1]

for ii in range(nTrials):
    print(f'Trial # {ii + 1}')
    # generate a random initializer, to be used for both methods
    x0 = (np.random.randn(Nc, 1) + 1j * np.random.randn(Nc, 1))

    # Shim with the batch algorithm 
    # Simply use same code with ALL b1 locations kept (batchsize = all rows)
    # Inefficient implementation but allows for direct comparison with variations in code controlled.
    # Gives batch GS an unfair advantage since ALSO calculating cost per iter
    batchsize = n_nonzero_rows
    opt = InterleavedShim(x0, A, y, batchsize, beta,
                     step_max=nIter, full_GS_per_iter=2, plotting=plotting, cost_func='rmse',
                     solver='pinv', verbose=verbose,)

    opt.optimize()

    # record RMSE error
    cost_mat_batch[0, ii] = opt.best_cost

    print(f'FINAL BATCH ERROR = {opt.best_cost}')

    # Shim with the minibatch algorithm
    batchsize = 12
    opt = InterleavedShim(x0, A, y, batchsize, beta,
                     step_max=nIter, full_GS_per_iter=2, plotting=plotting, cost_func='rmse',
                     solver='pinv', verbose=verbose,)

    opt.optimize()

    # record RMSE error
    cost_mat_minibatch[0, ii] = opt.best_cost

    print(f'FINAL MINIBATCH ERROR = {opt.best_cost}')


# Boxplot the distribution of RMSEs for both methods
pyplot.boxplot(np.concatenate([cost_mat_batch.T,cost_mat_minibatch.T], axis=1))
pyplot.xticks([1, 2], ['exact GS', 'inexact GS (proposed)'])
pyplot.ylabel('cost')
pyplot.show()
