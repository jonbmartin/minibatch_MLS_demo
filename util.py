import numpy as np
import scipy.io as sio

def dataloader(dataset, sliceno=None):
    print('Loading data ... ')
    if dataset == 'invivo_head':
        mask = b1['mask']
        b1 = b1['b1']  # [x y nslice ncoil] dims
        b1[np.isnan(b1)] = 0
        b1[np.isinf(b1)] = 0

        slice_ind = sliceno
        mask_slice = mask[:, :, slice_ind]
        test_slice = np.transpose(b1[:, :, slice_ind, :], (2, 0, 1))
        datasize = 80
        nslices = np.shape(b1)[2]
    if dataset == 'invivo_slice':
        b1 = sio.loadmat('data/demo_invivo_b1.mat')
        mask = b1['mask']
        b1 = b1['b1']
        b1[np.isnan(b1)] = 0
        b1[np.isinf(b1)] = 0

        b1 = b1[:, :, sliceno, :]
        mask = mask[:, :, sliceno]
        mask = np.expand_dims(mask,2)
        b1 = np.expand_dims(b1,2)
        datasize = 80
        nslices = np.shape(b1)[2]


    return b1, mask, nslices, datasize






