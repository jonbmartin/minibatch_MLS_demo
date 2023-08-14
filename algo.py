
# Dependencies
from random import randint
from random import random
from math import exp
from math import log

import numpy as np
import sigpy.plot as pl
import sigpy as sp


# Main
class InterleavedShim:
    '''Simple interleaved shimming for MLS optimization
    '''

    def __init__(self, x0, A, y, batchsize, beta, beta_fd = 0,
                 step_max=1000, full_GS_per_iter=2, cost_func='rmse', plotting=False, verbose=False,
                 solver='pinv', eps=None, n_cleanup_iter=10, record_loss_all_iter=False):

        # initialize optimization parameters
        self.step_max = step_max
        self.full_GS_per_iter = full_GS_per_iter
        self.n_cleanup_iter = n_cleanup_iter
        self.verbose = verbose
        self.eps = eps  # convergence threshold

        # initialize shim problem
        self.x0 = x0
        self.A = A                  # full ptx system matrix
        self.Nc = np.shape(A)[1]
        self.N = np.shape(A)[0]
        self.datasize = int(np.sqrt(self.N))
        self.y = y                  # target shim

        self.inds_nonzero = np.nonzero(y)[0]
        self.n_nonzero_inds = np.size(self.inds_nonzero)
        self.batchsize = batchsize  # batchsize, in # of spatial locations
        self.current_batchsize = batchsize # to track when the effective batchsize is full batch vs minibatch
        self.beta = beta * self.n_nonzero_inds            # power regularization
        self.phs = np.angle(self.A @ self.x0)
        self.m = self.A @ self.x0
        self.pinv_full = np.linalg.pinv(self.A.T.conj() @ self.A + self.beta * np.eye(self.Nc))

        self.n_cost_functions = 2
        if cost_func == 'rmse':
            self.cost_func = 'rmse'
            self.cost_function_index = 0
        elif cost_func == 'log':
            self.cost_func = 'log'
            self.cost_function_index = 1
        else:
            raise Exception("Unrecognized cost function specified")

        # recording best for both cost functions
        self.record_loss_all_iter = record_loss_all_iter
        self.current_state = self.x0
        self.state_vector = []
        self.current_cost = self.get_cost(self.x0, self.cost_func)
        self.best_state = self.current_state
        self.best_cost = self.current_cost
        self.solution_progress = []

        self.plotting = plotting

    def optimize(self):
        # reinitialize phase - just to be safe
        self.phs = np.angle(self.A @ self.x0)

        # begin optimizing
        self.step, self.accept = 1, 0
        self.cost_vector = []
        while self.step < self.step_max:

            # get neighbor
            exploration_neighbor, proposed_neighbor, E_n = self.interleaved_iter(n_full_iter=self.full_GS_per_iter)

            # compare cost
            if E_n < self.best_cost:
                self.best_cost = E_n
                if self.verbose:
                    print(f'iter {self.step} current best cost = {self.best_cost}')
                self.best_state = proposed_neighbor

            # check for convergence:
            if self.step > 1 and self.eps is not None:
                if abs(self.cost_vector[-2][1]-E_n[1]) < self.eps:
                    break

            # update some stuff
            self.step += 1

        if self.verbose:
            print(f'Best cost before refinement: {self.best_cost}')
        self.current_state = self.best_state
        self.m = self.A @ self.current_state
        self.phs = np.angle(self.m)

        # Finally, spend a few iterations at best minimum, to work down cost to local minimum
        for ii in range(self.n_cleanup_iter):

            proposed_neighbor = self.full_A_GS_step()
            self.best_state = proposed_neighbor
            E_n = self.get_cost(proposed_neighbor, self.cost_func)

            if self.verbose:
                print(f'refinement iter cost: {E_n}')
            if E_n < self.best_cost:
                self.best_cost = E_n
                self.best_state = proposed_neighbor[:]

        if self.plotting:
            pl.LinePlot(np.array(self.cost_vector).T, title='Energy of each iteraiton')
            pl.ImagePlot(np.reshape(self.A@self.best_state, (self.datasize, self.datasize)),
                         title=f'best sol, E_n = {self.best_cost}')

    def interleaved_iter(self, n_full_iter):
        # perform a minibatched gerchberg saxton update from the current location
        # assumes that self.phs has been assigned already

        # create minibatched A
        inds_pulled = np.random.choice(self.inds_nonzero, self.batchsize, replace=False)
        self.current_batchsize = self.batchsize
        self.A_samp = self.A[inds_pulled, :]

        # GS update with minibatch
        xHatGS_explore = np.linalg.pinv(self.A_samp.T.conj() @ self.A_samp + (self.current_batchsize / self.n_nonzero_inds)* self.beta * np.eye(self.Nc)) @ (
                self.A_samp.T.conj() @
                ((self.y[inds_pulled] * np.exp(
                    1j * self.phs[inds_pulled]))))

        self.state_vector.append(xHatGS_explore)
        if self.record_loss_all_iter:
            E_n = self.get_cost(xHatGS_explore, self.cost_func)
            self.cost_vector.append(E_n)

        # update current phase
        self.m = self.A @ xHatGS_explore
        self.phs = np.angle(self.m)

        xHatGS = xHatGS_explore 

        # perform full minibatch updates - use precomputed pseudoinverse
        self.current_batchsize = self.n_nonzero_inds
        for ii in range(n_full_iter):
            xHatGS = self.full_A_GS_step()
            self.state_vector.append(xHatGS)
            if self.record_loss_all_iter:
                E_n = self.get_cost(xHatGS, self.cost_func)
                self.cost_vector.append(E_n)

        # return feasible point and associated error.
        if not self.record_loss_all_iter:
            E_n = self.get_cost(xHatGS, self.cost_func)
            self.cost_vector.append(E_n)
        neighbor = xHatGS
        return xHatGS_explore, neighbor, E_n

    def full_A_GS_step(self):
        # take a full GS step using the entire pTx system matrix
        xHatGS = self.pinv_full @ (
                self.A.T.conj() @
                ((self.y * np.exp(
                    1j * self.phs))))

        # update current phase
        self.m = self.A @ xHatGS
        self.phs = np.angle(self.m)
        return xHatGS

    def RMSE_cost(self, x):
        self.m = self.A @ x

        err_rmse = np.linalg.norm((abs(self.m) * self.y - self.y)) ** 2
        err_rmse = np.sqrt(err_rmse / self.n_nonzero_inds)
        err_sar = np.abs(np.squeeze(np.real(x.T @ x)))

        err_total = err_rmse +  self.beta * err_sar

        return err_total


    def get_cost(self, x, cost_func):

        cost_rmse = self.RMSE_cost(x)

        return cost_rmse


