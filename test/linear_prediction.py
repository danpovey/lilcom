#!/usr/bin/env python3
"""
This script tests fast estimation of linear prediction coefficients,
using real data.
"""

# For parsing passed arguemtns, Built-in
import argparse
# Numpy library for array manipualtion, Required for Lilcom
import numpy as np
# Main package for audio compression and decompression, Principal
import lilcom
# For mathematic calculus, Built-in
import math
# For listing directories and os related tasks, Built-in
import os
# For loading wav audio to numpy array, Dependency
import scipy.io.wavfile
# For downsampling, Dependancy
import librosa
# For MP3 conversion, Dependancy
import pydub


defaultDatasetDir = "./OpenSLR81/samples"
defaultDataset = "OpenSLR81"
defaultDownloadLink = "http://www.openslr.org/resources/81/samples.tar.gz"



class LpcStats:
    def __init__(self, lpc_order, eta, dtype=np.float64):
        """
        Initialize the LpcStats object.  Note: it is usable to obtain stats for
        estimating LPC of lower order than the one specified.

        This is useful For predicting the next sample (e.g. x(t) given
        x(t-1),x(t-1)...,x(t-lpc_order)); `lpc_order` is the number of taps in
        the filter.

        However, if the scenario is that you want to predict y(t) given x
        values, then we predict y(t) given [ x(t), x(t-1), ... x(t-lpc_order) ],
        so the order of the filter would be lpc_order+1.  (In that case you will
        be providing the optional `y_block` argument to calls to
        accept_block()).

       Args:
         lpc_order: The LPC order, must be > 0.
         eta:       A per-sample forgetting factor, e.g. 0.99.
         dtype:     The NumPy data type to be used

      You will likely do:
         - Initialize the object x, then repeatedly:
            - call x.accept_block() for a new block of data
            - possibly call x.get_A()

         """
        self.lpc_order = lpc_order
        self.dtype=dtype
        self.autocorr = np.zeros(lpc_order + 1, dtype=dtype)
        self.A_minus = { }  # Will be dict from lpc-order to matrix.  See writeup
                            # for details.
        self.b_minus = { }  # Will be dict from lpc-order to vector b_minus is
                            # not defined in the writeup (it's only relevant in
                            # the case where cross-correlations are involved,
                            # i.e. you give the optional ``y_block`` argument to
                            # accept_block()), but it's analogous to A_minus;
                            # it's something relating to start effects that we
                            # need to subtract.
        self.eta = eta
        self.T = 0

        # Will contain the "x_hat" vector which is x (the input) itself;
        # the shape of x_hat will always be (lpc_order + size of most recent block),
        # where lpc_order is the required context.  Starts with zeros for t<0.
        self.x = np.zeros(lpc_order, dtype=dtype)

        # Will contain the "x_hat" vector which is x times a scale;
        # the shape of x_hat will always be (lpc_order + size of most recent block),
        # where lpc_order is the required context.  Starts with zeros for t<0.
        self.x_hat = np.zeros(lpc_order, dtype=dtype)


        # will contain the first `lpc_order` samples.
        # Needed to get A^- (see writeup).
        self.first_few_x_samples = np.zeros((0), dtype=dtype)
        # first_few_y_samples is only needed if we are predicting
        # something different from x_t, i.e. if the y_block
        # argument is given to accept_block().
        self.first_few_y_samples = np.zeros((0), dtype=dtype)

        # self.cross_correlation will only be used if the user
        # calls accept_block with the optional `y_block` argument;
        # this is for when you are predicting y_t from
        # x_{t-lpc_order}, ..., x_{t-1}, x_t.
        self.cross_correlation = np.zeros(lpc_order + 1, dtype=dtype)

        self.scale_vec = None       # will equal self.eta ** (2*T - np.arange(T))) for some T
        self.sqrt_scale_vec = None  # will equal self.eta ** (2*T - np.arange(T))) for some T

    def accept_block(self, x_block, y_block = None):
        assert len(x_block.shape) == 1 and (y_block is None or len(y_block) == len(x_block))
        # Update x_hat
        T_diff = x_block.shape[0]  # amount by which new T value (new num-frames) differs from old one.
        T_diff_sqrt_scale = self.eta ** T_diff
        self.x_hat = np.concatenate((self.x_hat[-self.lpc_order:] * T_diff_sqrt_scale,
                                     self._get_sqrt_scale(T_diff) * x_block))
        self.x = np.concatenate((self.x[-self.lpc_order:], x_block))

        self._update_autocorr_stats(y_block)
        self._update_first_few_samples(x_block, y_block)
        self.T += T_diff


    def get_A(self, lpc_order = None):
        """
        Returns the statistics.  This is a matrix A of shape (N+1, N+1) where
        N is self.lpc_order.  It can be expressed in Python-esque notation as:

           A = sum([ (self.eta**(2*(T-t))) * np.outer(get_hist(t), get_hist(t))
                     for t in range(self.lpc_order, T) ])

        where T is the total number of samples given to `accept_block()`

         Args:
               lpc_order: int     The lpc order, must satisfy
                               1 <= lpc_order <= self.lpc_order if set.  Defaults to
                               self.lpc_order.
         Returns:
               b: numpy.ndarray   The linear term in the objective function when
                            predicting a signal y given x; equals
                            sum_t w(t) [x(t-lpc_order),...,x(t)] y(t)

        """
        if lpc_order is None:
            lpc_order = self.lpc_order
        return self._get_A_all(lpc_order) - self._get_A_plus(lpc_order) - self._get_A_minus(lpc_order)

    def get_b(self, lpc_order = None):
        """
        Returns the weighted cross-correlation between x and y for delays 0, 1, ... ,lpc_order.

        Caution, the size of the returned value is (lpc_order + 1), not lpc_order..

         Args:
               lpc_order: int     The lpc order, must satisfy
                               1 <= lpc_order <= self.lpc_order if set.  Defaults to
                               self.lpc_order.
         Returns:
               b: numpy.ndarray   The linear term in the objective function when
                            predicting a signal y given x; equals
                            sum_t w(t) [x(t-lpc_order),...,x(t)] y(t)

        THIS IS ONLY RELEVANT IF you called `accept_block` with the optional
        y_block argument set; otherwise it is zero.
        """
        if lpc_order is None:
            lpc_order = self.lpc_order
        b_all = self.cross_correlation.copy()
        return b_all - self._get_b_minus(lpc_order)


    def get_autocorr_reflected(self, lpc_order = None):
        """
        Returns a version of the autocorrelation coefficients in which we
        imagine the exponentially-windowed signal is reflected in time T-1/2,
        we compute the autocorrelation coefficients of that signal,
        and then divide it by 2 to make it similar to A (A is the matrix
        that this class returns in get_A()).

        The point of the reflection is that it's a cheaper way to get more
        "reasonable" autocorrelation coefficients that we can use for
        preconditioning the fast update of the coefficients... without this
        reflection we are effectively using a windowing function that has a
        sharp discontinuity around t=T, which will tend to wash out the spectral
        information in the coefficients.

        This is explained more in the writeup.

         Args:
               lpc_order: int     The lpc order, must satisfy
                               1 <= lpc_order <= self.lpc_order if set.  Defaults to
                               self.lpc_order.
         Returns:
               b: numpy.ndarray   The autocorrelation statistics (not normalized
                            by count); if you construct Toeplitz matrix with
                            elements M[i,j] = b[abs(i-j)], it will be similar to
                            A (i.e. the result of calling get_A()).
        """
        if lpc_order is None:
            lpc_order = self.lpc_order

        ans = self.autocorr[0:lpc_order+1].copy()
        # Add in some terms which involve both halves of the reflected
        # signal.
        for k in range(1, lpc_order + 1):
            ans[k] += 0.5 * np.dot(self.x_hat[-k:], np.flip(self.x_hat[-k:]))
        return ans


    def _get_b_minus(self, lpc_order):
        """
        Returns a term that we need to subtract from the weighted cross-correlation between x and y
        to correct for start-of-sequence effects (the issue is: we need to exclude
        the samples numbered 0, 1, .. lpc_order - 1 because they have incomplete histories.)
        """

        if not lpc_order in self.b_minus:
            N = lpc_order
            b_minus = np.zeros(N + 1)
            x_samples = self.first_few_x_samples[:N]
            y_samples = self.first_few_y_samples[:N]
            # Below, the scaling factor with self.eta in it should really have
            # T - np.arange(lpc_order) instead of -np.arange(lpc_order), but
            # in order to make it possible to cache A_minus and have it be valid
            # for later, we omit the factor involving T for now and just use
            # N; we'll add the (T-N) part later.
            y_samples_weighted = y_samples * self._get_scale(N)
            for j in range(N):
                b_minus[:j+1] += y_samples_weighted[j] * np.flip(x_samples[:j+1])
            self.b_minus[lpc_order] = b_minus
        return self.b_minus[lpc_order] * (self.eta ** ((self.T - N) * 2))

    def _update_autocorr_stats(self, y_block = None):
        """
        Update the autocorrelation stats (self.autocorr)
        """
        N = self.lpc_order
        reverse_autocorr_stats = np.zeros(N + 1, dtype=self.dtype)
        x_hat = self.x_hat
        S = x_hat.shape[0]
        T_diff = S - N  # T_diff is number of new samples (i.e. excluding history)

        # Now `x_hat` corresponds to the weighted data
        # which we called \hat{x} in the writeup.
        for t in range(N, S):
            # weight that gets smaller as we go far back in time;
            # would be 1.0 at one sample past the end.
            reverse_autocorr_stats += x_hat[t-N:t+1] * x_hat[t]

        old_weight = self.eta ** (T_diff * 2)
        self.autocorr = self.autocorr * old_weight + np.flip(reverse_autocorr_stats)

        if y_block is not None:
            T = y_block.shape[0]
            y_hat = y_block * self._get_sqrt_scale(T)
            self.cross_correlation *= old_weight
            for k in range(N + 1):
                start = N - k
                self.cross_correlation[k] += np.dot(y_hat, x_hat[start:start+T]) * (self.eta ** (start-N))


    def _update_first_few_samples(self, x_block, y_block = None):
        if self.first_few_x_samples.shape[0] < self.lpc_order:
            full_x_block = np.concatenate((self.first_few_x_samples,
                                           x_block.astype(self.dtype)))
            current_num_samples = full_x_block.shape[0]
            self.first_few_x_samples = full_x_block[:min(current_num_samples, self.lpc_order)]

            if y_block is None:
                return
            full_y_block = np.concatenate((self.first_few_y_samples,
                                           y_block.astype(self.dtype)))
            self.first_few_y_samples = full_y_block[:min(current_num_samples, self.lpc_order)]



    def _get_A_all(self, lpc_order):
        """
        (This is explained in the writeup)  Gets the part of the statistics matrix
        A that comes from the autocorrelation stats.

         Params:
             lpc_order  The lpc order for which we are getting stats;
                  must satisfy 0 < lpc_order <= self.lpc_order
        """
        assert lpc_order > 0 and lpc_order <= self.lpc_order
        N1 = lpc_order + 1
        A_all = np.zeros((N1, N1), dtype=self.dtype)
        for j in range(N1):
            for k in range(N1):
                A_all[j,k] = (self.eta ** -(j+k)) * self.autocorr[abs(j-k)]
        return A_all

    def _get_A_plus(self, lpc_order):
        """
        (This is explained in the writeup).  Gets the part of the statistics matrix
        A that relates to end effets (at the "recent" end), to be subtracted
        from A_all.

         Params:
             lpc_order  The lpc order for which we are getting stats;
                  must satisfy 0 < lpc_order <= self.lpc_order

        """
        N = lpc_order
        N1 = lpc_order + 1
        A_plus = np.zeros((N1, N1), dtype=self.dtype)

        # Note: this 'T' is not really a time value, it's just the length of the
        # x_hat vector which is lpc_order plus the most recent block size; but it's
        # what we need to calculate the weighting factors, as it's the distance
        # from the end of x_hat vector that matters.

        # x_hat takes the place of the sequence x_hat in the writeup.  In fact
        # it just contains the tail (last elements) of x_hat, but for clarity we
        # call its length T.  (This code would still work if we were using the
        # entire sequence instead of self.history).
        x_hat = self.x_hat

        x = self.x

        T = self.x_hat.shape[0]


        for j in range(1, N1):
            #   A_plus[j,k] = ((self.eta ** -2) * A_plus[j-1,k-1] +
            #                       x_hat[T-j] * x_hat[T-k])
            # We vectorize this as:
            A_plus[j,1:] = (self.eta ** -2 * (A_plus[j-1,:-1]) +  x[T-j] * np.flip(x[T-N:T]))

        return A_plus

    def _get_A_minus(self, lpc_order):
        """
        (This is explained in the writeup).  Gets the part of the statistics matrix
        A that relates to effects at the beginning of the sequence.

         Params:
             lpc_order  The lpc order for which we are getting stats;
                  must satisfy 0 < lpc_order <= self.lpc_order, and
                  lpc_order < self.T (since otherwise there would
                  be no stats available yet for this order).
        """
        assert(lpc_order > 0 and lpc_order <= self.lpc_order and
               lpc_order < self.T)
        if not lpc_order in self.A_minus:
            samples = self.first_few_x_samples[:lpc_order]
            N = lpc_order
            N1 = lpc_order + 1
            A_minus = np.zeros((N1, N1), dtype=self.dtype)
            for j in range(N-1, -1, -1):  # for j in [N-1, N-2, .. 0]
                # This formula has undergone a range of simplifications and efficiency
                # improvements versus what was in the paper, but still gives the same result.
                A_minus[j,:N] = ((self.eta ** 2) * (A_minus[j+1,1:] +
                                                    samples[N-1-j] * np.flip(samples)))

            self.A_minus[lpc_order] = A_minus

        return self.A_minus[lpc_order] * (self.eta ** ((self.T - lpc_order) * 2))

    def _get_scale(self, T):
        """
        Returns self.eta ** (2*(T - np.arange(T))), except it caches this and doesn't do
        unnecessary computation.
        """
        if self.scale_vec is None or T > self.scale_vec.shape[0]:
            self.scale_vec = self.eta ** (2*(T - np.arange(T)))
        return self.scale_vec[-T:]

    def _get_sqrt_scale(self, T):
        """
        Returns self.eta ** (T - np.arange(T))), except it caches this and doesn't do
        unnecessary computation.
        """
        if self.sqrt_scale_vec is None or T > self.sqrt_scale_vec.shape[0]:
            self.sqrt_scale_vec = self.eta ** (T - np.arange(T))
        return self.sqrt_scale_vec[-T:]


class OnlineLinearSolver:
    """
    This class is for solving systems of the form A x = b with A symmetric
    positive definite, where A can be reasonably closely approximated by a
    Toeplitz matrix that the user can supply, and A and b are changing with time
    (so it makes sense to start the optimization from the previous value).

    This class uses preconditioned conjugate gradient descent (CGD)
    to approximately solve for x.

    The main intended use is to estimate linear prediction coefficients
    obtained from class LpcStats.
    """
    def __init__(self, N, num_cgd_iters = 3,
                 num_cgd_iters_initial = 5,
                 diag_smoothing = 1.0e-07,
                 toeplitz_smoothing = 1.0e-02,
                 abs_smoothing = 1.0e-20,
                 dtype=np.float64,
                 debug=False):
        """
        Initialize the object.
        Args:
             N: The dimension of the problem (e.g. LPC order)
     cgd_iters: The number of iterations of conjugate gradient
                descent that we do each time we solve.
   cgd_iters_initial:  The number of CGD iters we do the first time
               (should be more, since we are not starting from a
               reasonable estimate.)
   diag_smoothing:  Determines how much we scale up the zeroth
               autocorrelation coefficient to ensure we
               can limit the condition number of M (the
               Toeplitz matrix formed from the autocorr stats)
   toeplitz_smoothing:  Constant that controls how much we
               smooth A with M (this multiplied by diag_smoothing
               limits the condition number of the smoothed A).
   abs_smoothing:    A value that we add to the diagonal
               of M to make sure that it is positive definite (relevant if the
               data is exactly zero); should not matter that much.
   debug:  If true, verbose output will be printed
        """
        self.N = N
        self.num_cgd_iters = num_cgd_iters
        self.num_cgd_iters_initial = num_cgd_iters_initial
        self.diag_smoothing = diag_smoothing
        self.toeplitz_smoothing = toeplitz_smoothing
        self.abs_smoothing = abs_smoothing
        self.dtype = dtype
        self.cur_estimate = np.zeros(N, dtype=self.dtype)
        self.debug = debug


    def get_current_estimate(self):
        return self.cur_estimate

    def estimate(self, A, b, autocorr_stats):
        """
        Re-estimates the linear prediction coefficients and returns it
        as a vector.

        Args:
           autocorr_stats:  Autocorrelation coefficients, of dimension self.N,
                for use when smoothing A.
           A:   The quadratic term in the objective function x^T A x - 2 b x
           b:   The linear term in the objective function x^T A x - 2 b x

        Return:
          Returns the updated `x` value (e.g. the filter parameters)
        """
        N = self.N
        if self.cur_estimate is None:
            self.cur_estimate = np.zeros(N, dtype=self.dtype)
            num_iters = self.num_cgd_iters_initial
        else:
            num_iters = self.num_cgd_iters
        assert autocorr_stats.shape == (A.shape[0],) and b.shape == (A.shape[0],)
        x = self.cur_estimate
        A = A.copy().astype(self.dtype)
        b = b.copy().astype(self.dtype)
        autocorr_stats = autocorr_stats.copy()
        autocorr_stats[0] += self.abs_smoothing + (self.diag_smoothing * autocorr_stats[0])
        M = get_toeplitz_mat(autocorr_stats)
        t = self.toeplitz_smoothing
        A_orig = A
        A = (1.0 - t) * A + t * M
        # Adjust b for the change of A_orig -> A, keeping the derivative w.r.t. x of
        # the objective function (x^T A x - 2 b x) unchanged.
        b += np.dot(A, x) - np.dot(A_orig, x)

        r = b - np.dot(A, x)  # residual
        z = toeplitz_solve(autocorr_stats, r)  # preconditioned residual

        assert np.dot(z, r) >= 0.0

        p = z.copy()
        rsold = np.dot(r,z)
        rs_orig = rsold
        if self.debug:
            print("Residual0 is {}, objf0 is {}".format(rsold,
                                                        np.dot(np.dot(A,x),x) - 2.0 * np.dot(x,b)))

        for iter in range(num_iters):
            Ap = np.dot(A, p)
            alpha = rsold / np.dot(p, Ap)
            x += alpha * p
            if iter == num_iters-1:
                break
            r -= alpha * Ap;
            z = toeplitz_solve(autocorr_stats, r)
            rsnew = np.dot(r, z)
            assert(rsnew >= 0.0)
            if self.debug:
                print("ResidualN is {}, ratio={}, objf={} ".format(rsnew, rsnew / rs_orig,
                                                                   (np.dot(np.dot(A,x),x) - 2.0 * np.dot(x,b))))
            if rsnew / rs_orig < 1.0e-05:
                break
            p = z + (p * (rsnew / rsold))
            rsold = rsnew
        # We'll use this as the starting point for optimization the next time this is called.
        self.cur_estimate = x.copy()
        return x

def test_new_stats_accum_and_solver():
    """
    Tests that our formulas for the fast stats accumulation match the 'obvious'
    method of stats accumulation, and that the LpcStats object behaves as
    expected.
    """
    T = 100
    signal = np.random.rand(T)
    time_constant = 32
    #eta = (time_constant - 1.0) / time_constant
    eta = 0.999
    dtype=np.float64
    N = 10  # Order of filter
    N1 = N+1
    # A_ref is the stats A accumulated in the simple/obvious way.
    A_ref = np.zeros((N1,N1), dtype=dtype)
    autocorr = np.zeros(N1, dtype=dtype)
    for t in range(N, T):
        w_t = eta ** ((T - t) * 2)
        A_ref += w_t * np.outer(np.flip(signal[t-N:t+1]),
                                np.flip(signal[t-N:t+1]))

    A_plus_ref = np.zeros((N1, N1), dtype=dtype)
    A_minus_ref = np.zeros((N1, N1), dtype=dtype)
    for t in range(T, T+N):
        w_t = eta ** ((T - t) * 2)
        for j in range(N1):
            if t - j >= T:
                continue
            for k in range(N1):
                if t - k >= T:
                    continue
                A_plus_ref[j,k] += w_t * signal[t-j] * signal[t-k]

    for t in range(N):
        w_t = eta ** ((T - t) * 2)
        for j in range(t + 1):
            for k in range(t + 1):
                A_minus_ref[j,k] += w_t * signal[t-j] * signal[t-k]


    hat_x = signal * np.power(eta, T - np.arange(T))

    # Get autocorr stats.
    for t in range(T):
        for j in range(min(N+1, t+1)):
            autocorr[j] += hat_x[t] * hat_x[t - j]

    A_all = np.zeros((N1, N1), dtype=dtype)
    for j in range(N1):
        for k in range(N1):
            A_all[j,k] = (eta ** -(j+k)) * autocorr[abs(j-k)]
    A_plus = np.zeros((N1, N1), dtype=dtype)
    A_minus = np.zeros((N1, N1), dtype=dtype)

    for j in range(1, N1):
        for k in range(j, N1):
            A_plus[j,k] = ((eta ** -2) * A_plus[j-1,k-1]) + ((eta**-(j+k)) * hat_x[T-j] * hat_x[T-k])
    for j in range(N-1, -1, -1):  # for j in [N-1, N-2, .. 0]
        for k in range(j, N):  # Note: this excludes k == N, since those elements are zero.
            A_minus[j,k] = ((eta ** 2) * A_minus[j+1,k+1]) + ((eta**-(j+k)) * hat_x[N-1-j] * hat_x[N-1-k])

    for j in range(N1):
        for k in range(j):
            # Copy upper to lower triangle of A_plus and A_minus
            A_plus[j,k] = A_plus[k,j]
            A_minus[j,k] = A_minus[k,j]


    A = A_all - A_plus - A_minus
    error = A - A_ref
    rel_error = np.abs(error).sum() / np.abs(A_ref).sum()
    print("Relative error in new stats accumulation is {}".format(rel_error))
    assert rel_error < 1.0e-05

    error = A_plus - A_plus_ref
    rel_error = np.abs(error).sum() / np.abs(A_plus).sum()
    print("Relative error in A+ accumulation is {}".format(rel_error))
    assert rel_error < 1.0e-05

    error = A_minus - A_minus_ref
    rel_error = np.abs(error).sum() / np.abs(A_minus).sum()
    print("Relative error in A- accumulation is {}".format(rel_error))
    print("A- = {}, A-_ref = {}", A_minus, A_minus_ref)
    assert rel_error < 1.0e-05

    print("A^all is: ", A_all)
    print("A+ is: ", A_plus)
    print("A- is: ", A_minus)

    if True:
        # This block tests LpcStats with 1 block, and with the
        # same lpc_order.
        stats = LpcStats(lpc_order=N, eta=eta, dtype=dtype)
        stats.accept_block(signal)
        A_from_stats = stats.get_A(N)
        print("New A^all is: ", stats._get_A_all(N))
        print("New A+ is: ", stats._get_A_plus(N))
        print("New A- is: ", stats._get_A_minus(N))

        error = A - A_from_stats
        rel_error = np.abs(error).sum() / np.abs(A).sum()
        print("Relative error in LpcStats-based stats accumulation is {}".format(rel_error))
        assert rel_error < 1.0e-05

        if True:
            # This block tests the linear solver.
            solver = OnlineLinearSolver(N)
            A_for_solver = A[1:,1:]
            b_for_solver = A[0,:-1]
            autocorr_for_solver = autocorr[:-1]
            x = solver.estimate(A_for_solver, b_for_solver, autocorr_for_solver)
            residual = np.dot(A_for_solver, x) - b_for_solver
            rel_error = (np.dot(residual, residual) / np.dot(b_for_solver,
                                                             b_for_solver))
            print("Relative error in LPC solver = {}".format(rel_error))

    if True:
        # This block tests LpcStats with 1 block, and with
        # lpc_order given to the object higher than N.
        stats = LpcStats(lpc_order=N+2, eta=eta, dtype=dtype)
        stats.accept_block(signal)
        A_from_stats = stats.get_A(N)
        print("New A^all is: ", stats._get_A_all(N))
        print("New A+ is: ", stats._get_A_plus(N))
        print("New A- is: ", stats._get_A_minus(N))

        error = A - A_from_stats
        rel_error = np.abs(error).sum() / np.abs(A).sum()
        print("Relative error in LpcStats-based stats accumulation (diff. LPC-order) is {}".format(rel_error))
        assert rel_error < 1.0e-05

    if True:
        # This block tests LpcStats with 2 blocks, and with
        # lpc_order given to the object higher than N.
        stats = LpcStats(lpc_order=N+2, eta=eta, dtype=dtype)

        len2 = signal.shape[0] // 2
        stats.accept_block(signal[:len2])
        stats.accept_block(signal[len2:])

        A_from_stats = stats.get_A(N)
        print("New A^all is: ", stats._get_A_all(N))
        print("New A+ is: ", stats._get_A_plus(N))
        print("New A- is: ", stats._get_A_minus(N))

        error = A - A_from_stats
        rel_error = np.abs(error).sum() / np.abs(A).sum()
        print("Relative error in LpcStats-based stats accumulation (diff. LPC-order, 2 blocks) is {}".format(rel_error))
        assert rel_error < 1.0e-05


    if True:
        # This block tests LpcStats with 2 blocks, and with
        # lpc_order given to the object higher than N.
        stats = LpcStats(lpc_order=N+2, eta=eta, dtype=dtype)

        stats.accept_block(signal[:1])
        stats.accept_block(signal[1:])

        A_from_stats = stats.get_A(N)
        print("New A^all is: ", stats._get_A_all(N))
        print("New A+ is: ", stats._get_A_plus(N))
        print("New A- is: ", stats._get_A_minus(N))

        error = A - A_from_stats
        rel_error = np.abs(error).sum() / np.abs(A).sum()
        print("Relative error in LpcStats-based stats accumulation (diff. LPC-order, 2 blocks, one small) is {}".format(rel_error))
        assert rel_error < 1.0e-05

    if True:
        # This block tests the reflected autocorrelation stats
        hat_x_reflected = np.concatenate((hat_x, np.flip(hat_x)))
        autocorr_reflected_ref = np.zeros(N + 1, dtype=dtype)
        for t in range(2*T):
            for j in range(min(N+1, t+1)):
                autocorr_reflected_ref[j] += hat_x_reflected[t] * hat_x_reflected[t - j]

        for higher_order in range(N, N+2):
            for tiny_blocks in [False, True]:
                stats = LpcStats(lpc_order=higher_order, eta=eta, dtype=dtype)
                if tiny_blocks:
                    for t in range(0, T, 32):
                        end_t = min(T, t + 32)
                        stats.accept_block(signal[t:end_t])
                else:
                    stats.accept_block(signal)
                autocorr_reflected = stats.get_autocorr_reflected(N)
                # it actually returns half the autocorr of the reflected signal.
                error = (0.5 * autocorr_reflected_ref) - autocorr_reflected
                rel_error = np.abs(error).sum() / np.abs(autocorr_reflected).sum()
                print("Relative error in accumulating reflected autocorr stats (order={},higher-order={},tiny-blocks={} is {}".format(
                        N, higher_order, tiny_blocks, rel_error))
                assert rel_error < 1.0e-05
                toeplitz_solve(autocorr_reflected, autocorr_reflected)


def test_new_stats_accum_cross():
    """
    Tests that the LpcStats object behaves as expected for 'cross-correlations',
    i.e. when we are predicting y given (x_{t-lpc_order}, ... , x_{t-1}, x_t).
    """
    T = 100
    x = np.random.rand(T)
    y = np.random.rand(T)
    time_constant = 32
    eta = 0.999
    dtype=np.float64
    N = 2  # Order of filter
    N1 = N+1
    A_ref = np.zeros((N1,N1), dtype=dtype)
    b_ref = np.zeros(N1, dtype=dtype)
    for t in range(N, T):
        w_t = eta ** ((T - t) * 2)
        hist = np.flip(x[t-N:t+1])
        A_ref += w_t * np.outer(hist, hist)
        b_ref += w_t * y[t] * hist

    if True:
        # This block tests LpcStats for cross-correlations with 1 block, and
        # with the same lpc_order.
        stats = LpcStats(lpc_order=N, eta=eta, dtype=dtype)
        stats.accept_block(x, y)
        A_from_stats = stats.get_A(N)
        print("New A^all is: ", stats._get_A_all(N))
        print("New A+ is: ", stats._get_A_plus(N))
        print("New A- is: ", stats._get_A_minus(N))

        error = A_ref - A_from_stats
        rel_error = np.abs(error).sum() / np.abs(A_ref).sum()
        print("Relative error in LpcStats-based stats accumulation of A is {}".format(rel_error))
        assert rel_error < 1.0e-05

        b_from_stats = stats.get_b(N)
        error = b_ref - b_from_stats
        rel_error = np.abs(error).sum() / np.abs(b_ref).sum()
        print("Relative error in LpcStats-based stats accumulation of b is {}, b={}, b_ref={}".format(rel_error, b_from_stats, b_ref))
        assert rel_error < 1.0e-05

def get_toeplitz_mat(autocorr):
    """
    Returns a Toeplitz matrix constructed from the provided autocorrelation
    coefficients

    Args:
       autocorr must be a NumPy array with one axis (a vector)
    Return:
       Returns a square matrix M with ``autocorr.shape[0]`` rows and
       columns, elements M[i,j] = autocorr[abs(i-j)]
    """
    N = autocorr.shape[0]
    autocorr_flip = np.flip(autocorr)
    M = np.zeros((N, N))
    for k in range(N):
        M[k,:k] = autocorr_flip[-k-1:-1]
        M[k,k:] = autocorr[:N-k]
    return M

def toeplitz_solve(autocorr, y):
    """
    Let y be a vector of dimension N and let
    `autocorr` be vector of dimension N representing, conceptually,
    a Toeplitz matrix A(i,j) = autocorr[abs(i-j)].

      This function solves the linear system A x = y, returning x.

    We require for the Toeplitz matrix to satisfy the usual conditions for
    algorithms on Toeplitz matrices, meaning no singular leading minor may be
    singular (i.e. not det(A[0:n,0:n])==0 for any n).  This will naturally be
    satisfied if A is the autocorrelation of a finite nonzero sequence (I
    believe).

    This function solves for x using the Levinson-Trench-Zohar
    algorithm/recursion..  I had to look this up... in this case
    y is an arbitrary vector, so the Levinson-Durbin recursion as normally
    used in signal processing doesn't apply.

    I am looking at:
       https://core.ac.uk/download/pdf/4382193.pdf:
      "Levinson and fast Choleski algorithms for Toeplitz and almost
      Toeplitz matrices", RLE technical report no. 538 by Bruce R Muscius,
      Research Laboratory of Electronics, MIT,

    particularly equations 2.4, 2.6, 2.7, 2.8, (unnumbered formula below 2.9),
    2.10.  There is opportunity for simplification because this Toeplitz matrix
    is symmetric.
    """

    # The technical report I am looking at deals with things of dimension (N+1),
    # so for consistency I am letting N be the dimension minus one.
    N = autocorr.shape[0] - 1
    assert y.shape[0] == N + 1
    x = np.zeros(N+1)
    b = np.zeros(N+1)
    b_temp = np.zeros(N+1)
    r = autocorr

    b[0] = 1.0
    epsilon = r[0]
    x[0] = y[0] / epsilon

    for n in range(1, N+1):

        # eqn 2.6 for \nu_n.  We are not evaluating \xi_n because it is
        # the same.  Be careful with the indexing of b.  Notice in Eq.
        # (2.3) that the elements of b are in a very strange order,
        # so you have to interpret (2.6) very carefully.
        nu_n = (-1.0 / epsilon) * sum([r[j+1] * b[j] for j in range(n)])

        # next few lines are Eq. 2.7
        b_temp[0] = 0.0
        b_temp[1:n+1] = b[:n]
        b_temp[:n] += nu_n * np.flip(b[:n])
        b[:n+1] = b_temp[:n+1]

        # Eq. 2.8
        epsilon *= (1.0 - nu_n * nu_n)
        if not abs(nu_n) < 1.0:
            M = get_toeplitz_mat(autocorr)
            w, v = np.linalg.eig(M)
            print("Eigs of Toeplitz matrix are {}".format(w))
            raise RuntimeError("Something went wrong, nu_n = {}".format(nu_n))

        # The following is an unnumbered formula below Eq. 2.9
        lambda_n = y[n] - sum([ r[n-j] * x[j] for j in range(n)])

        x[:n+1] += (lambda_n / epsilon) * b[:n+1];

    return x

def test_toeplitz_solve():
    for dim in [ 1, 2, 5, 9 ]:
        signal_len = 100
        signal = np.random.rand(signal_len).astype(np.float64)
        # We want the Toeplitz matrix to be invertib
        autocorr = np.zeros(dim, dtype=np.float64)
        for offset in range(dim):
            autocorr[offset] = np.dot(signal[:signal_len-offset], signal[offset:])

        y = np.random.rand(dim).astype(np.float64)
        b = toeplitz_solve(autocorr, y)
        A = np.zeros((dim, dim), dtype=np.float64)
        for i in range(dim):
            for j in range(dim):
                A[i,j] = autocorr[abs(i-j)]
        err = np.dot(A, b) - y
        relative_error = np.abs(err).sum() / np.abs(y).sum()
        print("Toeplitz solver: relative error is {}".format(relative_error))


def test_get_toeplitz_mat():
    autocorr = np.array([ 1.0, 2.0, 3.0, 4.0])
    N = autocorr.shape[0]
    M = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            M[i,j] = autocorr[abs(i-j)]
    assert np.array_equal(M, get_toeplitz_mat(autocorr))


def test_toeplitz_solve_compare():
    autocorr = np.array([ 10.0, 5.0, 2.0, 1.0 ])
    y = np.array([ 1.0, 2.0, 3.0, 4.0 ])
    b = toeplitz_solve(autocorr, y)
    print("b is {}".format(b))


def compute_residual_sumsq(x, t_begin, t_end, lp_coeffs):
    """
    Computes the sum-squared residual of the data in `x` versus the
    prediction by the linear prediction coefficients.

    Args:
         x:  np.ndarray of a floating type, the array that is being predicted.
         t_begin,t_end: int   The first and one-past-the-last 't' index
                              (index into x) on which to compute the
                              residuals
         lp_coeffs:           linear prediction coefficients: the
                              coefficients on x(t-1), x(t-2), etc.

    Returns:
         (orig_sumsq, pred_sumsq): (float, float)

         orig_sumsq is the sum-square of the elements x[t_begin:t_end]
         pred_sumsq is the sum-square of, for t in range(t_begin, t_end),
         the difference between x[t] and the predicted value given
         the linear prediction coefficients `lp_coeffs`.
    """
    orig_sumsq = (x[t_begin:t_end] ** 2).sum()

    order = lp_coeffs.shape[0]
    lp_coeffs_rev = np.flip(np.concatenate((np.asarray([-1.0], dtype=lp_coeffs.dtype),
                                           lp_coeffs))).astype(x.dtype)
    pred_sumsq = 0.0
    if t_begin > order:
        for t in range(t_begin, t_end):
            residual = np.dot(x[t-order:t+1], lp_coeffs_rev)
            pred_sumsq += residual * residual

    return (orig_sumsq, pred_sumsq)


def test_prediction(array):
    # Operate on a linear signal.
    assert(len(array.shape) == 1)
    order = 25
    dtype = np.float64
    array = array.astype(dtype)
    orderp1 = order + 1
    T = array.shape[0]
    block_size = 32
    eta = 1.0 - (1.0/128.0)
    stats = LpcStats(lpc_order=order, eta=eta, dtype=dtype)
    solver = OnlineLinearSolver(N=order, dtype=dtype)  # otherwise defaults

    pred_sumsq_tot = 0.0
    raw_sumsq_tot = 0.0

    for t in range(0, T, block_size):
        t_end = min(t + block_size, T)

        (raw_sumsq, pred_sumsq) = compute_residual_sumsq(
            array, t, t_end, solver.get_current_estimate())

        pred_sumsq_tot += pred_sumsq
        raw_sumsq_tot += raw_sumsq

        stats.accept_block(array[t:t_end].copy())

        if t >= order:
            A = stats.get_A()
            autocorr = stats.get_autocorr_reflected()
            A_for_solver = A[1:,1:]
            b_for_solver = A[0,1:]
            autocorr_for_solver = autocorr[:-1]
            solver.estimate(A_for_solver, b_for_solver, autocorr_for_solver)

    print("Ratio of residual-sumsq / raw-sumsq is %f" % (pred_sumsq_tot / raw_sumsq_tot))



def waveRead(filename, sampleRate=0):
    """ This function reads a wavefile at a desired sample rate and returns
            it in form of a numpy array

       Args:
        filename: The name of the file. File should be in wav format
            otherwise it will encounter error.
        sampleRate: an integer denoting the number of samples in a unit time.
            In case this is set to None, the function will choose the sample-
            -rate determined by arguemtns passed in while calling the script,
            otherwise the samplerate of the original wav file.

       Returns:
           a Numpy array of the given audio file.
    """
    audioArray = None
    if sampleRate == 0:
        sampleRate, audioArray = scipy.io.wavfile.read(filename)
        settings["sample-rate"] = sampleRate
        return audioArray
    if sampleRate != 0:
        sr, audioArray = scipy.io.wavfile.read(filename)
        if (sampleRate != sr):
            if audioArray.dtype == np.int16:
                audioArray = audioArray.astype(np.float32) / 32768
                downsampledArray = librosa.core.resample(
                                                audioArray.transpose(),
                                                sr, sampleRate).transpose()

            return downsampledArray
        return audioArray
    return None


# Parsing input arguments
parser = argparse.ArgumentParser(description="Lilcom reconstruction test \
            module")
parser.add_argument("--dataset", "-d",
                    help="The directory of test dataset")
parser.add_argument("--samplerate", "-s",
                    help="Number of samplings in a unit time for each audio")
parser.add_argument("--releaselog", "-l",
                    help="The name of the log file")
parser.add_argument("--releasedf", "-c",
                    help="The name of the csv file including results")
args = parser.parse_args()

# Global values for settings
settings = dict.fromkeys(["dataset-dir", "sample-rate", "release-log",
                         "release-df"])

# Assigning system values based on passed arguments
if args.dataset:
    settings["dataset-dir"] = args.dataset
else:
    if defaultDataset not in os.listdir():
        print("Downloading the dataset")
        os.system("mkdir ./" + defaultDataset)
        os.system("wget " + defaultDownloadLink + " -P ./" + defaultDataset)
        os.system("tar -xf ./" + defaultDataset + "/samples.tar.gz -C "
                  + defaultDataset)
        settings["dataset-dir"] = defaultDatasetDir
    else:
        settings["dataset-dir"] = defaultDatasetDir
if settings["dataset-dir"][-1] == "/":
    settings["dataset-dir"] = settings["dataset-dir"][:-1]
    # Removes the / if existing at the end of directory

if args.samplerate:
    settings["sample-rate"] = int(args.samplerate)
else:
    settings["sample-rate"] = 0


fileList = [settings["dataset-dir"] + "/" + item
            for item in os.listdir(settings["dataset-dir"])
            if ".wav" in item]

def test_new_stats_accum_and_solver_compare():
    """
    This function is to help debug the "C" version of the code in
    ../lilcom/prediction_math.

    Tests that our formulas for the fast stats accumulation match the 'obvious'
    method of stats accumulation, and that the LpcStats object behaves as
    expected.
    """
    T = 10
    signal = np.asarray([1,2,3,4,5,7,9,11,13,15], dtype=np.float64)


    eta = 0.5  # Just for test, will change later!
    dtype=np.float64
    N = 4  # Order of filter
    N1 = N+1

    stats = LpcStats(N, eta, dtype=dtype)

    # two blocks.
    stats.accept_block(signal[:5])

    aux_order = 4
    print("autocorr is: ", stats.autocorr)
    print("x_hat is: ", stats.x_hat)
    print("autocorr-reflected is: ", stats.get_autocorr_reflected(aux_order))

    stats.accept_block(signal[5:])

    print("autocorr is: ", stats.autocorr)
    print("x_hat is: ", stats.x_hat)
    print("autocorr-reflected is: ", stats.get_autocorr_reflected(aux_order))

    stats._get_A_minus(aux_order)
    print("A'^- is: ", stats.A_minus[aux_order])
    print("A^+ is: ", stats._get_A_plus(aux_order))
    print("A^all is: ", stats._get_A_all(aux_order))

    print("A is: ", stats.get_A(aux_order))


test_new_stats_accum_and_solver()
test_new_stats_accum_cross()
test_toeplitz_solve()
test_toeplitz_solve_compare()
test_get_toeplitz_mat()
test_new_stats_accum_and_solver_compare()

for file in fileList:
    audioArray = waveRead(file, settings["sample-rate"])
    test_prediction(audioArray[:, 0])

