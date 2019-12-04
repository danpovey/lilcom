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

        For predicting the next sample (e.g. x(t) given x(t-1),..., )
        `lpc_order` is the number of taps in the filter, so we predict
        x(t) given [ x(t-1), ... , x(t-lpc_order) ].

        However, if the scenario is that you want to predict y(t) given x
        values, then we predict y(t) given [ x(t), x(t-1), ... x(t-lpc_order) ],
        so the order of the filter would be lpc_order+1.
        (In that case you will provide the optional `y_block` argument to
        calls to accept_block()).

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
        self.A_minus = { }  # Will be dict from lpc-order to matrix
        self.b_minus = { }  # Will be dict from lpc-order to vector b_minus is
                            # not defined in the writeup but it's analogous to
                            # A_minus; it's something relating to start effects
                            # that we need to subtract.
        self.eta = eta
        self.T = 0
        # Will contain the most recent history of length
        # up to `lpc_order` (less only if the block was shorter
        # than that).
        self.history = np.zeros(lpc_order, dtype=dtype)

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

    def accept_block(self, x_block, y_block = None):
        assert len(x_block.shape) == 1 and (y_block is None or len(y_block) == len(x_block))
        T_diff = x_block.shape[0]  # amount by which new T value (new num-frames) differs from old one.

        self._update_autocorr_stats(x_block, y_block)
        self._update_history(x_block)
        self._update_first_few_samples(x_block, y_block)
        self.T += T_diff


    def get_A(self, lpc_order):
        """
        Returns the statistics.  This is a matrix A of shape (N+1, N+1) where
        N is self.lpc_order.  It can be expressed in Python-esque notation as:

           A = sum([ (self.eta**(2*(T-t))) * np.outer(get_hist(t), get_hist(t))
                     for t in range(self.lpc_order, T) ])

        where T is the total number of samples given to the `accept_block`
        call.
        """
        return self._get_A_all(lpc_order) - self._get_A_plus(lpc_order) - self._get_A_minus(lpc_order)

    def get_b(self, lpc_order):
        """
        Returns the weighted cross-correlation between x and y for delays 0, 1, ... ,lpc_order;
        caution, the size of the returned value is (lpc_order + 1).
        This is only relevant if you called `accept_block` with the optional
        y_block argument set.
        """
        b_all = self.cross_correlation.copy()
        return b_all - self._get_b_minus(lpc_order)


    def get_autocorr_reflected(self, lpc_order):
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
        """
        T = self.history.shape[0]  # This is a kind of 'fake T', using the
                                   # highly truncated history in `self.history`.
                                   # It behaves in the equations like T, so we
                                   # call it that.
        # x_hat is like the 'tail' (last few elements of) of x_hat which
        # is the exponentially weighted signal.
        x_hat = self.history * self.eta ** (T - np.arange(T))

        ans = self.autocorr[0:lpc_order+1]
        # Add in some terms which involve both halves of the reflected
        # signal.
        for k in range(1,lpc_order + 1):
            ans[k] += 0.5 * np.dot(x_hat[-k:], np.flip(x_hat[-k:]))
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
            # for later, we omit the factor involving T for now.
            y_samples_weighted = y_samples * (self.eta **  (-2 * np.arange(N)))
            for j in range(N):
                b_minus[:j+1] += y_samples_weighted[j] * np.flip(x_samples[:j+1])
            self.b_minus[lpc_order] = b_minus
        return self.b_minus[lpc_order] * (self.eta ** (self.T * 2))

    def _update_autocorr_stats(self, x_block, y_block = None):
        """
        Update the autocorrelation stats (self.autocorr)
        """
        assert len(x_block.shape) == 1
        full_x_block = np.concatenate((self.history, x_block.astype(self.dtype)))
        N = self.lpc_order
        reverse_autocorr_stats = np.zeros(N + 1, dtype=self.dtype)
        S = full_x_block.shape[0]
        # Don't do *=, we don't want to modify `x_block` in case
        # np.concatenate
        x_hat = full_x_block * (self.eta ** (S - np.arange(S)))
        # Now `x_hat` corresponds to the weighted data
        # which we called \hat{x} in the writeup.
        for t in range(N, S):
            # weight that gets smaller as we go far back in time;
            # would be 1.0 at one sample past the end.
            reverse_autocorr_stats += x_hat[t-N:t+1] * x_hat[t]

        old_weight = self.eta ** (x_block.shape[0] * 2)
        self.autocorr = self.autocorr * old_weight + np.flip(reverse_autocorr_stats)

        if y_block is not None:
            T = y_block.shape[0]
            # the weighting factor below has the factor of 2; it's the
            # weight of each sample in the objective function.
            y_hat = y_block * (self.eta ** (2 * (T - np.arange(T))))
            self.cross_correlation *= old_weight
            for k in range(self.lpc_order + 1):
                # full_x_block has `self.lpc_order` extra samples added at the
                # beginning, it's otherwise the same as x_block.  So we are
                # multiplying each weighted y_t by x_t delayed by k samples, and
                # the sum gets added to self.cross_correlation[k]
                start = self.lpc_order - k
                self.cross_correlation[k] += np.dot(y_hat, full_x_block[start:start+T])

    def _update_history(self, x_block):
        """ Keeps self.history up to date (self.history is the last
          `self.lpc_order` samples).
        """
        x_block_len = x_block.shape[0]
        if x_block_len < self.lpc_order:
            x_block = np.concatenate((self.history, x_block))
        self.history = x_block[-self.lpc_order:].copy()


    def _update_first_few_samples(self, x_block, y_block = None):
        if self.first_few_x_samples.shape[0] < self.lpc_order:
            full_x_block = np.concatenate((self.first_few_x_samples,
                                         x_block.astype(self.dtype)))
            num_new_samples = x_block.shape[0]
            self.first_few_x_samples = full_x_block[:min(num_new_samples, self.lpc_order)]

            if y_block is None:
                return
            full_y_block = np.concatenate((self.first_few_y_samples,
                                         y_block.astype(self.dtype)))
            num_new_samples = y_block.shape[0]
            self.first_few_y_samples = full_y_block[:min(num_new_samples, self.lpc_order)]



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
        N1 = lpc_order + 1
        A_plus = np.zeros((N1, N1), dtype=self.dtype)

        # Note: this 'T' is not really a time value, it's just self.lpc_order, but
        # it's what we need to calculate the weighting factors, as it's the distance
        # from the end of the array self.history that matters.
        T = self.history.shape[0]
        # x_hat takes the places of the sequence x_hat in the writeup.  In fact
        # it just contains the last `self.lpc_order` samples of x_hat, but for
        # clarity we call its length T.  (This code would still work if we were
        # using the entire sequence instead of self.history).
        x_hat = self.history * self.eta ** (T - np.arange(T))


        for j in range(1, N1):
            for k in range(j, N1):
                A_plus[j,k] = ((self.eta ** -2) * A_plus[j-1,k-1] +
                               (self.eta**-(j+k)) * x_hat[T-j] * x_hat[T-k])
        # Copy upper to lower triangle of A_plus
        for j in range(N1):
            for k in range(j):
                A_plus[j,k] = A_plus[k,j]
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
            # Below, the scaling factor with self.eta in it should really have
            # T - np.arange(lpc_order) instead of -np.arange(lpc_order), but
            # in order to make it possible to cache A_minus and have it be valid
            # for later, we omit the factor involving T for now.
            x_hat = samples * (self.eta **  -np.arange(lpc_order))
            N = lpc_order
            N1 = lpc_order + 1
            A_minus = np.zeros((N1, N1), dtype=self.dtype)
            for j in range(N-1, -1, -1):  # for j in [N-1, N-2, .. 0]
                for k in range(j, N):  # Note: this excludes k == N, since those elements are zero.
                    A_minus[j,k] = (((self.eta ** 2) * A_minus[j+1,k+1]) +
                                    ((self.eta**-(j+k)) * x_hat[N-1-j] * x_hat[N-1-k]))
            # Copy upper to lower triangle of A_plus
            for j in range(N1):
                for k in range(j):
                    A_minus[j,k] = A_minus[k,j]

            self.A_minus[lpc_order] = A_minus

        return self.A_minus[lpc_order] * (self.eta ** (self.T * 2))


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
    def __init__(self, N, num_cgd_iters = 2,
                 num_cgd_iters_initial = 5,
                 diag_smoothing = 1.0e-07,
                 toeplitz_smoothing = 1.0e-02,
                 abs_smoothing = 1.0e-20,
                 dtype=np.float64):
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
        """
        self.N = N
        self.num_cgd_iters = num_cgd_iters
        self.num_cgd_iters_initial = num_cgd_iters_initial
        self.diag_smoothing = diag_smoothing
        self.toeplitz_smoothing = toeplitz_smoothing
        self.abs_smoothing = abs_smoothing
        self.dtype = dtype
        self.cur_estimate = None


    def estimate(self, autocorr_stats, A, b):
        """
        Re-estimates the linear prediction coefficients and returns it
        as a vector.

        Args:
           autocorr_stats:  Autocorrelation statistics, of dimension self.N,
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

        print("{}, {}, {}", autocorr_stats.shape[0], A.shape[0], b.shape[0])
        assert autocorr_stats.shape[0] == A.shape[0]

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
        # TODO: cleanup
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
    method of satts accumulation, and that the LpcStats object behaves as
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
            x = solver.estimate(autocorr_for_solver, A_for_solver,
                                b_for_solver)
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
                    for t in range(T):
                        stats.accept_block(signal[t:t+1])
                else:
                    stats.accept_block(signal)
                autocorr_reflected = stats.get_autocorr_reflected(N)
                # it actually returns half the autocorr of the reflected signal.
                error = (0.5 * autocorr_reflected_ref) - autocorr_reflected
                rel_error = np.abs(error).sum() / np.abs(autocorr_reflected).sum()
                print("Relative error in accumulating reflected autocorr stats (order={},higher-order={},tiny-blocks={} is {}".format(
                        N, higher_order, tiny_blocks, rel_error))
                assert rel_error < 1.0e-05


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
        assert abs(nu_n) < 1.0

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

#For n=1, epsilon=10.0, nu_n=-0.5
#.y[n] = 2, .. lambda_n = 1.5
#For n=2, epsilon=7.5, nu_n=0.06666666666666667
#... lambda_n = 2.0
#For n=3, epsilon=7.466666666666667, nu_n=-0.035714285714285705
#... lambda_n = 2.5285714285714285
# b is [0.00574713 0.0862069  0.0862069  0.33908046]



def conj_optim(cur_coeffs, quad_mat, autocorr_stats,
               num_iters=2, order=None, dtype=np.float64,
               proportional_smoothing=1.0e-10):
    # Note: this modifies cur_coeffs in place.
    #  Uses conjugate gradient method to minimize the function
    #  cur_coeffs^T quad_mat cur_coeffs, subject to the constraint
    #  that cur_coeffs[0] == -1.    quad_mat is symmetric.
    #  If we define A = quad_mat[1:,1:] and b = quad_mat[0,1:],
    #  then we can say we are minimizing
    #   objf = x^T A x  - 2 x b
    #  dobjf/dx = 2Ax - 2b = 0, so we are solving Ax = b.
    #
    #  This function actually smooths M with something derived from the
    # autorrelation stats, dictated by `proportional_smoothing`.

    if order is None:
        order = quad_mat.shape[0] - 1
    if order != quad_mat.shape[0] - 1:
        conj_optim(cur_coeffs[0:order+1], quad_mat[0:order+1, 0:order+1],
                   autocorr_stats[0:order+1], num_iters, dtype=dtype)
        return



    abs_smoothing = 1.0e-10

    b = quad_mat[0,1:].copy().astype(dtype)
    A = quad_mat[1:,1:].copy().astype(dtype)


    w, v = np.linalg.eig(A)
    if not w.min() > 0.0:
        w2, v = np.linalg.eig(quad_mat)
        print("WARN:eigs are not positive: A={}, eigs={}, quad-mat-eigs={}".format(A, w, w2))

    ## we are solving Ax = b.  Trivial solution is: x = A^{-1} b
    x = cur_coeffs[1:].copy()

    if True:
        exact_x = np.dot(np.linalg.inv(A), b)
        print("Exact objf is {}".format(np.dot(np.dot(A,exact_x),exact_x) - 2.0 * np.dot(exact_x,b)))
        #cur_coeffs[1:]  = exact_x
        #return


    r = b - np.dot(A, x)
    z = toeplitz_solve(autocorr_stats[:-1], r)
    assert np.dot(z, r) >= 0.0

    p = z.copy()
    rsold = np.dot(r,z)
    rs_orig = rsold
    print("Residual0 is {}, objf0 is {}".format(rsold,
                                                np.dot(np.dot(A,x),x) - 2.0 * np.dot(x,b)))

    for iter in range(num_iters):
        Ap = np.dot(A, p)
        alpha = rsold / np.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap;
        z = toeplitz_solve(autocorr_stats[:-1], r)
        rsnew = np.dot(r, z)
        assert(rsnew >= 0.0)
        print("ResidualN is {}, ratio={}, objf={} ".format(rsnew, rsnew / rs_orig,
              (np.dot(np.dot(A,x),x) - 2.0 * np.dot(x,b))))
        if rsnew / rs_orig < 1.0e-05:
            break
        p = z + (p * (rsnew / rsold))
        rsold = rsnew
    cur_coeffs[1:] = x



reflection = True


def add_prev_block_terms(array, t_start, order, quad_mat,
                         optimize = False):
    """ Add in some terms to quad_mat that come from products of x(t) within the
    previous block where the y(t) has t >= t_start.  Only modifies upper
    triangle.
    """
    orderp1 = order + 1
    if not optimize:
        # Here is the un-optimized code.  Just modify upper triangle for now.
        for k in range(order):
            t = t_start + k
            for i in range(k + 1, orderp1):
                for j in range(i, orderp1):
                    quad_mat[i,j] += array[t-i] * array[t-j]

    else:
        # This is more optimized; it's O(order^2) vs. O(order^3).  The path from
        # the one above to here is a little complicated but you can verify that
        # they give the same results.
        # only the upper triangle is set.
        for j in range(order):
            local_sum = 0.0
            for i in range(1, orderp1 - j):
                local_sum += array[t_start-i] * array[t_start-i-j]
                quad_mat[i,i+j] += local_sum

def subtract_block_end_terms(array, t_end, order, quad_mat,
                             optimize = False):
    """ Subtracts some quad_mat some products of terms near the
    end of a block that will have been included in the autocorrelation
    stats we computed but which we don't want because they arise from
    the prediction of y(t) for t >= t_end.
    """
    orderp1 = order + 1

    if not optimize:
        # The slower but easier-to-understand version.
        for k in range(order):
            t = t_end + k
            for i in range(k + 1, orderp1):
                for j in range(k + 1, orderp1):
                    quad_mat[i,j] -= array[t-i] * array[t-j]
    else:
        # The optimized version
        for j in range(order):
            local_sum = 0.0
            for i in range(1, orderp1 - j):
                local_sum += array[t_end-i] * array[t_end-i-j]
                quad_mat[i,i+j] -= local_sum


def add_reflection(array, t_end, order, autocorr_stats, sign = 1):
    """ Add in some temporary stats to `autocorr_stats` due to a notional
    reflection of the signal at time t_end+1/2.
    This reflection avoids the large discontinuity at t_end, which otherwise
    would degrade the autocorrelation prediction quality. (NB: we're
    actually only using this for preconditioning, but the same applies).
    """
    scale = 0.5 * sign
    for i in range(order):
        for j in range(i + 1, order):
            autocorr_stats[j] += scale * array[t_end - (j-i)] * array[t_end - 1 - i]




def init_stats(array, order, block_size, optimize = False, dtype=np.float64):
    """
    Initializes the stats (quad_mat, autocorr_stats)

    and returns them as a tuple.  The first block is assumed to start at frame zero,
    but for the quad_mat part of the stats, the first 'order' samples are not
    included as the 'x' but only as history.
    """
    orderp1 = order + 1
    quad_mat = np.zeros((orderp1, orderp1), dtype=dtype)
    autocorr_stats = np.zeros(orderp1, dtype=dtype)
    assert block_size > order

    # Get autocorrelation stats for the rest of the block except for
    # t < order.  These include products with t < order.
    autocorr = np.zeros(orderp1, dtype=dtype)
    for t in range(order, block_size):
        for j in range(orderp1):
            autocorr_stats[j] += array[t] * array[t-j]

    # commit to upper triangle of quad_mat
    for i in range(orderp1):
        for j in range(i, orderp1):
            quad_mat[i,j] += autocorr_stats[abs(i-j)]


    add_prev_block_terms(array, order, order, quad_mat, optimize)

    subtract_block_end_terms(array, block_size, order, quad_mat, optimize)


    if True:  # test code
        # Copy upper to lower triangle of quad_mat
        for i in range(orderp1):
            for j in range(i):
                quad_mat[i,j] = quad_mat[j,i]

        w, v = np.linalg.eig(quad_mat)
        print("After subtracting block-end terms, smallest eig of quad_mat is: {}".format(
                w.min()))


    # Copy upper to lower triangle of quad_mat
    for i in range(orderp1):
        for j in range(i):
            quad_mat[i,j] = quad_mat[j,i]

    # Include in `autocorr_stats` some terms from near the beginning that we
    # omitted so that they would not be included in quad_mat.  (The autocorrelation
    # needs to include everything.)
    for i in range(order):
        for j in range(i + 1):
            autocorr_stats[j] += array[i] * array[i-j]

    if reflection:
        add_reflection(array, block_size, order, autocorr_stats)

    return (quad_mat, autocorr_stats)



def update_stats(array, t_start,
                 t_end, quad_mat,
                 autocorr_stats,
                 prev_scale, dtype=np.float32,
                 proportional_smoothing=1.0e-10):
    """
    Update autocorrelation and quadratic stats.. scale down previous stats by 0 < prev_scale <= 1.
    The aim is for quad_mat to contain the following, where N == order:

       quad_mat(i,j) = \sum_{t=N}^{t_end-1} weight(t) \sum_{i=0}^N \sum_{j=0}^N array[t-i] array[t-j]  (1)

    ... where weight(t) is 1.0 for the current segment starting at t_start, and exponentially
    decays in the past according to prev_scale... so we want to scale down any previously
    accumulated stats by weight(t).

    Note: in the equations above, t starts from N, i.e. we ignore the first N samples of
    the signal, because for those samples we don't have the full history.  This is equivalent
    to setting the weight to be zero for those samples.  We accomplish this by letting the first
    block of the signal start at N.  Note: we assume that all blocks are of length greater than
    N.

    Within each block we primarily update quad_mat using autocorrelation stats,
    taking advantage of the almosty-Toeplitz sructure but we need to also
    account for edge effects.
    """
    orderp1 = quad_mat.shape[0]
    order = orderp1 - 1
    assert t_start >= order and (t_end - t_start) > order

    quad_mat *= prev_scale

    autocorr_within_block = np.zeros(orderp1, dtype=dtype)
    autocorr_cross_block = np.zeros(orderp1, dtype=dtype)

    # Get the autocorrelation stats for which the later frame in the product is
    # within the current block.  This includes some cross-block terms, which
    # we need to treat separately.
    for i in range(order):
        for j in range(i + 1):
            autocorr_within_block[j] += array[t_start + i] * array[t_start + i - j]
        for j in range(i + 1, orderp1):
            autocorr_cross_block[j] += array[t_start + i] * array[t_start + i - j]

    for t in range(t_start + order, t_end):
        for i in range(orderp1):
            autocorr_within_block[i] += array[t] * array[t-i]


    # subtract the `temporary stats` added in the last
    # block.
    if reflection and t_start > order:
        prev_t_end = t_start
        add_reflection(array, prev_t_end, order, autocorr_stats, -1)

    # Update autocorr_stats, our more-permanent version of the autocorr
    # stats.  We need to make sure that these could be the autocorrelation
    # of an actual signal, which involves some special changes.
    autocorr_stats *= prev_scale

    fast = True
    assert t_start > order  # first block would be handled in init_stats
    if fast:
        # We view the signal itself as decaying with sqrt(prev_scale); the
        # autocorrelation stats decay with the square of that since they
        # are products of signals.
        autocorr_stats += autocorr_cross_block * math.sqrt(prev_scale)
        autocorr_stats += autocorr_within_block
    else:
        # Special case at first block.
        sqrt_scale = math.sqrt(prev_scale)
        for t in range(t_start, t_end):
            for i in range(orderp1):
                t_prev = t - i
                if t_prev < 0:
                    continue
                elif t_prev >= t_start:
                    autocorr_stats[i] += array[t] * array[t_prev]
                else:
                    autocorr_stats[i] += array[t] * array[t_prev] * sqrt_scale


    # Add in some temporary stats due to a notional reflection of the signal at time t_end+1/2
    if reflection:
        add_reflection(array, t_end, order, autocorr_stats)

    # Add in the autocorrelation stats to quad_mat
    autocorr_tot = autocorr_within_block + autocorr_cross_block
    for i in range(orderp1):
        for j in range(orderp1):
            quad_mat[i,j] += autocorr_tot[abs(i-j)]


    # Add in some terms that are really from the autocorrelation of the
    # previous block, and which we had previously subtracted / canceled
    # out when processing it.  (If this is the first block, we'll have
    # t_start == order and those will be fresh terms that we do want.)

    optimize = False
    add_prev_block_terms(array, t_start, order, quad_mat,
                         optimize)

    if True:  # test code
        # Copy upper to lower triangle of quad_mat
        for i in range(orderp1):
            for j in range(i):
                quad_mat[i,j] = quad_mat[j,i]

        w, v = np.linalg.eig(quad_mat)
        if w.min() < 0 and w.min() < proportional_smoothing * w.sum():
            print("tstart,end={},{}; WARN: After adding before-the-beginning terms, smallest eig of quad_mat is: {}, ratio={}".format(
                    t_start, t_end, w.min(), w.min() / w.sum()));

    subtract_block_end_terms(array, t_end, order, quad_mat, optimize)

    # Copy upper to lower triangle of quad_mat
    for i in range(orderp1):
        for j in range(i):
            quad_mat[i,j] = quad_mat[j,i]


    if True: # test code
        w, v = np.linalg.eig(quad_mat)
        threshold = 1.0e-05
        proportional_smoothing = 1.0e-10
        if w.min() < 0 and w.min() < proportional_smoothing * w.sum():
            print("tstart,end={},{}; WARN: after subtracting after-the-end terms, smallest eig of quad_mat is: {}, ratio={}".format(
                    t_start, t_end, w.min(), w.min() / w.sum()));



def test_prediction(array):
    # Operate on a linear signal.
    assert(len(array.shape) == 1)
    order = 25
    dtype = np.float64
    array = array.astype(dtype)
    orderp1 = order + 1
    T = array.shape[0]
    autocorr = np.zeros(orderp1, dtype=dtype)

    cur_coeffs = np.zeros(orderp1, dtype=dtype)
    cur_coeffs[0] = -1


    weight = 0.75
    num_cg_iters = 3
    pred_sumsq_tot = 0.0
    raw_sumsq_tot = 0.0
    BLOCK = 32
    proportional_smoothing = 1.0e-01
    assert(BLOCK > order)

    for t in range(T):

        if (t % BLOCK == 0) and t > 0:
            optimize = True
            # This block updates quad_mat_stats.
            if t == BLOCK:
                (quad_mat_stats, autocorr_stats) = init_stats(array, order, BLOCK,
                                                              optimize, dtype=dtype)
            else:
                update_stats(array, max(order, t-BLOCK), t,
                             quad_mat_stats, autocorr_stats,
                             weight, dtype=dtype,
                             proportional_smoothing=proportional_smoothing)


            if True:
                quad_mat = quad_mat_stats.copy()
                orig_zero_element = quad_mat[0,0]

                conj_optim(cur_coeffs, quad_mat,
                           autocorr_stats, num_cg_iters,
                           order=(None if t > 5*BLOCK else min(t // 16, order)),
                           dtype=np.float32,
                           proportional_smoothing=proportional_smoothing)
                max_elem = np.max(np.abs(cur_coeffs[1:]))
                print("Current residual / unpredicted-residual is (after update): {}, max coeff is {}".format(
                        np.dot(cur_coeffs, np.dot(quad_mat, cur_coeffs)) / orig_zero_element, max_elem))

            raw_sumsq = 0.0
            pred_sumsq = 0.0
            if t+BLOCK > array.shape[0]:
                continue
            # The rest is diagnostics: see how prediction compares with raw sumsq.
            for t2 in range(t, t+BLOCK):
                raw_sumsq += array[t2] * array[t2]

                pred = 0.0
                for i in range(order+1):
                    pred += cur_coeffs[i] * array[t2 - i]
                pred_sumsq += pred * pred
            print("For this block, pred_sumsq / raw_sumsq = {}".format(pred_sumsq / raw_sumsq))
            if t > BLOCK:
                pred_sumsq_tot += pred_sumsq
                raw_sumsq_tot += raw_sumsq
                print("For blocks till now (t={}),, pred_sumsq_tot / raw_sumsq_tot = {}".format(
                        t, pred_sumsq_tot / raw_sumsq_tot))


        for i in range(orderp1):
            if t-i >= 0:
                autocorr[i] += array[t-i] * array[t]




def logger(logmod="initialization", reportList=None):
    """ This function prints out the log given the initialization mode or
            the result of test on a single file. As a result it prints out
            the result on screen and in case that additional reports are
            requested it prints them out too.

       Args:
        logmod: There are two possible values, first is ``initialization''
            which is for the first line of the report. Also the other mode is
            the report mode in which an additional result list will be needed
            to be printed out.
        reportList: A list of dictionaries returned by the code, which is the
            evaluation result on one file.
    """
    global settings

    text = ""
    headerLine = ""
    if logmod == "initialization":
        text = "Reconstruction Test called for lilcom... \n"
        text += "The report is on the dataset placed at: " + \
                settings["dataset-dir"] + "\n"
        text += "The sample-rate is set to "
        if settings["sample-rate"] == 0:
            text += "the default sample rate of dataset"
        else:
            text += str(settings["sample-rate"])

        global evaulators

        headerLine += "filename" + "\t"
        for evaluator in evaulators:
            headerLine += \
                evaluator["algorithm"] + str(evaluator["additionalParam"]) + \
                "-bitrate"
            headerLine += "\t"
            headerLine += \
                evaluator["algorithm"] + str(evaluator["additionalParam"]) + \
                "-psnr"
            headerLine += "\t"
            headerLine += \
                evaluator["algorithm"] + str(evaluator["additionalParam"]) + \
                "-hash"
            headerLine += "\t"

        text += "\n"
        text += headerLine

    if logmod == "result":
        if reportList is None:
            return
        """
            Elements are each a dictionary of "evaluator" and "result"
        """
        text += reportList[0] + "\t"
        for element in reportList[1:]:
            elementResult = element["result"]
            text += str(elementResult["bitrate"])
            text += "\t\t"
            text += '%.2f' % elementResult["psnr"]
            text += "\t\t"
            text += str(elementResult["hash"])
            text += "\t\t"

    print(text)

    # Checks for output logfile settings
    if settings["release-log"] is not None:
        settings["release-log"].write(text)
        settings["release-log"].write("\n")

    if settings["release-df"]:
        if logmod == "initialization":
            settings["release-df"].write(headerLine.replace("\t", ","))
            settings["release-df"].write("\n")
        else:
            settings["release-df"].write(text.replace("\t", ","))
            settings["release-df"].write("\n")
    return


def lilcomReconstruct(audioArray, lpcOrder):
    """ This function will reconstruct the given audio array in form of a
            conescutive compression and decompression procedure.

       Args:
        audioArray: A numpy array as the audio signal
        lcpOrder: Same as lcpOrder in the main lilcom functions

       Returns:
           an Audio array with same size to the array passed as input which
            is a result of compresion and decompresion
    """
    bitPerSample = 6  # Issue make it passed by the operator
    # bitsPerSample Should be recieved from settings
    audioArray = audioArray.astype(np.float32)
    outputShape = list(audioArray.shape)

    outputShape[0] += 4
    outputShape = tuple(outputShape)

    outputArray = np.ndarray(outputShape, np.int8)
    reconstructedArray = np.ndarray(audioArray.shape, np.int16)

    c = lilcom.compress(audioArray, lpc_order=lpcOrder,
                        bits_per_sample=bitPerSample, axis=0)
    reconstructedArray = lilcom.decompress(c, dtype=audioArray.dtype)
    return reconstructedArray


def MP3Reconstruct(filename, bitrate):
    # Creating a temporary path for MP3 and reconstruction File
    tmpPath = "./ReconstTemp"
    if tmpPath[2:] in os.listdir("./"):
        os.system("rm -dR " + tmpPath)
    os.system("mkdir " + tmpPath)
    wavFile = pydub.AudioSegment.from_wav(filename)
    wavFile.export(tmpPath + "/output.mp3", format="mp3", bitrate=bitrate)
    # print("At bitrate {}, file {} compresses to {} bytes".format(
    #    bitrate, filename, os.path.getsize(tmpPath + "/output.mp3")))
    mp3File = pydub.AudioSegment.from_mp3(tmpPath + "/output.mp3")
    mp3File.export(tmpPath + "/reconst.wav", format="wav")
    sampleRateReconst, audioReconst = \
        scipy.io.wavfile.read(tmpPath + "/reconst.wav")
    os.system("rm -dR " + tmpPath)
    return audioReconst


def evaluate(filename=None, audioArray=None, algorithm="lilcom",
             additionalParam=None):
    """ This function does an evaluation on the given audio array, with
            the requested algorithm and additional parameters. As a result
            it returnes a map including the bitrate, a hash and the psnr
            result with the given audio array.

       Args:
        filename: The name of the file used for compression. It is requiered
            for some compression algorithms like MP3 which needs some file
            manipulations
        audioArray: Numpy array including original file. It is required for
            PSNR evaulation. If None value was passed, the function will
            load it from the passed filename.
        algorithm: The desired algorithm which will show which algorithm is
            chosen to be evaulated. Default value is set to lilcom
        additionalParam: Parameters which each algorithm is supposed to have.
            i.e. For lilcom it contains lpc-order and for MP3 it will have
            chosen bitrate.
       Returns:
            A dictionary with three keys; bitrate, psnr and hash.
    """
    global settings
    returnValue = dict.fromkeys(["bitrate", "psnr", "hash"])
    returnValue["bitrate"] = 0
    returnValue["psnr"] = 0
    returnValue["hash"] = 0

    """
    In case of empty input audio array it loads the array. The audio array is
        required for evaluation subroutine call 'PSNR'
    """
    if audioArray is None:
        if settings["sample-rate"] != 0:
            audioArray = waveRead(filename, settings["sample-rate"])
        if settings["sample-rate"] == 0:  # DOOOOO STH
            audioArray = waveRead(filename, settings["sample-rate"])
    reconstructedArray = None
    # Evaluation Procedure for lilcom
    if algorithm == "lilcom":
        reconstructedArray = lilcomReconstruct(audioArray,
                                               lpcOrder=additionalParam)
        returnValue["psnr"] = PSNR(audioArray, reconstructedArray)
        returnValue["bitrate"] = 8 * settings["sample-rate"]
        returnValue["hash"] = hash(reconstructedArray)
    # Evaluation Procedure for MP3
    elif algorithm == "MP3":
        reconstructedArray = MP3Reconstruct(filename,
                                            bitrate=additionalParam)
        returnValue["psnr"] = PSNR(audioArray, reconstructedArray)
        returnValue["bitrate"] = int(additionalParam[:3])*1000
        returnValue["hash"] = hash(reconstructedArray)
    # Evaluation for additional compression library
    else:
        pass

    return returnValue


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


test_new_stats_accum_and_solver()
test_new_stats_accum_cross()
test_toeplitz_solve()
test_toeplitz_solve_compare()
test_get_toeplitz_mat()

for file in fileList:
    audioArray = waveRead(file, settings["sample-rate"])

    print("Shape is {}".format(audioArray.shape))
    test_prediction(audioArray[:, 0])

