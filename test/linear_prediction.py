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



class ToeplitzLpcEstimator:
    def __init__(self, lpc_order, eta,
                 diag_smoothing = 1.0e-07,
                 abs_smoothing = 1.0e-20,
                 dtype=np.float64):
        assert(eta ** (-2 * lpc_order) < 2.0)  # convergence requirement, at
                                               # least in an earlier version
                                               # before simplification.


        self.lpc_order = lpc_order
        self.eta = eta
        self.diag_smoothing = diag_smoothing
        self.abs_smoothing = abs_smoothing
        self.dtype = dtype
        self.deriv = None
        self.autocorr = np.zeros(lpc_order, dtype=dtype)
        self.lpc_coeffs = np.zeros(lpc_order, dtype=dtype)
        self.sqrt_scale_vec = None
        self.eta_power_vec = None
        self.scale_vec = None


        # Will contain the most recent lpc_order + size samples of the input x,
        # where `size` is the length of the most recent block of samples
        # provided to accept_block().  Starts with zeros for t<0.
        self.x = np.zeros(lpc_order, dtype=dtype)

        ## Will contain the "x_hat" vector which is x times a scale; the shape of
        ## x_hat after calling accept_block() will be (lpc_order + size of most
        ## recent block), where lpc_order is the required context.  Starts with
        ## lpc_order zeros for t<0.
        #self.x_hat = np.zeros(lpc_order, dtype=dtype)


    def accept_block(self, x_block, residual):
        assert len(x_block.shape) == 1 and x_block.shape == residual.shape

        # Update x_hat
        #T_diff = x_block.shape[0]  # amount by which new T value (new num-frames) differs from old one.
        #T_diff_sqrt_scale = self._get_sqrt_scale(T_diff)[0]  # self.eta ** T_diff
        #self.x_hat = np.concatenate((self.x_hat[-self.lpc_order:] * T_diff_sqrt_scale,
        #                             self._get_sqrt_scale(T_diff) * x_block))

        self.x = np.concatenate((self.x[-self.lpc_order:], x_block))

        self._set_derivative(residual)
        self._update_autocorr_stats()
        au = self._get_autocorr_reflected(self.lpc_order)
        # Smooth autocorr stats to avoid NaNs and the like.
        au[0] += self.abs_smoothing + (self.diag_smoothing * au[0])
        self.lpc_coeffs += toeplitz_solve(au, self.deriv)
        return self.lpc_coeffs

    def get_current_estimate(self):
        return self.lpc_coeffs


    def _get_autocorr_reflected(self, lpc_order = None):
        """
        Returns a version of the autocorrelation coefficients in which we
        imagine the exponentially-windowed signal is reflected in time T-1/2
        (where T is the number of samples seen so far); we compute the
        autocorrelation coefficients of that signal, and then divide it by 2 to
        make it similar to A (A is the matrix that this class returns in
        get_A()).

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
                            by count); if you construct a Toeplitz matrix with
                            elements M[i,j] = b[abs(i-j)], it will be similar to
                            A (i.e. the result of calling get_A()).
        """
        if lpc_order is None:
            lpc_order = self.lpc_order

        ans = self.autocorr.copy()
        # Add in some terms which involve both halves of the reflected
        # signal.
        # CAUTION: below we go to lpc_order - 1 because we are not including
        # the "lpc_order'th" coefficient.
        for k in range(1, lpc_order):
            #ans[k] += 0.5 * np.dot(self.x_hat[-k:], np.flip(self.x_hat[-k:]))
            ans[k] += 0.5 * np.dot(self.x[-k:], np.flip(self.x[-k:])) * self._get_eta_power(k+1)

        return ans

    def _set_derivative(self, residual):
        """
        [For use in simplified update] Sets self.deriv to the deriviative
        contribution from just this block, at the current values of the LPC
        coeffs.
        """
        N = self.lpc_order
        T_diff = residual.shape[0]
        S = T_diff + N  # length of x
        assert(self.x.shape[0] == S)

        scaled_residual = residual * self._get_scale(residual.shape[0])
        reverse_deriv = np.zeros(N, dtype=self.dtype)
        for t in range(T_diff):
            reverse_deriv += self.x[t:t+N] * scaled_residual[t]
        self.deriv = np.flip(reverse_deriv)

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

    def _get_eta_power(self, t):
        """
        Returns self.eta ** t, except it caches this and doesn't do unnecessary
        computation.
        """
        assert t >= 0
        if self.eta_power_vec is None or t >= self.eta_power_vec.shape[0]:
            self.eta_power_vec = self.eta ** np.arange(t + 1)
        return self.eta_power_vec[t]


    def _update_autocorr_stats(self):
        """
        Update the autocorrelation stats (self.autocorr)
        Only need them from 0 to lpc_order - 1.
        """
        N = self.lpc_order
        reverse_autocorr_stats = np.zeros(N, dtype=self.dtype)
        x = self.x
        S = x.shape[0]
        T_diff = S - N  # T_diff is number of new samples (i.e. excluding history)

        # Now `x_hat` corresponds to the weighted data
        # which we called \hat{x} in the writeup.
        for t in range(N, S):
            # weight that gets smaller as we go far back in time;
            # would be 1.0 at one sample past the end.
            reverse_autocorr_stats += x[t-N+1:t+1] * x[t] * self._get_eta_power(2*(S - t) - 1)

        old_weight = self.eta ** (T_diff * 2)  ## self._get_scale(T_diff)[0]  # == self.eta ** (T_diff * 2)
        self.autocorr = old_weight * self.autocorr + np.flip(reverse_autocorr_stats * self._get_sqrt_scale(N))



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

        #print("iter={}, nu_n={} epsilon={}, lambda={}, b[:n+1]={}, x[:n+1] = {}".format(
        #        n, nu_n, epsilon, lambda_n, b[:n+1], x[:n+1]))

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
    print("Comparing toeplitz solver")
    autocorr = np.array([ 10.0, 5.0, 2.0, 1.0 ])
    y = np.array([ 1.0, 2.0, 3.0, 4.0 ])
    b = toeplitz_solve(autocorr, y)
    print("b is {}".format(b))


def compute_residual(x, t_begin, t_end, lp_coeffs):
    """
    Computes the residual of the data in `x` for the
    block x[t_begin:t_end], with the provided LPC coefficients.

    Args:
         x:  np.ndarray of a floating type, the array that is being predicted.
         t_begin,t_end: int   The first and one-past-the-last 't' index
                              (index into x) on which to compute the
                              residuals
         lp_coeffs:           linear prediction coefficients: the
                              coefficients on x(t-1), x(t-2), etc.

    Returns:
         The residual for the block x[t_begin:t_end] as numpy array
         with same dtype as x.
    """

    residual = np.zeros(t_end - t_begin, dtype=x.dtype)


    order = lp_coeffs.shape[0]
    lp_coeffs_rev = np.flip(np.concatenate((np.asarray([1.0], dtype=lp_coeffs.dtype),
                                            -lp_coeffs))).astype(x.dtype)
    pred_sumsq = 0.0
    if t_begin > order:
        for t in range(t_begin, t_end):
            residual[t - t_begin] = np.dot(x[t-order:t+1], lp_coeffs_rev)

    return residual



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

    # The following inequality is required for convergence, i.e.
    # to prevent the possibility of the update of x diverging.
    # eta ** (-2 * order) is a constant that if we multiplied
    # M by it, we could prove that have M >= A, in the sense that for
    # any x, x^T M x >= x^T A x.  [the -2*order relates to the -(j+k)
    # in eq. (17) in the writeup].  The < 2.0 is because for
    # a quadratic objective function, we converge as long as the
    # learning rate is less than twice the "optimal" learning rate.

    # perl -e 'print (log(2.0)/2.0);'
    # 0.346573590279973mac:lilcom:
    # Suppose eta is of the form eta = (1.0 - delta), where delta = 1 / eta_inv.
    # we can use this to get a limit on eta_inv as compared to 'order'.
    # By taking logs, we have:
    # log(eta) * -2 * order < log(2.0)
    # approximating log(eta) as linear:
    # -delta * -2 * order < log(2.0)
    #  delta * order < log(2.0)
    #     2/log(2.0) *  order     < eta_inv
    # eta_inv > 2.88 * order.
    # We don't want to get too close to the point of instability, though,
    # so we'll require that eta_inv > 3 * order.
    assert(eta ** (-2 * order) < 2.0)


    lpc = ToeplitzLpcEstimator(lpc_order=order, eta=eta, dtype=dtype)

    pred_sumsq_tot = 0.0
    raw_sumsq_tot = 0.0

    for t in range(0, T, block_size):
        t_end = min(t + block_size, T)

        residual = compute_residual(
            array, t, t_end, lpc.get_current_estimate())

        raw_sumsq = np.dot(array[t:t_end], array[t:t_end])
        pred_sumsq = np.dot(residual, residual)

        pred_sumsq_tot += pred_sumsq
        raw_sumsq_tot += raw_sumsq

        lpc.accept_block(array[t:t_end].copy(), residual=residual)


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

def test_new_lpc_compare():
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

    lpc = ToeplitzLpcEstimator(N, eta, dtype=dtype)

    # two blocks.
    lpc.accept_block(signal[:5])

    print("autocorr[1] is: ", stats.autocorr)
    print("autocorr_reflected[1] is: ", stats.get_autocorr_reflected())
    print("x[1] is: ", stats.x)


    autocorr_r = stats.get_autocorr_reflected(aux_order)
    print("autocorr-reflected[1] is: ", autocorr_r)
    print("x[1] is ", solver.estimate(A[1:,1:], A[0,1:], autocorr_r[:-1]))

    stats.accept_block(signal[5:])

    print("autocorr[2] is: ", stats.autocorr)
    print("x[2] is: ", stats.x)
    autocorr_r = stats.get_autocorr_reflected(aux_order)
    print("autocorr-reflected[2] is: ", autocorr_r)

    stats._get_A_minus(aux_order)
    print("A'^-[2] is: ", stats.A_minus[aux_order])
    print("A^+[2] is: ", stats._get_A_plus(aux_order))
    print("A^all[2] is: ", stats._get_A_all(aux_order))
    A = stats.get_A()
    print("A[2] is: ", A)
    print("x[2] is ", solver.estimate(A[1:,1:], A[0,1:], autocorr_r[:-1]))



test_toeplitz_solve()
test_toeplitz_solve_compare()
test_get_toeplitz_mat()

for file in fileList:
    audioArray = waveRead(file, settings["sample-rate"])
    test_prediction(audioArray[:, 0])

