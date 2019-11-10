#!/usr/bin/python3
"""
Details about the Doument:
This script runs a test to compare lilcom's reconstruction accuracy
    and compare it with known MP3 bitrates. The script accepts additional
    arguments which by running `./test_reconstruction.py --help` all
    arguments are documented.
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


def PSNR(originalArray, reconstructedArray):
    """ This function calculates the peak signal to noise ratio between a
    signal and its reconstruction

       Args:
        originalArray: A numpy array which should be the original array
        before compression and reconstruction.
        reconstructedArray: A numpy array which in this case should be an
        array which is reconstructed from the compression function.
       Returns:
           A rational number which is the result of psnr between two given
    """
    # Convert both to float.
    if originalArray.dtype == np.int16:
        originalArray = originalArray.astype(np.float32) / 32768
    if reconstructedArray.dtype == np.int16:
        reconstructedArray = reconstructedArray.astype(np.float32) / 32768

    # Correct for any differences in dynamic range, which might be caused
    # by attempts of compression libraries like mp3 to avoid overflow.
    reconstructedArray *= np.sqrt((originalArray ** 2).sum() /
                                  (reconstructedArray ** 2).sum())

    max_value = float(np.max(np.abs(originalArray)))
    mean_square_error = (((originalArray - reconstructedArray) ** 2).sum() /
                         originalArray.size)
    if mean_square_error != 0:
        psnr = 20 * math.log10(max_value) - 10 * math.log10(mean_square_error)
    else:
        psnr = math.inf

    return psnr


def toeplitz_solve(autocorr, y):
    """
    Let y be a vector of dimension N and let
    `autocorr` be vector of dimension N representing, conceptually,
    a Toeplitz matrix A(i,j) = autocorr[abs(i-j)].

      This function solves the linear system A x = b, returning x.

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

    if order is None:
        order = quad_mat.shape[0] - 1
    if order != quad_mat.shape[0] - 1:
        conj_optim(cur_coeffs[0:order+1], quad_mat[0:order+1, 0:order+1],
                   autocorr_stats[0:order+1], num_iters, dtype=dtype)
        return



    abs_smoothing = 1.0e-10
    if False:  #proportional_smoothing != 0.0 or abs_smoothing != 0.0:
        dim = quad_mat.shape[0]
        quad_mat = quad_mat + np.eye(dim, dtype=dtype) * (abs_smoothing + proportional_smoothing * quad_mat.trace())
        autocorr_stats = autocorr_stats.copy().astype(dtype)
        autocorr_stats[0] += abs_smoothing + proportional_smoothing * autocorr_stats[0]


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


    if True:
        # use preconditioner
        M = get_autocorr_matrix(autocorr_stats, dtype=np.float64)
        Minv = np.linalg.inv(M)
        abs_smoothing = 1.0e-10
        dim = quad_mat.shape[0] - 1
        A += proportional_smoothing * M + abs_smoothing * np.eye(dim, dtype=dtype)

        w, v = np.linalg.eig(Minv)
        if not w.min() > 0.0:
            print("Eigs of Minv are {}".format(w))
            sys.exit(1)
    else:
        Minv = np.eye(autocorr_stats.shape[0] - 1)

    if False:
        # This block would use autocorrelation-based LPC.
        x = np.dot(Minv, b)
        cur_coeffs[1:] = x
        return

    r = b - np.dot(A, x)
    z = toeplitz_solve(autocorr_stats[:-1], r)
    assert np.dot(z, r) >= 0.0
    if False:
        z_test = np.dot(Minv, r)
        print("z = {}, z_test = {}".format(z, z_test))

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
        #z = np.dot(Minv, r)
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


def get_autocorr_matrix(autocorr_stats, dtype = np.float64):
    """
    Returns a matrix of dimension N-1 by N-1 if autocorr_stats.shape[0] == N.
    """
    order = autocorr_stats.shape[0] - 1
    A = np.zeros((order, order), dtype=dtype)
    for i in range(order):
        for j in range(order):
            A[i,j] = autocorr_stats[abs(i-j)]
    return A


def get_autocorr_preconditioner(autocorr_stats, dtype = np.float64):
    """
    Returns the preconditioner implied by these autocorrelation stats.
    We're using the inverse of the Toeplitz matrix.
    """
    order = autocorr_stats.shape[0] - 1
    A = np.zeros((order, order))
    for i in range(order):
        for j in range(order):
            A[i,j] = autocorr_stats[abs(i-j)]
    return np.linalg.inv(A)

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
    proportional_smoothing = 1.0e-04
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



# Suppose we are minimizing f(x) = 0.5 x^2.   2nd deriv is 1.
#  x <== x - d/dx f(x)   is:    x <=== x - x = 0.



def hash(array):
    return int(np.sum(np.abs(array))*2000) % int((2**16) - 1)


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

if args.releasedf:
    settings["release-df"] = args.releasedf
    csvOpener = open(settings["release-df"], "w+")
    settings["release-df"] = csvOpener
else:
    settings["release-df"] = None

if (args.releaselog):
    settings["release-log"] = args.releaselog
    fileOpener = open(settings["release-log"], "w+")
    settings["release-log"] = fileOpener

else:
    settings["release-log"] = None


evaulators = [
    {
        "algorithm": "lilcom",
        "additionalParam": 4
    },
    {
        "algorithm": "MP3",
        "additionalParam": "320k"
    },
    {
        "algorithm": "MP3",
        "additionalParam": "256k"
    },
    {
        "algorithm": "MP3",
        "additionalParam": "224k"
    },
    {
        "algorithm": "MP3",
        "additionalParam": "192k"
    },
    {
        "algorithm": "MP3",
        "additionalParam": "160k"
    }
]

fileList = [settings["dataset-dir"] + "/" + item
            for item in os.listdir(settings["dataset-dir"])
            if ".wav" in item]

# Initial prints
logger(logmod="initialization")

for file in fileList:
    audioArray = waveRead(file, settings["sample-rate"])

    print("Shape is {}".format(audioArray.shape))
    test_prediction(audioArray[:, 0])

    fileEvaluationResultList = [os.path.basename(file)]
    for evaluator in evaulators:
        evaluationResult = evaluate(file, audioArray,
                                    evaluator["algorithm"],
                                    evaluator["additionalParam"])
        fileEvaluationResultList.append({"evaluator": evaluator,
                                        "result": evaluationResult})

    logger("result", fileEvaluationResultList)

