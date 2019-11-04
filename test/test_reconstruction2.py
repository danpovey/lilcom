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


def conj_optim(cur_coeffs, quad_mat, autocorr_stats,
               num_iters=2):
    # Note: this modifies cur_coeffs in place.
    #  Uses conjugate gradient method to minimize the function
    #  cur_coeffs^T quad_mat cur_coeffs, subject to the constraint
    #  that cur_coeffs[0] == -1.    quad_mat is symmetric.
    #  If we define A = quad_mat[1:,1:] and b = quad_mat[0,1:],
    #  then we can say we are minimizing
    #   objf = x^T A x  - 2 x b
    #  dobjf/dx = 2Ax - 2b = 0, so we are solving Ax = b.

    b = quad_mat[0,1:]
    A = quad_mat[1:,1:]

    w, v = np.linalg.eig(A)
    print("Eigs of A are {}".format(w))


    ## we are solving Ax = b.  Trivial solution is: x = A^{-1} b
    x = cur_coeffs[1:]

    if True:
        exact_x = np.dot(np.linalg.inv(A), b)
        print("Exact objf is {}".format(np.dot(np.dot(A,exact_x),exact_x) - 2.0 * np.dot(exact_x,b)))
        #cur_coeffs[1:]  = exact_x
        #return
    x = cur_coeffs[1:].astype(np.float64)
    A = A.astype(np.float64)
    b = b.astype(np.float64)


    if True:
        Minv = get_autocorr_preconditioner(autocorr_stats)
    else:
        Minv = np.eye(autocorr_stats.shape[0] - 1)

    r = b - np.dot(A, x)
    z = np.dot(Minv, r)
    p = z.copy()
    rsold = np.dot(r,z)
    rs_orig = rsold
    print("Residual0 is {}, objf0 is {}".format(rsold,
                                                np.dot(np.dot(A,x),x) - 2.0 * np.dot(x,b)))

    for iter in range(num_iters):
        Ap = np.dot(A, p)
        alpha = rsold / np.dot(p, Ap)
        foo = x + alpha * p
        x += alpha * p
        r -= alpha * Ap;
        z = np.dot(Minv, r)
        rsnew = np.dot(r, z)
        print("ResidualN is {}, ratio={}, objf={} ".format(rsnew, rsnew / rs_orig,
              (np.dot(np.dot(A,x),x) - 2.0 * np.dot(x,b))))
        if rsnew / rs_orig < 1.0e-05:
            break
        p = z + (p * (rsnew / rsold))
        rsold = rsnew
    cur_coeffs[1:] = x


def conj_optim2(cur_coeffs, quad_mat, num_iters=2):
    # This version of conj_optim tests whether we can use a Toeplitz approximation
    # for the quadratic matrix and still get good prediction accuracy.
    # Answer seems to be no.

    b = quad_mat[0,1:]
    A = quad_mat[1:,1:]

    orderp1 = quad_mat.shape[0]
    approx_autocorr = np.zeros(orderp1)
    count = np.zeros(orderp1)
    for i in range(orderp1):
        for j in range(i, orderp1):
            index = j - i
            count[index] += 1
            approx_autocorr[index] += quad_mat[i,j]
    approx_autocorr /= count
    approx_quad_mat = np.zeros((orderp1, orderp1))
    for i in range(orderp1):
        for j in range(orderp1):
            approx_quad_mat[i,j] = approx_autocorr[abs(i-j)]


    approx_b = approx_quad_mat[0,1:]
    approx_A = approx_quad_mat[1:,1:]

    w, v = np.linalg.eig(A)
    print("Eigs of A are {}".format(w))
    w, v = np.linalg.eig(approx_A)
    print("Eigs of approx-A are {}".format(w))


    ## we are solving Ax = b.  Trivial solution is: x = A^{-1} b
    x = cur_coeffs[1:]

    if True:
        exact_x = np.dot(np.linalg.inv(A), b)
        approx_x = np.dot(np.linalg.inv(approx_A), approx_b)
        print("Exact objf is {}".format(np.dot(np.dot(A,exact_x),exact_x) - 2.0 * np.dot(exact_x,b)))
        print("Approx objf is {}".format(np.dot(np.dot(A,approx_x),approx_x) - 2.0 * np.dot(approx_x,b)))

        cur_coeffs[1:]  = approx_x
        return


def get_autocorr_preconditioner(autocorr_stats):
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


def update_stats(array, t_start,
                 t_end, quad_mat,
                 autocorr_stats,
                 autocorr_loading,
                 zero_order_term,
                 linear_term, prev_scale):
    """
    Update autocorrelation and quadratic stats.. scale down previous stats by 0 < prev_scale <= 1.
    The aim is for quad_mat to contain the following, where N == order:

       zero_order_term = \sum_{t=N}^{t_end-1} weight(t)
       linear_term = \sum_{t=N}^{t_end-1} weight(t) array[t]
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

    RETURNS (zero_order_term, linear_term)

    """
    orderp1 = quad_mat.shape[0]
    order = orderp1 - 1
    assert t_start >= order

    zero_order_term *= prev_scale
    linear_term *= prev_scale
    quad_mat *= prev_scale

    zero_order_term += t_end - t_start
    linear_term += np.sum(array[t_start:t_end])
    autocorr = np.zeros(orderp1)
    # Get the autocorrelation stats for which the later frame in the product is
    # within the current block.  This includes some cross-block terms.
    for t in range(t_start, t_end):
        for i in range(orderp1):
            autocorr[i] += array[t] * array[t-i]

    if True:

        # Update autocorr_stats, our more-permanent version of the autocorr stats.
        # We need to make sure that these could be the autocorrelation of a plausible
        # signal, which involves some special changes.
        autocorr_stats *= prev_scale
        autocorr_loading *= prev_scale


        reflection = True  # set False to disable reflection of signal at current time.
        if reflection and t_start > order: # subtract the `temporary stats` added in the last
                            # block.
            for i in range(order):
                for j in range(i + 1, order):
                    autocorr_stats[j] -= 0.5 * array[t_start - (j-i)] * array[t_start - 1 - i]
                    autocorr_loading[j] -= 0.5

        # Special case at first block.
        local_t_start = 0 if t_start == order else t_start

        sqrt_scale = math.sqrt(prev_scale)
        for t in range(local_t_start, t_end):
            for i in range(orderp1):
                t_prev = t - i
                if t_prev < 0:
                    continue
                elif t_prev >= t_start:
                    autocorr_stats[i] += array[t] * array[t_prev]
                    autocorr_loading[i] += 1.0
                else:
                    autocorr_stats[i] += array[t] * array[t_prev] * sqrt_scale
                    autocorr_loading[i] += sqrt_scale

        # Add in some temporary stats due to a notional reflection of the signal at time t_end.
        if reflection:
            for i in range(order):
                for j in range(i + 1, order):
                    autocorr_stats[j] += 0.5 * array[t_end - (j-i)] * array[t_end - 1 - i]
                    autocorr_loading[j] += 0.5


    # Add in the autocorrelation stats to quad_mat
    for i in range(orderp1):
        for j in range(orderp1):
            quad_mat[i,j] += autocorr[abs(i-j)]

    #if True:
    #    # This skips the correction terms.
    #    return (zero_order_term, linear_term)

    # Add in some terms that are really from the autocorrelation of the
    # previous block, and which we had previously subtracted / canceled
    # out when processing it.


    if False:
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


    # Now subtract some terms that were included in the autocorrelation stats
    # but which we want to cancel out from quad_mat because they come
    # from eq. (1) with t >= t_end.
    if False:
        # The slower but easier-to-understand version.
        for k in range(order):
            t = t_end + k
            for i in range(k + 1, orderp1):
                for j in range(k + 1, orderp1):
                    quad_mat[i,j] -= array[t-i] * array[t-j]
    else:
        #The optimized version
        for j in range(order):
            local_sum = 0.0
            for i in range(1, orderp1 - j):
                local_sum += array[t_end-i] * array[t_end-i-j]
                quad_mat[i,i+j] -= local_sum


    # Copy upper to lower triangle of quad_mat
    for i in range(orderp1):
        for j in range(i):
            quad_mat[i,j] = quad_mat[j,i]

    return (zero_order_term, linear_term)


def test_prediction(array):
    # Operate on a linear signal.
    assert(len(array.shape) == 1)
    array = array.astype(np.float64)
    order = 16
    orderp1 = order + 1
    T = array.shape[0]
    autocorr = np.zeros(orderp1)

    cur_coeff = np.zeros(orderp1)
    cur_coeff[0] = -1


    quad_mat_stats = np.zeros((orderp1, orderp1))
    zero_order_term = 0.0
    autocorr_stats = np.zeros(orderp1)
    autocorr_loading = np.zeros(orderp1)  # says how much we'd have to modify
                                          # autocorr_stats to account for DC
                                          # bias.
    linear_term = 0.0

    weight = 0.75
    pred_sumsq_tot = 0.0
    raw_sumsq_tot = 0.0
    BLOCK = 32

    for t in range(T):

        print("array[t] = {}".format(array[t]))

        if (t % BLOCK == 0) and t > 0:
            # This block updates quad_mat_stats.
            (zero_order_term, linear_term) = update_stats(array, max(order, t-BLOCK), t,
                                                          quad_mat_stats, autocorr_stats,
                                                          autocorr_loading,
                                                          zero_order_term,
                                                          linear_term, weight)
            # quad_mat is symmetric.
            quad_mat = np.zeros((orderp1, orderp1))
            for i in range(orderp1):
                for j in range(orderp1):
                    quad_mat[i,j] = autocorr[abs(i-j)]

            if True:
                # Now modify for end effects.  We subtract from quad_mat any terms that
                # should not be included because they are at the edges.

                # First, the very beginning of the signal; we give weight zero
                # to times t=0..order-1.
                for t1 in range(order):
                    for i in range(t1+1):
                        for j in range(t1+1):
                            quad_mat[i,j] -= array[t1-i] * array[t1-j]

                for k in range(order):
                    this_t = t + k
                    for i in range(k + 1, orderp1):
                        for j in range(k + 1, orderp1):
                            quad_mat[i,j] -= array[this_t-i] * array[this_t-j]


            if weight == 1.0:
                if not np.array_equal(quad_mat, quad_mat_stats):
                    print("Arrays differ: t={}, quad_mat={}, quad_mat_stats={}".format(
                            t, quad_mat, quad_mat_stats))
            else:
                quad_mat = quad_mat_stats.copy()  # Use the weighted one for prediction

            if True:
                orig_zero_element = quad_mat[0,0]
                # subtract the constant-offset from quad_mat.. for prediction we would
                # include linear_term / zero_order_term as a zeroth-order prediction.
                quad_mat -= linear_term * linear_term / zero_order_term

                # Get corrected autocorr-stats as if the signal had our DC offset applied.
                autocorr_temp = autocorr_stats - ((linear_term/zero_order_term)**2)*autocorr_loading
                conj_optim(cur_coeff, quad_mat,
                           autocorr_temp, 3)
                print("Current residual / unpredicted-residual is (after update): {}".format(
                        np.dot(cur_coeff, np.dot(quad_mat, cur_coeff) / orig_zero_element)))

            raw_sumsq = 0.0
            pred_sumsq = 0.0
            if t+BLOCK > array.shape[0]:
                continue
            for t2 in range(t, t+BLOCK):
                raw_sumsq += array[t2] * array[t2]

                offset = linear_term / zero_order_term
                pred = 0.0
                for i in range(order+1):
                    pred += cur_coeff[i] * (array[t2 - i] - offset)
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
