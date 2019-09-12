import lilcom_c_extension
import numpy
import math
import os
from scipy.io import wavfile

def evaluate (inputArray):
    outputArray = numpy.ndarray(inputArray.shape, numpy.int8)
    reconstructionArray = numpy.ndarray(inputArray.shape, numpy.int16)

    lilcom_c_extension.compress_int16(inputArray, outputArray, 0, 1)

    lilcom_c_extension.decompress_int16(outputArray, reconstructionArray)


    # for i in range(len(inputArray)):
    #     print (inputArray[i], " - " ,outputArray[i], " - ", reconstructionArray[i])

    mse = 0
    for x, y in numpy.nditer([inputArray, reconstructionArray]):
        # print (x, " : ", y)
        mse += (x - y)**2
    mse /= inputArray.size

    maxi = 2**16 - 1

    psnr = 20 * math.log10(maxi) - 10 * math.log10(mse)

    # print (mse)
    # print (psnr)

    result = dict.fromkeys("mse", "psnr")
    result["mse"] = mse
    result["psnr"] = psnr
    
    return result

import random

dataAddresses = ["./samples/" + item for item in os.listdir("samples")]
for dirs in dataAddresses:
    files = os.listdir(dirs)

    dirResults = []
    for file in files:
        filename = dirs + "/" + file
        print (filename)

        sampleRate, audioArray = wavfile.read(filename)
        dirResults.append(evaluate(audioArray))
    
    sumResult = [sum([item["psnr"]for item in dirResults]) , sum([item["mse"] for item in dirResults])]
    meanResult = [item / len(dirResults) for item in sumResult]

    print (dirs)
    print (meanResult)

