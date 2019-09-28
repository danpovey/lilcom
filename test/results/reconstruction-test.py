import lilcom
import numpy
import math
import os
from scipy.io import wavfile

def evaluate (inputArray):
    print("DEBUG = ", inputArray.shape)
    outputArray = numpy.ndarray(inputArray.shape, numpy.int8)
    reconstructedArray = numpy.ndarray(inputArray.shape, numpy.int16)

    lilcom.compress(inputArray,out=outputArray)

    lilcom.decompress(outputArray, out=reconstructedArray)


    # for i in range(len(inputArray)):
    #     print (inputArray[i], " - " ,outputArray[i], " - ", reconstructionArray[i])

    mse = 0
    for x, y in numpy.nditer([inputArray, reconstructedArray]):
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

dataAddresses = ["./audio-samples/" + item for item in os.listdir("audio-samples")]
for dirs in dataAddresses:
    files = os.listdir(dirs)

    dirResults = []
    for file in files:
        filename = dirs + "/" + file
        print (filename)

        sampleRate, audioArray = wavfile.read(filename)
        print (audioArray.shape)
        dirResults.append(evaluate(audioArray))
    
    sumResult = [sum([item["psnr"]for item in dirResults]) , sum([item["mse"] for item in dirResults])]
    meanResult = [item / len(dirResults) for item in sumResult]

    print (dirs)
    print (meanResult)

