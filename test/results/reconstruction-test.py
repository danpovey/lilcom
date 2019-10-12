AudioFormats = ["lilcom", "mp3_320", "mp3_256", "mp3_224", "mp3_192", "mp3_160"]

import lilcom
import numpy
import math
import os
from scipy.io import wavfile
import random
import pydub
import pandas

def MSE (arr, reconst):
    result = 0
    for x, y in numpy.nditer([arr, reconst]):
        result += (x - y)**2
    result /= arr.size
    return result

def PSNR (arr, reconst):
    MAXI = 2**16 - 1
    mse = MSE (arr, reconst)
    if mse != 0:
        psnr = 20 * math.log10(MAXI) - 10 * math.log10(mse)
    else:
        psnr = math.inf

    return psnr


def evaluateMP3(filename, bitrate):
    tmpPath = "./ReconstTemp"

    if tmpPath[2:] in os.listdir("./"):
        os.system("rm -dR "+ tmpPath)

    os.system("mkdir "+ tmpPath)

    wavFile = pydub.AudioSegment.from_wav(filename)
    wavFile.export(tmpPath + "/output.mp3", format="mp3", bitrate=bitrate)

    mp3File = pydub.AudioSegment.from_mp3(tmpPath + "/output.mp3")
    mp3File.export(tmpPath + "/reconst.wav", format="wav")

    # os.system("ffmpeg -i "+ filename +" -codec:a libmp3lame -qscale:a 2 " + tmpPath + "/output.mp3")
    # os.system("ffmpeg -i "+ tmpPath + "/output.mp3 " + tmpPath + "/reconst.wav")

    sampleRate, audioArray = wavfile.read(filename)
    sampleRateReconst, audioReconst = wavfile.read(tmpPath + "/reconst.wav")

    psnr = PSNR(audioArray, audioReconst)

    os.system("rm -dR "+ tmpPath)
    return psnr


def evaluateLilcom(filename, lpc = 4):

    sampleRate, inputArray = wavfile.read(filename)

    outputShape = list(inputArray.shape)

    outputShape[0] += 4
    outputShape = tuple(outputShape)

    outputArray = numpy.ndarray(outputShape, numpy.int8)
    reconstructedArray = numpy.ndarray(inputArray.shape, numpy.int16)

    lilcom.compress(inputArray, out=outputArray, lpc_order = lpc, axis = 0)
    lilcom.decompress(outputArray, out=reconstructedArray, axis = 0)

    psnr = PSNR(inputArray, reconstructedArray)
    return psnr



# Goes through the test folder and find test scenarios
psnrTestResult = []
lpcTestResult = []

dataAddresses = ["./audio-samples/" + item for item in os.listdir("audio-samples")]
if "./audio-samples/.DS_Store" in dataAddresses:
        dataAddresses.remove("./audio-samples/.DS_Store")
for dirs in dataAddresses:
    files = os.listdir(dirs)
    if ".DS_Store" in files:
        files.remove(".DS_Store")

    scenarioPSNR = []
    scenarioLPC = []

    print ("Test scenraio: ", dirs)
#     dirResults = []
    for file in files:
        filename = dirs + "/" + file

        trackLPCPSNR = [file]

        resLilcom = evaluateLilcom(filename)
        resMp3_320 = evaluateMP3(filename, "320k")
        resMp3_256 = evaluateMP3(filename, "256k")
        resMp3_224 = evaluateMP3(filename, "224k")
        resMp3_192 = evaluateMP3(filename, "192k")
        resMp3_160 = evaluateMP3(filename, "160k")

        trackPSNR = [file, resLilcom, resMp3_320, resMp3_256,\
            resMp3_224, resMp3_192, resMp3_160]

        print(trackPSNR)
        scenarioPSNR.append(trackPSNR)

        for lpc in range (0 , 14):
            trackLPCPSNR.append(evaluateLilcom(filename, lpc))
        print (trackLPCPSNR)
        scenarioLPC.append(trackLPCPSNR)


    scenarioPSNR = pandas.DataFrame(data= scenarioPSNR, columns= ["file"]+AudioFormats)
    scenarioPSNR.to_csv(dirs[16:]+"-Scenario.csv")
    scenarioLPC = pandas.DataFrame(data= scenarioLPC, columns= ["file"]+list(range(0 , 16)))
    scenarioLPC.to_csv(dirs[16:]+"-LPC.csv")

    psnrTestResult.append(scenarioPSNR)


