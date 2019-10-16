import lilcom   # The target module
import numpy    # To manipulate arrays
import math     # Mathemtical functions i.e. logarithm
import os       # For directory managements i.e. ls, dir, etc
from scipy.io import wavfile    # To read and write wavefiles
import random
import pydub    # For conversion between audio formats
import pandas   # For using dataframes
import librosa  # For downsampling

# Audio formats for compaison
AudioFormats = ["lilcom",
                "mp3_320",
                "mp3_256",
                "mp3_224",
                "mp3_192",
                "mp3_160"]

MP3BitRates = [320, 256, 224, 192, 160]

dataSetDirectory = "./audio-samples/tiny-samples"


def MSE(originalArray, reconstructedArray):
    """ This dunction calculates the mean square error between a signal and its recons-
    -truction

       Args:
        originalArray:   A numpy array which should be the original array before compres-
                          -sion and reconstruction.
        reconstructedArray: A numpy array which in this case should be an array which is
                            reconstructed from the compression function.  
       
       Returns:
           A rational number which is the result of mean square error between two given 
           arrays
   """
    #initial value
    result = 0
    # iterating on both arrays
    for x, y in numpy.nditer([originalArray, reconstructedArray]):
        result += (x - y)**2
    result /= originalArray.size
    return result


def PSNR (originalArray, reconstructedArray, quantizationLevel = 16):
    """ This dunction calculates the peak signal to noise ratio between a signal 
    and its reconstruction

       Args:
        originalArray:   A numpy array which should be the original array before compres-
                          -sion and reconstruction.
        reconstructedArray: A numpy array which in this case should be an array which is
                            reconstructed from the compression function.  
        quantizationLevel:  The level of quantization which an audio is supposed to be in
                            By default it is supposed to be 16
       
       Returns:
           A rational number which is the result of psnr between two given 
    """
    
    MAXI = 2**quantizationLevel - 1
    mse = MSE (originalArray, reconstructedArray)
    if mse != 0:
        psnr = 20 * math.log10(MAXI) - 10 * math.log10(mse)
    else:
        psnr = math.inf

    return psnr


def evaluateMP3(filename, audioArray,bitrate):
    
    # Creating a temporary path for MP3 and reconstruction File
    tmpPath = "./ReconstTemp"
    if tmpPath[2:] in os.listdir("./"):
        os.system("rm -dR "+ tmpPath)
    os.system("mkdir "+ tmpPath)
    wavFile = pydub.AudioSegment.from_wav(filename)
    wavFile.export(tmpPath + "/output.mp3", format="mp3", bitrate=bitrate)
    mp3File = pydub.AudioSegment.from_mp3(tmpPath + "/output.mp3")
    mp3File.export(tmpPath + "/reconst.wav", format="wav")
    sampleRateReconst, audioReconst = wavfile.read(tmpPath + "/reconst.wav")
    # print (audioArray.shape)
    # print (audioReconst.shape)

    psnr = PSNR(audioArray, audioReconst)

    os.system("rm -dR "+ tmpPath)
    return psnr


def evaluateLilcom(audioArray, lpc = 4):

    outputShape = list(audioArray.shape)

    outputShape[0] += 4
    outputShape = tuple(outputShape)

    outputArray = numpy.ndarray(outputShape, numpy.int8)
    reconstructedArray = numpy.ndarray(audioArray.shape, numpy.int16)

    lilcom.compress(audioArray, out=outputArray, lpc_order = lpc, axis = 0)
    lilcom.decompress(outputArray, out=reconstructedArray, axis = 0)

    psnr = PSNR(audioArray, reconstructedArray)
    return psnr

# Empty lists for test results
psnrComparisonResults = []
psnrLpcResults = []

# Fetching sample files
testFiles = os.listdir(dataSetDirectory)
if ".DS_Store" in testFiles:
    testFiles.remove(".DS_Store")

# Resampler
for testFile in testFiles:
    filePath = dataSetDirectory + "/" + testFile
    sampleRate , audioArray =  wavfile.read(filePath)
    print ("Subsampling " + filePath)
    for sr in [16000, 8000]:
        wavfile.write(dataSetDirectory + "/subsampled-"+str(sr)+testFile,
                    sr, audioArray)
        print (dataSetDirectory + "/subsampled-"+str(sr)+testFile)
    
# renewing sample files
testFiles = os.listdir(dataSetDirectory)
if ".DS_Store" in testFiles:
    testFiles.remove(".DS_Store")

for testFile in testFiles:
    filePath = dataSetDirectory + "/" + testFile
    
    # Just for debug
    print(filePath)

    
    
    # Reading audio file
    sampleRate, audioArray = wavfile.read(filePath)

    # Convertying audio file sampling rates needed and 
    #   evaluating it 
        
    # PSNR for MP3 vs Lilcom
    entityPsnrComparisonResults = [testFile, sampleRate, 
                                    evaluateLilcom(audioArray),
                                    evaluateMP3(filePath, audioArray, "320k"),
                                    evaluateMP3(filePath, audioArray, "256k"),
                                    evaluateMP3(filePath, audioArray, "224k"),
                                    evaluateMP3(filePath, audioArray, "192k"),
                                    evaluateMP3(filePath, audioArray, "160k")]
    psnrComparisonResults.append(entityPsnrComparisonResults)
    
    # PSNR for comparison among LPCs
    entityPsnrLpcResults = [testFile, sampleRate]+\
            [evaluateLilcom(audioArray, l) for l in range(0, 14)]
    psnrLpcResults.append(entityPsnrLpcResults)

    # Just for Debuging
    print(entityPsnrComparisonResults)
    print(entityPsnrLpcResults)

# Dataframe related works

comparisonDataFrame = pandas.DataFrame(psnrComparisonResults)
LpcDataFrame = pandas.DataFrame(psnrLpcResults)

comparisonDataFrame.to_csv("Comparison.csv")
LpcDataFrame.to_csv("Lpc.csv")

# Removing Subsampled Data
os.system("rm "+dataSetDirectory+"/subsampled*")
