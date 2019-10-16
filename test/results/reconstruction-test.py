import lilcom   # The target module
import numpy  as np  # To manipulate arrays
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

dataSetDirectory = "./audio-samples/temp"


def PSNR (originalArray, reconstructedArray):
    """ This dunction calculates the peak signal to noise ratio between a signal
    and its reconstruction

       Args:
        originalArray:   A numpy array which should be the original array before compres-
                          -sion and reconstruction.
        reconstructedArray: A numpy array which in this case should be an array which is
                            reconstructed from the compression function.
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
    reconstructedArray *= np.sqrt((originalArray ** 2).sum()  /
                                  (reconstructedArray ** 2).sum())


    max_value = float(np.max(np.abs(originalArray)))
    mean_square_error = ((originalArray - reconstructedArray) ** 2).sum() / originalArray.size
    if mean_square_error != 0:
        psnr = 20 * math.log10(max_value) - 10 * math.log10(mean_square_error)
    else:
        psnr = math.inf

    return psnr


def evaluateMP3(filename, audioArray, bitrate):

    # Creating a temporary path for MP3 and reconstruction File
    tmpPath = "./ReconstTemp"
    if tmpPath[2:] in os.listdir("./"):
        os.system("rm -dR "+ tmpPath)
    os.system("mkdir "+ tmpPath)
    wavFile = pydub.AudioSegment.from_wav(filename)
    wavFile.export(tmpPath + "/output.mp3", format="mp3", bitrate=bitrate)
    print("At bitrate {}, file {} compresses to {} bytes".format(
        bitrate, filename, os.path.getsize(tmpPath + "/output.mp3")))
    mp3File = pydub.AudioSegment.from_mp3(tmpPath + "/output.mp3")
    mp3File.export(tmpPath + "/reconst.wav", format="wav")
    sampleRateReconst, audioReconst = wavfile.read(tmpPath + "/reconst.wav")

    psnr = PSNR(audioArray, audioReconst)

    os.system("rm -dR "+ tmpPath)
    return psnr


def evaluateLilcom(audioArray, lpc = 4):
    audioArray =  audioArray.astype(np.float32)
    outputShape = list(audioArray.shape)

    outputShape[0] += 4
    outputShape = tuple(outputShape)

    outputArray = np.ndarray(outputShape, np.int8)
    reconstructedArray = np.ndarray(audioArray.shape, np.int16)

    lilcom.compress(audioArray, out=outputArray, lpc_order = lpc, axis = 0)
    reconstructedArray = lilcom.decompress(outputArray, axis = 0, dtype=audioArray.dtype)

    psnr = PSNR(audioArray, reconstructedArray)
    return psnr

# Empty lists for test results
psnrComparisonResults = []
psnrLpcResults = []

# Fetching sample files
testFiles = os.listdir(dataSetDirectory)

print("TestFiles is ", testFiles)
if ".DS_Store" in testFiles:
    testFiles.remove(".DS_Store")

# Resampler
for testFile in testFiles:
    filePath = dataSetDirectory + "/" + testFile
    sampleRate , audioArray =  wavfile.read(filePath)
    print ("Subsampling " + filePath)
    for sr in [16000, 8000]:
        if audioArray.dtype == np.int16:
            audioArray = audioArray.astype(np.float32) / 32768
        downsampledArray = librosa.core.resample(audioArray.transpose(),
                                                 sampleRate, sr).transpose()
        wavfile.write(dataSetDirectory + "/subsampled-"+str(sr)+testFile,
                    sr, downsampledArray)
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

# Remove
os.system("rm "+dataSetDirectory+"/subsampled*")
