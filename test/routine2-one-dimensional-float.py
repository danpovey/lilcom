# Signals are supposed to be given as numpy arrays
import numpy
# The main library
import lilcom_c_extension as lilcom
# To create random samples for the signal
import random

n_samples = 1000

inputArray = numpy.array([random.uniform(0, 1)
                         for i in range(n_samples)]).astype(numpy.float32)
outputArray = numpy.zeros((inputArray.shape[0]+4), numpy.int8)
reconstruction = numpy.zeros(inputArray.shape, numpy.float32)

lilcom.compress_float(inputArray, outputArray, lpc_order=10)

c_exponent = lilcom.decompress_float(outputArray, reconstruction)

for i in range(n_samples):
    print("Sample no ", i, "original number = ", inputArray[i],
          " compressed = ", outputArray[i], " reconstructed number = ",
          reconstruction[i])

print("conversion exponent = ", c_exponent)
